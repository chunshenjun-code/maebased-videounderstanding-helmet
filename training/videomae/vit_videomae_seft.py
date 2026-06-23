import os
import cv2
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, AutoImageProcessor

# ==================== 配置 ====================
# 定义各个视角的数据根目录（请修改为您的实际路径）
data_domains = {
    "first": "data_str_blur4",   # 第一人称
    "down":  "data2_down_str_blur4",    # 俯视
    "left":  "data2_left_str_blur4",    # 左侧
    "right": "data2_right_str_blur4",   # 右侧
}

class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(class_names)

# 模型参数
num_frames = 16
num_frames_per_clip = 64
frame_height, frame_width = 224, 224

# 分段微调训练参数
batch_size = 12                     # 根据显存调整
learning_rate = 5e-5               # 较小学习率，抑制遗忘
num_epochs_per_domain = 5          # 每个视角微调 2 个 epoch
weight_decay = 0.1
gradient_accumulation_steps = 2
max_grad_norm = 1.0
warmup_ratio = 0.1

# 经验重放配置（缓解遗忘）
use_replay = True                  # 是否混合第一人称数据
replay_ratio = 0.1                 # 每个 batch 中第一人称样本比例（0~1）
replay_buffer_size = 200           # 从第一人称训练集中随机采样的最大样本数

# 早停配置
first_person_drop_threshold = 0.05 # 第一人称准确率下降超过 5% 则停止继续微调
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 模型保存路径
base_model_path = "videomae_best_blur4.pth"   # 已训练好的第一人称模型（若存在）
final_model_save_path = "videomae_seft.pth"

# ==================== 数据集定义（复用您的代码） ====================
class VideoMAEDataset(Dataset):
    def __init__(self, root_dir, class_names, num_frames=16, num_clips=1,
                 num_frames_per_clip=64, processor=None, training=True):
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.num_frames_per_clip = num_frames_per_clip
        self.processor = processor
        self.training = training
        self.samples = []

        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(class_dir, fname)
                    self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if len(frames) == 0:
            dummy = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
            inputs = self.processor(images=dummy, return_tensors="np")
            pixel_values = torch.from_numpy(inputs.pixel_values).squeeze(0)
            return pixel_values, label

        total_frames = len(frames)
        if total_frames < self.num_frames_per_clip:
            frames = frames + [frames[-1]] * (self.num_frames_per_clip - total_frames)
            total_frames = self.num_frames_per_clip

        if self.training:
            start = np.random.randint(0, total_frames - self.num_frames_per_clip + 1)
        else:
            start = (total_frames - self.num_frames_per_clip) // 2

        clip = frames[start:start + self.num_frames_per_clip]
        indices = np.linspace(0, self.num_frames_per_clip - 1, self.num_frames, dtype=int)
        sampled_frames = [clip[i] for i in indices]

        inputs = self.processor(images=sampled_frames, return_tensors="np")
        pixel_values = torch.from_numpy(inputs.pixel_values).squeeze(0)
        return pixel_values, label


# ==================== 辅助函数 ====================
def create_model_and_processor(num_classes):
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model, processor


def get_dataloaders(domain_path, processor, batch_size, training=True, shuffle=True):
    """为单个域创建 DataLoader（仅训练集或验证/测试集）"""
    dataset = VideoMAEDataset(
        root_dir=domain_path,
        class_names=class_names,
        num_frames=num_frames,
        num_frames_per_clip=num_frames_per_clip,
        processor=processor,
        training=training
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=2, pin_memory=True)
    return loader


def evaluate(model, dataloader, criterion, device, domain_name=""):
    """评估模型在某个数据加载器上的损失和准确率"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"评估 {domain_name}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / total if total > 0 else 0
    acc = correct / total if total > 0 else 0
    return avg_loss, acc


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    scheduler=None, gradient_accumulation_steps=1, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for step, (inputs, labels) in enumerate(tqdm(dataloader, desc="训练", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def create_replay_dataloader(first_domain_train_loader, replay_ratio, batch_size, processor):
    """
    从第一人称训练集中随机采样 replay_buffer_size 个样本，构建一个固定大小的 replay 数据集。
    实际使用时，可以动态混合：每个 batch 中直接从 first_loader 采样一部分样本。
    为简化，这里返回一个可以按比例混合的 Sampler，或是直接构建一个新的 DataLoader。
    更高效的方法：在训练循环中手动混合两个 loader 的 batch。
    下面实现一个自定义的 Iterator，每次产生一个 batch，其中包含 replay_ratio 比例的第一人称数据。
    """
    # 将第一人称训练集的所有样本预先提取出来（注意：数据量可能很大，这里只取缓冲大小）
    first_samples = []
    first_loader_iter = iter(first_domain_train_loader)
    while len(first_samples) < replay_buffer_size:
        try:
            batch_inputs, batch_labels = next(first_loader_iter)
            for i in range(batch_inputs.size(0)):
                first_samples.append((batch_inputs[i], batch_labels[i]))
        except StopIteration:
            break
    # 如果不足 replay_buffer_size，则重复采样
    if len(first_samples) < replay_buffer_size:
        first_samples = first_samples * (replay_buffer_size // len(first_samples) + 1)
    first_samples = first_samples[:replay_buffer_size]

    class ReplayDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    replay_dataset = ReplayDataset(first_samples)
    replay_loader = DataLoader(replay_dataset, batch_size=int(batch_size * replay_ratio),
                               shuffle=True, drop_last=True)
    return replay_loader


def train_on_domain(model, train_loader, val_loaders_dict, criterion, optimizer, scheduler,
                    device, domain_name, num_epochs, replay_loader=None):
    """
    在某个域上微调模型，每个 epoch 结束后评估所有域的验证集。
    若第一人称验证准确率下降超过阈值，则停止训练并回滚最佳模型。
    """
    best_model_state = copy.deepcopy(model.state_dict())
    best_first_acc = 0.0
    patience = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n  --- {domain_name} Epoch {epoch}/{num_epochs} ---")
        # 训练：如果提供了 replay_loader，则与当前域的数据混合
        if replay_loader is not None:
            # 混合训练：交替取 current loader 和 replay loader 的 batch
            # 简单实现：将两个 loader 组合成一个新的迭代器
            from itertools import cycle
            current_iter = iter(train_loader)
            replay_iter = iter(cycle(replay_loader)) if len(replay_loader) > 0 else None
            combined_loader = []
            # 构建一个临时列表，每个 step 取一个 current batch 和一个 replay batch
            # 实际操作中可以直接在训练循环中混合
            # 这里为了复用 train_one_epoch，我们创建一个新的生成器
            class MixedLoader:
                def __init__(self, main_loader, replay_loader, replay_ratio):
                    self.main_loader = main_loader
                    self.replay_loader = replay_loader
                    self.replay_ratio = replay_ratio
                    self.main_iter = iter(main_loader)
                    self.replay_iter = iter(cycle(replay_loader)) if replay_loader else None

                def __iter__(self):
                    return self

                def __next__(self):
                    try:
                        main_batch = next(self.main_iter)
                    except StopIteration:
                        raise StopIteration
                    # 如果使用 replay，则从 replay 中取一个 batch 并拼接
                    if self.replay_iter and np.random.rand() < self.replay_ratio:
                        replay_batch = next(self.replay_iter)
                        # 拼接像素值和标签
                        mixed_inputs = torch.cat([main_batch[0], replay_batch[0]], dim=0)
                        mixed_labels = torch.cat([main_batch[1], replay_batch[1]], dim=0)
                        return mixed_inputs, mixed_labels
                    else:
                        return main_batch

            mixed_loader = MixedLoader(train_loader, replay_loader, replay_ratio)
            # 注意：MixedLoader 需要实现 __len__，但 train_one_epoch 中未使用 len，只迭代即可
            train_loss, train_acc = train_one_epoch(
                model, mixed_loader, criterion, optimizer, device,
                scheduler=None,  # 可选，为简化不设置 scheduler
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                scheduler=None,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm
            )
        print(f"    训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        # 评估所有域
        print("    评估各域验证集:")
        for eval_name, val_loader in val_loaders_dict.items():
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, eval_name)
            print(f"      {eval_name} 验证集: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            if eval_name == "first":
                current_first_acc = val_acc
                if current_first_acc > best_first_acc:
                    best_first_acc = current_first_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                # 检查下降
                if epoch > 1 and current_first_acc < best_first_acc - first_person_drop_threshold:
                    print(f"    [早停] 第一人称准确率下降超过 {first_person_drop_threshold*100}%，停止本域微调")
                    model.load_state_dict(best_model_state)
                    return  # 提前结束该域训练

        # 每个 epoch 后调整学习率（可选）
        if scheduler:
            scheduler.step()

    # 恢复最佳模型（保留第一人称准确率最高的状态）
    model.load_state_dict(best_model_state)


# ==================== 主程序 ====================
def main():
    # 1. 创建基础模型和 processor
    print("加载 VideoMAE 基础模型...")
    model, processor = create_model_and_processor(num_classes)
    model = model.to(device)

    # 2. 如果已有预训练的第一人称模型权重，加载之
    if os.path.exists(base_model_path):
        print(f"加载已有第一人称预训练模型: {base_model_path}")
        model.load_state_dict(torch.load(base_model_path, map_location=device))

    # 3. 为每个域创建训练、验证、测试 DataLoader
    # 假设每个域文件夹下都有 train / val / test 子目录
    # 若您的结构不同，请相应修改
    domain_loaders = {}
    for domain_name, domain_root in data_domains.items():
        train_dir = os.path.join(domain_root, "train")
        val_dir = os.path.join(domain_root, "val")
        test_dir = os.path.join(domain_root, "test")
        # 仅当目录存在时创建
        if not os.path.exists(train_dir):
            print(f"警告: {train_dir} 不存在，跳过 {domain_name}")
            continue
        domain_loaders[domain_name] = {
            "train": get_dataloaders(train_dir, processor, batch_size, training=True, shuffle=True),
            "val": get_dataloaders(val_dir, processor, batch_size, training=False, shuffle=False),
            "test": get_dataloaders(test_dir, processor, batch_size, training=False, shuffle=False),
        }

    if "first" not in domain_loaders:
        raise ValueError("必须提供第一人称数据域 (first)！")

    # 4. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 5. 分段微调顺序（按您希望的顺序，例如 first -> down -> left -> right）
    #    注意：first 本身已经微调过，可以跳过，直接从 down 开始；如果需要重新微调 first 也可包括。
    #    这里我们从 down 开始，依次在 new 域上微调，同时使用第一人称经验重放。
    domain_order = ["down", "left", "right"]   # 依次在这些新视角上微调

    # 准备第一人称 replay loader（如果启用）
    first_train_loader = domain_loaders["first"]["train"]
    replay_loader = None
    if use_replay:
        print(f"构建经验重放缓冲区 (大小={replay_buffer_size}, 混合比例={replay_ratio})")
        replay_loader = create_replay_dataloader(first_train_loader, replay_ratio, batch_size, processor)

    # 每个阶段构建验证集字典（包含所有域的验证集，用于监控第一人称性能）
    val_loaders_all = {name: info["val"] for name, info in domain_loaders.items()}

    # 依次微调
    for domain_name in domain_order:
        if domain_name not in domain_loaders:
            print(f"跳过 {domain_name}，未找到数据")
            continue
        print(f"\n========== 开始微调域: {domain_name} ==========")
        train_loader = domain_loaders[domain_name]["train"]
        # 优化器每次都重新创建（使用当前学习率）
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # 可选：学习率调度器（余弦退火等），这里简单使用固定学习率
        total_steps = len(train_loader) * num_epochs_per_domain // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-7, end_factor=1.0, total_iters=warmup_steps)

        train_on_domain(
            model=model,
            train_loader=train_loader,
            val_loaders_dict=val_loaders_all,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            domain_name=domain_name,
            num_epochs=num_epochs_per_domain,
            replay_loader=replay_loader  # 传入 replay loader 进行混合训练
        )
        # 每个阶段结束保存一次模型
        torch.save(model.state_dict(), f"{domain_name}_stage_model.pth")
        print(f"完成 {domain_name} 阶段微调，模型已保存")

    # 6. 最终评估所有域的测试集
    print("\n========== 最终评估（测试集） ==========")
    final_val_loaders = {name: info["test"] for name, info in domain_loaders.items()}
    for domain_name, test_loader in final_val_loaders.items():
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, domain_name)
        print(f"{domain_name} 测试集: Loss={test_loss:.4f}, Acc={test_acc:.4f}")

    # 7. 保存最终模型
    torch.save(model.state_dict(), final_model_save_path)
    print(f"最终模型已保存至 {final_model_save_path}")

    # 8. 可选：输出详细的分类报告
    for domain_name, test_loader in final_val_loaders.items():
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(pixel_values=inputs).logits
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        print(f"\n{domain_name} 分类报告:")
        print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    main()
