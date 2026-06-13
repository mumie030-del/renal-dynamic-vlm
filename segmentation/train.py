##new


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import Data3Dataset
from tqdm import tqdm
from module import UnetWithLTAE
import random, numpy as np
# ==================== 配置参数 ====================
DATA_DIR = '../data3'
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    """固定所有相关随机种子，保证划分和初始化可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, 1, H, W) -> 需要 sigmoid
        # targets: (B, 1, H, W)
        probs = torch.sigmoid(logits)
        
        # 拉平向量
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        
        return 1 - dice.mean()
   
class BceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)

criterion = BceDiceLoss()


def compute_binary_dice(logits, targets, threshold: float = 0.5, eps: float = 1e-5) -> float:
    """根据 logits 和 GT 计算批量 Dice（用于监控，不参与反向传播）"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = (2.0 * intersection + eps) / (
        preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + eps
    )
    return dice.mean().item()


def compute_binary_iou(logits, targets, threshold: float = 0.5, eps: float = 1e-5) -> float:
    """根据 logits 和 GT 计算批量 IoU（Intersection over Union）"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

# ==================== 主训练流程 ====================
if __name__ == '__main__':
    print(f"使用设备: {DEVICE}")

    # 0. 固定随机种子（保证每次运行划分一致）
    set_seed(42)

    # 1. 加载数据集
    dataset = Data3Dataset(data_root=DATA_DIR, target_size=(256, 256), num_channels=26)
    print(f"总数据集大小: {len(dataset)}")
    
    # 2. 划分训练集和验证集 (75% 训练, 25% 验证)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    # 使用固定 generator，保证每次划分出的索引完全一致
    g = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 3. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 4. 初始化模型 (纯净版 U-Net)
    model = UnetWithLTAE(in_channels=26, out_channels=1).to(DEVICE)
    print(f"输入通道数: 26, 输出通道数: 1")
    
    # 测试一个 batch
    sample_img, sample_mask = next(iter(train_loader))
    print(f"图像形状: {sample_img[0].shape}, 掩码形状: {sample_mask[0].shape}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 5. 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        
        for batch_idx, (images, masks) in enumerate(train_bar):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            if epoch % 10 == 0 and batch_idx == 0: # 每10个epoch看一次
                probs = torch.sigmoid(outputs)
                print(f" [Debug] Max Prob: {probs.max().item():.4f}, Mean Prob: {probs.mean().item():.4f}")
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_batches = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]')
            for images, masks in val_bar:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                # 计算本 batch 的 Dice / IoU（阈值与测试脚本保持一致，这里使用 0.5）
                batch_dice = compute_binary_dice(outputs, masks, threshold=0.5)
                batch_iou = compute_binary_iou(outputs, masks, threshold=0.5)
                val_dice += batch_dice
                val_iou += batch_iou
                val_batches += 1

                val_bar.set_postfix({'loss': loss.item(), 'dice': batch_dice, 'iou': batch_iou})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / max(val_batches, 1)
        avg_val_iou = val_iou / max(val_batches, 1)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}]')
        print(f'  训练损失: {avg_train_loss:.4f}')
        print(f'  验证损失: {avg_val_loss:.4f}')
        print(f'  验证 Dice: {avg_val_dice:.4f}')
        print(f'  验证 IoU : {avg_val_iou:.4f}')
        print(f'  学习率: {current_lr:.6f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice,
                'val_iou': avg_val_iou,
            }, checkpoint_path)
            print(f'  ✓ 保存最佳模型到 {checkpoint_path}')
        
        # 每 10 个 epoch 保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice,
                'val_iou': avg_val_iou,
            }, checkpoint_path)
            print(f'  ✓ 保存检查点到 {checkpoint_path}')
        
        print('-' * 60)
    
    print('\n训练完成！')
    print(f'最佳验证损失: {best_val_loss:.4f}')