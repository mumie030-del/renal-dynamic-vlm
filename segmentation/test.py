##new
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from datasets import Data3Dataset

from tqdm import tqdm
from scipy.ndimage import label, center_of_mass
from module import UnetWithLTAE
# ==================== 配置参数 ====================
DATA_DIR = '../data3'
CHECKPOINT_PATH = './checkpoints/best_model.pth'
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './test_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== 评估函数 ====================
def calculate_metric_dice(pred, target, smooth=1e-5):
    """计算二值化后的精确 Dice 分数"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

# ==================== 主测试流程 ====================
if __name__ == '__main__':
    print(f"使用设备: {DEVICE}")
    
    # 1. 加载数据集
    dataset = Data3Dataset(data_root=DATA_DIR, target_size=(256, 256), num_channels=26)
    
    # 抽取最后 20 个样本进行测试
    test_size = min(20, len(dataset))
    test_indices = list(range(len(dataset) - test_size, len(dataset)))
    test_dataset = Subset(dataset, test_indices)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"测试样本数量: {len(test_dataset)}")
    
    # 2. 初始化模型
    model = UnetWithLTAE(in_channels=26, out_channels=1).to(DEVICE)
    
    # 3. 加载训练好的权重
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"找不到权重文件: {CHECKPOINT_PATH}")
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载最佳模型权重! (Epoch {checkpoint.get('epoch', '?')})")
    print(f"验证集 Loss: {checkpoint.get('val_loss', '?'):.4f}")
    
    # 4. 开始测试
    model.eval()
    total_dice = 0.0
    
    print("\n开始推理并生成对比图...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # 前向传播
            outputs = model(images)
            
            # 降低阈值到0.1，让预测区域更完整
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.1).float()
            
            # 计算 Dice
            dice_score = calculate_metric_dice(preds, masks)
            total_dice += dice_score
            
            # ========== 准备可视化数据 ==========
            img_show = images[0, 12, :, :].cpu().numpy()  # 第13帧
            mask_show = masks[0, 0, :, :].cpu().numpy()
            pred_show = preds[0, 0, :, :].cpu().numpy()
            prob_show = probs[0, 0, :, :].cpu().numpy()
            
            # 归一化原图到 0-1
            img_norm = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
            
            # ========== 分离左右肾 ==========
            labeled_mask, num_features = label(pred_show)
            
            # 统计连通域信息
            pred_pixels = int(pred_show.sum())
            gt_pixels = int(mask_show.sum())
            
            # 创建叠加图：原图 + 半透明彩色掩码
            overlay = np.stack([img_norm, img_norm, img_norm], axis=-1)
            
            left_kidney = np.zeros_like(pred_show)
            right_kidney = np.zeros_like(pred_show)
            
            if num_features >= 2:
                # 找出面积最大的两个连通域
                sizes = [(idx, np.sum(labeled_mask == idx)) for idx in range(1, num_features + 1)]
                sizes.sort(key=lambda x: x[1], reverse=True)
                
                comp1_idx, comp1_size = sizes[0]
                comp2_idx, comp2_size = sizes[1]
                
                # 根据重心x坐标判断左右
                center1 = center_of_mass(labeled_mask == comp1_idx)
                center2 = center_of_mass(labeled_mask == comp2_idx)
                
                if center1[1] < center2[1]:  # center[1]是x坐标
                    left_idx, right_idx = comp1_idx, comp2_idx
                else:
                    left_idx, right_idx = comp2_idx, comp1_idx
                
                left_kidney = (labeled_mask == left_idx).astype(float)
                right_kidney = (labeled_mask == right_idx).astype(float)
                
                # 叠加颜色：左肾绿色，右肾红色（0.9让颜色更深更明显）
                overlay[:, :, 1] = np.clip(overlay[:, :, 1] + left_kidney * 0.9, 0, 1)   # 绿色
                overlay[:, :, 0] = np.clip(overlay[:, :, 0] + right_kidney * 0.9, 0, 1)  # 红色
                
                status = f"Left: {int(comp1_size if left_idx==comp1_idx else comp2_size)}px, Right: {int(comp2_size if right_idx==comp2_idx else comp1_size)}px"
            elif num_features == 1:
                # 只有一个连通域，标成黄色
                overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred_show * 0.9, 0, 1)
                overlay[:, :, 1] = np.clip(overlay[:, :, 1] + pred_show * 0.9, 0, 1)
                status = "Only 1 region (yellow)"
            else:
                status = "No prediction"
            
            # ========== 绘制5个子图 ==========
            plt.figure(figsize=(20, 4))
            
            # 1. 原图
            plt.subplot(1, 5, 1)
            plt.title("Input (Frame 13/26)", fontsize=10)
            plt.imshow(img_show, cmap='gray')
            plt.axis('off')
            
            # 2. Ground Truth
            plt.subplot(1, 5, 2)
            plt.title(f"Ground Truth ({gt_pixels}px)", fontsize=10)
            plt.imshow(mask_show, cmap='gray')
            plt.axis('off')
            
            # 3. 概率热力图
            plt.subplot(1, 5, 3)
            plt.title(f"Probability (max={prob_show.max():.3f})", fontsize=10)
            plt.imshow(prob_show, cmap='hot', vmin=0, vmax=1)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            # 4. 二值预测
            plt.subplot(1, 5, 4)
            plt.title(f"Prediction ({pred_pixels}px)", fontsize=10)
            plt.imshow(pred_show, cmap='gray')
            plt.axis('off')
            
            # 5. 叠加图：原图 + 彩色掩码
            plt.subplot(1, 5, 5)
            plt.title(f"Overlay (Dice={dice_score:.3f})\n{status}", fontsize=9)
            plt.imshow(overlay)
            plt.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(SAVE_DIR, f'result_sample_{i+1}.png')
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
    
    avg_dice = total_dice / len(test_loader)
    print(f"\n测试完成！平均 Dice: {avg_dice:.4f}")
    print(f"结果保存在: {SAVE_DIR}")