##new

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CenterLoss(nn.Module):
    """中心损失：修复了类别极度不平衡时的梯度漂移问题"""
    def __init__(self, num_classes=2, feat_dim=64):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        centers = self.centers.to(device)

        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(centers, 2).sum(dim=1, keepdim=True).t().expand(batch_size, self.num_classes)
        )
        distmat.addmm_(x, centers.t(), beta=1.0, alpha=-2.0)

        classes = torch.arange(self.num_classes, device=device, dtype=torch.long)
        labels_expand = labels.view(-1, 1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        
        # --- 【修复区：按类别均摊 Loss】 ---
        # 防止少数类（机械型）的损失被淹没
        class_counts = mask.sum(dim=0).clamp(min=1) # 统计每个类的样本数
        loss = (dist.sum(dim=0) / class_counts).mean() # 先求类内均值，再求跨类均值
        # ---------------------------------
        return loss


class PharmacokineticLatentEncoder(nn.Module):
    """
    固定第 15 分钟(index 14)截断的潜空间编码器
    """
    def __init__(self, input_dim=26, latent_dim=64, num_classes=2):
        super(PharmacokineticLatentEncoder, self).__init__()
        self.injection_frame = 14
        
        # 打药前特征提取
        self.pre_diuretic_net = nn.Sequential(
            nn.Linear(self.injection_frame, 16),
            nn.ReLU(inplace=True)
        )
        
        # 打药后特征提取
        post_frames = input_dim - self.injection_frame
        self.post_diuretic_net = nn.Sequential(
            nn.Linear(post_frames, 16),
            nn.ReLU(inplace=True)
        )
        
        # 融合与潜空间投影
        self.latent_mapping = nn.Sequential(
            nn.Linear(16 + 16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, tac_curve):
        pre_tac = tac_curve[:, :self.injection_frame]
        post_tac = tac_curve[:, self.injection_frame:]
        
        pre_feat = self.pre_diuretic_net(pre_tac)
        post_feat = self.post_diuretic_net(post_tac)
        
        combined_feat = torch.cat([pre_feat, post_feat], dim=1)
        latent_features = self.latent_mapping(combined_feat)
        logits = self.classifier(latent_features)
        
        return logits, latent_features


def train_anchors():
    """训练潜空间锚点（使用左右肾分离的100个样本）"""
    
    # 1. 加载左右肾TAC特征
    tacs = np.load('extracted_tacs_left_right.npy')  # 形状: (100, 26)
    
    # 归一化
    tacs_min = tacs.min(axis=1, keepdims=True)
    tacs_max = tacs.max(axis=1, keepdims=True)
    tacs = (tacs - tacs_min) / (tacs_max - tacs_min + 1e-8)

    # 2. 读取左右肾标签
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clinical_labels_left_right.csv')
    df = pd.read_csv(csv_path)
    
    class_labels = df['label'].values.astype(np.int64)
    
    if len(class_labels) != tacs.shape[0]:
        raise ValueError(
            f'clinical_labels_left_right.csv 中样本数 {len(class_labels)} 与 '
            f'extracted_tacs_left_right.npy 行数 {tacs.shape[0]} 不一致'
        )
    
    print(f"加载数据: {tacs.shape[0]} 个样本")
    print(f"  功能型: {(class_labels == 0).sum()} 例")
    print(f"  机械型: {(class_labels == 1).sum()} 例")
    print(f"  混合型: {(class_labels == 2).sum()} 例")
    print(f"  正常: {(class_labels == 3).sum()} 例")

    # 3. 只保留纯功能型(0)和纯机械型(1)，舍弃混合型(2)和正常(3)
    pure_idx = (class_labels == 0) | (class_labels == 1)
    pure_tacs = torch.tensor(tacs[pure_idx], dtype=torch.float32).to(DEVICE)
    pure_labels = torch.tensor(class_labels[pure_idx], dtype=torch.long).to(DEVICE)
    
    print(f"\n用于训练的纯样本: {pure_tacs.shape[0]} 个")
    print(f"  功能型: {(pure_labels == 0).sum().item()} 例")
    print(f"  机械型: {(pure_labels == 1).sum().item()} 例")

    # 4. 初始化网络与损失函数
    model = PharmacokineticLatentEncoder(input_dim=26, latent_dim=64, num_classes=2).to(DEVICE)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=64).to(DEVICE)
    
    # 双优化器
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_cent = optim.SGD(criterion_cent.parameters(), lr=0.5)

    model.train()
    print("\n开始构建潜空间两极流形...")
    
    for epoch in range(100):
        optimizer_model.zero_grad()
        optimizer_cent.zero_grad()

        logits, latent_feats = model(pure_tacs)
        
        # 交叉熵分开类别，Center Loss 紧凑类内特征
        loss_cls = criterion_cls(logits, pure_labels)
        loss_cent = criterion_cent(latent_feats, pure_labels)
        loss = loss_cls + 0.1 * loss_cent
        
        loss.backward()
        optimizer_model.step()
        
        # 更新 CenterLoss 的中心
        for param in criterion_cent.parameters():
            if param.grad is not None:
                param.grad.data *= (1.0 / 0.1)
        optimizer_cent.step()
        
        if (epoch + 1) % 20 == 0:
            acc = (logits.argmax(dim=1) == pure_labels).float().mean().item()
            print(f"Epoch {epoch+1}/100: Loss={loss.item():.4f}, Acc={acc:.4f}")
            # === 5. 校准并保存完美的中心坐标 ===
    print("\n正在校准并保存最终的潜空间锚点...")
    model.eval() # 切换到推理模式，关闭 Dropout，固定 BN
    with torch.no_grad():
        _, final_feats = model(pure_tacs)
        
        # 直接计算纯类样本在干净特征空间中的几何中心
        final_center_0 = final_feats[pure_labels == 0].mean(dim=0)
        final_center_1 = final_feats[pure_labels == 1].mean(dim=0)
        
        calibrated_centers = torch.stack([final_center_0, final_center_1])

    torch.save(model.state_dict(), 'latent_encoder.pth')
    # 不再保存 criterion_cent.centers，而是保存校准后的中心
    torch.save(calibrated_centers.cpu(), 'manifold_centers.pth') 
    
    print("\n✓ 潜空间锚点训练与校准完毕！")
    print("  - 模型权重: latent_encoder.pth")
    print("  - 中心坐标: manifold_centers.pth")

    # 5. 保存网络权重和潜空间中心坐标
    torch.save(model.state_dict(), 'latent_encoder.pth')
    torch.save(criterion_cent.centers.data.detach().cpu(), 'manifold_centers.pth')
    print("\n✓ 潜空间锚点训练完毕！")
    print("  - 模型权重: latent_encoder.pth")
    print("  - 中心坐标: manifold_centers.pth")


if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    train_anchors()