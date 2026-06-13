##new

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 封装一个"双卷积块"，这是 U-Net 的基本单元
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 2. LTAE 模块：轻量级时间注意力（修正版，避免内存溢出）
class LTAE_Block(nn.Module):
    """
    轻量级时间注意力模块 (Lightweight Temporal Attention Encoder)
    修正版：使用通道注意力机制，避免计算巨大的空间注意力矩阵
    """
    def __init__(self, in_channels=26, reduction=4):
        super(LTAE_Block, self).__init__()
        # 使用全局平均池化 + MLP 来计算通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, 26, H, W)
        B, C, H, W = x.shape
        
        # 全局平均池化：(B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x).view(B, C)
        
        # 通过 MLP 计算通道注意力权重
        y = self.fc(y).view(B, C, 1, 1)
        
        # 将注意力权重应用到输入上（残差连接）
        return x * y.expand_as(x) + x


# 3. 基础 U-Net（用于对比）
class Unet(nn.Module):
    def __init__(self, in_channels=26, out_channels=1):
        super(Unet, self).__init__()

        # 编码器
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(64, 128) 
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256) 

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64) 

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1) 

    def forward(self, x):
        # 编码器
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # 瓶颈层
        bottleneck = self.bottleneck(p4) 
        
        # 解码器
        u1 = self.up1(bottleneck)
        cat1 = torch.cat([c4, u1], dim=1) 
        x = self.conv_up1(cat1)

        u2 = self.up2(x)
        cat2 = torch.cat([c3, u2], dim=1)
        x = self.conv_up2(cat2)

        u3 = self.up3(x)
        cat3 = torch.cat([c2, u3], dim=1)
        x = self.conv_up3(cat3)

        u4 = self.up4(x)
        cat4 = torch.cat([c1, u4], dim=1)
        x = self.conv_up4(cat4)

        return self.out_conv(x)


# 4. 融合版模型：U-Net + LTAE（修正版）
class UnetWithLTAE(nn.Module):
    """
    结合了轻量级时间注意力机制的 U-Net
    修正版：避免内存溢出，使用通道注意力代替空间注意力
    """
    def __init__(self, in_channels=26, out_channels=1):
        super(UnetWithLTAE, self).__init__()

        # 时间编码器（修正版）
        self.temporal_encoder = LTAE_Block(in_channels=in_channels, reduction=4)

        # 编码器
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(64, 128) 
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256) 

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64) 

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1) 

    def forward(self, x):
        # 0. 先过 LTAE 模块重组时间维度特征
        x = self.temporal_encoder(x)

        # 1. 编码器
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # 2. 瓶颈层
        bottleneck = self.bottleneck(p4) 
        
        # 3. 解码器
        u1 = self.up1(bottleneck)
        cat1 = torch.cat([c4, u1], dim=1) 
        x = self.conv_up1(cat1)

        u2 = self.up2(x)
        cat2 = torch.cat([c3, u2], dim=1)
        x = self.conv_up2(cat2)

        u3 = self.up3(x)
        cat3 = torch.cat([c2, u3], dim=1)
        x = self.conv_up3(cat3)

        u4 = self.up4(x)
        cat4 = torch.cat([c1, u4], dim=1)
        x = self.conv_up4(cat4)

        # 4. 输出
        return self.out_conv(x)

class TAC(nn.Module):
    def __init__(self, in_channels=26, out_channels=1 ,laten_dim=64):
        super(TAC, self).__init__()
        self.unetwithltae=UnetWithLTAE(in_channels=in_channels, out_channels=out_channels)
        self.diagnostic_head=LatentEncoder




if __name__ == '__main__':
    print("="*60)
    print("测试 U-Net 和 U-Net+LTAE 模型（修正版）")
    print("="*60)
    
    # 模拟输入数据：Batch=2, Channels=26, H=256, W=256
    input_data = torch.randn([2, 26, 256, 256])
    print(f"\n输入特征维度: {input_data.shape}")
    
    # 1. 测试基础 U-Net
    print("\n--- 基础 U-Net ---")
    model_base = Unet(in_channels=26, out_channels=1)
    output_base = model_base(input_data)
    params_base = sum(p.numel() for p in model_base.parameters())
    print(f"输出维度: {output_base.shape}")
    print(f"参数量: {params_base:,}")
    
    # 2. 测试 U-Net + LTAE（修正版）
    print("\n--- U-Net + LTAE (修正版) ---")
    model_ltae = UnetWithLTAE(in_channels=26, out_channels=1)
    output_ltae = model_ltae(input_data)
    params_ltae = sum(p.numel() for p in model_ltae.parameters())
    print(f"输出维度: {output_ltae.shape}")
    print(f"参数量: {params_ltae:,}")
    print(f"增加参数: +{params_ltae - params_base:,}")
    
    print("\n✓ 测试通过！内存溢出问题已解决。")