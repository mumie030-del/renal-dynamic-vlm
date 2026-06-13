##new

import json
import os
import glob
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
# 引入数据增强库
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data3Dataset(Dataset):
    def __init__(self, data_root, target_size=(256, 256), num_channels=26, transform=None):
        """
        Args:
            data_root: 数据根目录
            target_size: (H, W)
            num_channels: 序列长度
            transform: albumentations 的增强管道 (新增参数)
        """
        self.data_root = data_root
        self.target_size = target_size
        self.num_channels = num_channels
        self.transform = transform  # <--- 1. 保存 transform
        self.samples = []
        
        def _collect_numeric_images(search_dir: str):
            """
            收集形如 101.jpg / 603.png 这类“纯数字文件名”的图片序列。
            这样可以自动忽略像 '图片1.png' 这类预览图，避免把样本数搞成 27。
            """
            exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            candidates = []
            for fn in os.listdir(search_dir):
                fp = join(search_dir, fn)
                if not os.path.isfile(fp):
                    continue
                lower = fn.lower()
                if not lower.endswith(exts):
                    continue
                stem = os.path.splitext(fn)[0]
                if stem.isdigit():
                    candidates.append((int(stem), fp))
            candidates.sort(key=lambda x: x[0])
            return [fp for _, fp in candidates]

        # 扫描逻辑保持不变...
        all_folders = sorted([
            join(data_root, d) for d in os.listdir(data_root) 
            if os.path.isdir(join(data_root, d))
        ])

        for folder_path in all_folders:
            # 兼容两种结构：
            # 1) folder/images/*.jpg
            # 2) folder/*.jpg (扁平目录)
            images_dir = join(folder_path, "images")
            search_dir = images_dir if os.path.isdir(images_dir) else folder_path

            image_files = _collect_numeric_images(search_dir)
            if len(image_files) != self.num_channels:
                continue

            # 兼容 json 标注：
            # - 优先 folder/masks/*.json
            # - 否则尝试 folder/*.json
            masks_dir = join(folder_path, "masks")
            json_files = glob.glob(join(masks_dir, "*.json")) if os.path.isdir(masks_dir) else []
            if not json_files:
                json_files = glob.glob(join(folder_path, "*.json"))
            target_json = json_files[0] if len(json_files) > 0 else None
            
            self.samples.append((image_files, target_json))
            
        print(f"成功加载 {len(self.samples)} 个样本")

    def _json_to_mask(self, json_path, original_h, original_w, target_h, target_w):
        # 1. 在【原始尺寸】上创建画布，这样 json 的坐标才能正确对应
        mask = Image.new('L', (original_w, original_h), 0)
        if json_path is not None and os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                draw = ImageDraw.Draw(mask)
                for shape in data.get('shapes', []):
                    points = shape.get('points', [])
                    if len(points) < 3: continue
                    polygon = tuple(map(tuple, points))
                    draw.polygon(polygon, outline=1, fill=1)
            except Exception:
                pass
        
        # 2. 画完之后，再将 Mask 缩放到【目标尺寸】
        # ⚠️ 注意：Mask 缩放绝对不能用默认的平滑插值，必须用 NEAREST (最近邻)，否则边缘会糊掉
        if (original_w, original_h) != (target_w, target_h):
            mask = mask.resize((target_w, target_h), Image.NEAREST)
            
        return np.array(mask)

    def __getitem__(self, index):
        image_paths, json_path = self.samples[index]
        
        # --- A. 读取图片 ---
        img_stack = []
        # 读取原始尺寸用于 mask
        with Image.open(image_paths[0]) as ref_img:
            original_w, original_h = ref_img.size

        # 确定目标尺寸
        h_target, w_target = (self.target_size if self.target_size else (original_h, original_w))

        for p in image_paths:
            with Image.open(p) as img_obj:
                if self.target_size:
                    img_obj = img_obj.resize((w_target, h_target), Image.BILINEAR)
                img_np = np.array(img_obj)
                if img_np.ndim == 3:
                    # 使用标准加权公式：0.299*R + 0.587*G + 0.114*B
                    img_np = np.dot(img_np[...,:3], [0.299, 0.587, 0.114])
                img_stack.append(img_np)
            
        # --- B. 堆叠 (关键修改) ---
        # Albumentations 要求输入 shape 为 (Height, Width, Channels)
        # 所以这里 axis=2，而不是 axis=0
        images_np = np.stack(img_stack, axis=2) # Shape: (H, W, 26)
        
        # --- C. 读取 Mask ---
        # --- C. 读取 Mask ---
        # 传入 original_h 和 original_w，让它先按原尺寸画，再缩小到 h_target, w_target
        mask_np = self._json_to_mask(json_path, original_h, original_w, h_target, w_target) # Shape: (H, W)

        # --- D. 应用数据增强 (关键修改) ---
        if self.transform is not None:
            # 同时传入 image 和 mask，保证旋转角度一致
            augmented = self.transform(image=images_np, mask=mask_np)
            images_np = augmented['image']
            mask_np = augmented['mask']

        # --- E. 转 Tensor 并归一化 ---
        # 注意：经过 albumentations 后，如果是 ToTensorV2，已经变成了 Tensor 且通道在前 (C, H, W)
        # 但由于我们要手动处理 16-bit 归一化，建议 transform 里不要包含 ToTensorV2，或者是 numpy 操作
        
        # 如果 transform 里没有 ToTensorV2，这里 images_np 还是 numpy (H, W, 26)
        if isinstance(images_np, np.ndarray):
            images_tensor = torch.from_numpy(images_np).float()
            # 调整维度: (H, W, 26) -> (26, H, W)
            images_tensor = images_tensor.permute(2, 0, 1)
        else:
            # 如果用了 ToTensorV2，已经是 Tensor (26, H, W)
            images_tensor = images_np.float()

        mask_tensor = torch.from_numpy(mask_np).float() if isinstance(mask_np, np.ndarray) else mask_np.float()
        
        # 归一化处理：根据数据类型和最大值判断
        # 更准确的判断：如果最大值超过 255，很可能是 16-bit (0-65535)
        max_val = images_tensor.max().item() if isinstance(images_tensor, torch.Tensor) else float(images_tensor.max())
        if max_val > 255:
            images_tensor = images_tensor / 65535.0
        else:
            images_tensor = images_tensor / 255.0
            
        # Mask 增加通道维度 (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
            
        return images_tensor, mask_tensor
    
    def __len__(self):
        return len(self.samples)

# --- 测试代码 ---
if __name__ == '__main__':
    ROOT_DIR = '../data3'
    
    print("="*50)
    print("测试 Data3Dataset (带增强版)")
    print("="*50)
    
    # 1. 定义增强流水线
    # 注意：这里去掉了 ToTensorV2，因为我们在 Dataset 里手动处理 Tensor 转换和维度置换
    # 这样更方便控制 16-bit 的归一化逻辑
    train_transform = A.Compose([
        A.Rotate(limit=15, p=0.5),        # 随机旋转
        A.HorizontalFlip(p=0.5),          # 水平翻转
        A.VerticalFlip(p=0.5),            # 垂直翻转
        A.ElasticTransform(p=0.3, alpha=1, sigma=50, alpha_affine=50), # 弹性形变
    ])

    # 2. 实例化 Dataset (传入 transform)
    dataset = Data3Dataset(
        data_root=ROOT_DIR, 
        target_size=(256, 256), 
        num_channels=26,
        transform=train_transform  # <--- 传入增强
    )
    
    if len(dataset) > 0:
        img, mask = dataset[0]
        print(f"Image Shape: {img.shape}") # 应该是 (26, 256, 256)
        print(f"Mask Shape : {mask.shape}") # 应该是 (1, 256, 256)
        print("测试通过！")
    else:
        print("没有找到数据，无法测试。")