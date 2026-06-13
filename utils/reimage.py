"""
图像预处理脚本
1. 隔帧抽帧，保留关键帧
2. 裁剪肾脏区域并放大到512x512
3. 增强对比度
"""

import os
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import glob


DATASET_DIR = Path("/root/autodl-tmp/LLM/dataset_new")
OUTPUT_DIR = Path("/root/autodl-tmp/LLM/dataset_processed")


def load_roi_from_label(label_path: Path) -> Tuple[List[List[float]], List[List[float]]]:
    """从label文件加载ROI坐标"""
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    left_roi = None
    right_roi = None
    
    for shape in data['shapes']:
        points = shape['points']
        if shape['label'] == 'l':
            left_roi = points
        elif shape['label'] == 'r':
            right_roi = points
    
    return left_roi, right_roi


def get_bounding_box(points: List[List[float]]) -> Tuple[int, int, int, int]:
    """从ROI点获取边界框 (x_min, y_min, x_max, y_max)"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def calculate_roi_bboxes(left_roi: List[List[float]], right_roi: List[List[float]], 
                          padding: float = 0.3) -> Tuple[Tuple[int, int, int, int], 
                                                        Tuple[int, int, int, int]]:
    """
    计算双肾的边界框，考虑padding
    padding: 边距占ROI宽/高的比例
    """
    left_bbox = get_bounding_box(left_roi)
    right_bbox = get_bounding_box(right_roi)
    
    def expand_bbox(bbox, img_width, img_height, pad_ratio):
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        # 添加padding
        pad_x = width * pad_ratio
        pad_y = height * pad_ratio
        
        new_x_min = max(0, x_min - pad_x)
        new_y_min = max(0, y_min - pad_y)
        new_x_max = min(img_width, x_max + pad_x)
        new_y_max = min(img_height, y_max + pad_y)
        
        return (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max))
    
    # 假设原始图像尺寸 (需要从第一帧获取)
    return left_bbox, right_bbox


def frame_sampling(frame_indices: List[int], sample_interval: int = 3) -> List[int]:
    """
    隔帧采样
    frame_indices: 帧索引列表
    sample_interval: 采样间隔
    """
    return frame_indices[::sample_interval]


def get_frame_indices_from_filename(filename: str) -> List[int]:
    """从文件名提取帧索引"""
    # 例如: fused_01_1-30.jpg -> [1, 2, ..., 30]
    # 或者: fused_26_128-130.jpg -> [128, 129, 130]
    
    parts = filename.replace('.jpg', '').split('_')
    if len(parts) >= 3:
        range_str = parts[-1]
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            return list(range(start, end + 1))
    
    # 尝试从文件名提取数字
    numbers = ''.join(filter(str.isdigit, filename))
    if numbers:
        return [int(numbers)]
    
    return []


def process_single_image(img: Image.Image, bbox: Tuple[int, int, int, int], 
                          output_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """裁剪图像到指定边界框并resize"""
    x_min, y_min, x_max, y_max = bbox
    cropped = img.crop((x_min, y_min, x_max, y_max))
    resized = cropped.resize(output_size, Image.Resampling.LANCZOS)
    return resized


def enhance_contrast(img: Image.Image, factor: float = 2.0) -> Image.Image:
    """增强图像对比度"""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def enhance_brightness(img: Image.Image, factor: float = 1.1) -> Image.Image:
    """增强图像亮度"""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def process_case(case_dir: Path, sample_interval: int = 3, output_size: int = 512,
                 contrast_factor: float = 2.0, padding: float = 0.4):
    """处理单个病例的所有图像"""
    case_name = case_dir.name
    print(f"\n处理病例: {case_name}")
    
    # 查找label文件
    labels_dir = case_dir / "labels"
    label_files = list(labels_dir.glob("*.json"))
    
    if not label_files:
        print(f"  [跳过] 未找到label文件")
        return
    
    label_path = label_files[0]
    
    # 加载ROI
    try:
        left_roi, right_roi = load_roi_from_label(label_path)
        if not left_roi or not right_roi:
            print(f"  [跳过] ROI数据不完整")
            return
    except Exception as e:
        print(f"  [错误] 加载ROI失败: {e}")
        return
    
    # 创建输出目录
    output_case_dir = OUTPUT_DIR / case_name / "kidney_crop"
    output_case_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理images目录中的原始图像
    images_dir = case_dir / "images_fused_26"
    if not images_dir.exists():
        print(f"  [跳过] images目录不存在")
        return
    
    # 获取所有图像并排序
    image_files = sorted(images_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"  [跳过] 未找到图像文件")
        return
    
    print(f"  找到 {len(image_files)} 张原始图像")
    
    # 获取第一张图像确定尺寸
    first_img = Image.open(image_files[0])
    orig_width, orig_height = first_img.size
    
    # 计算双肾的全局边界框（合并左右肾ROI）
    left_bbox = get_bounding_box(left_roi)
    right_bbox = get_bounding_box(right_roi)
    
    # 合并边界框
    combined_x_min = min(left_bbox[0], right_bbox[0])
    combined_y_min = min(left_bbox[1], right_bbox[1])
    combined_x_max = max(left_bbox[2], right_bbox[2])
    combined_y_max = max(left_bbox[3], right_bbox[3])
    
    # 添加padding
    width = combined_x_max - combined_x_min
    height = combined_y_max - combined_y_min
    pad_x = width * padding
    pad_y = height * padding
    
    crop_x_min = max(0, combined_x_min - pad_x)
    crop_y_min = max(0, combined_y_min - pad_y)
    crop_x_max = min(orig_width, combined_x_max + pad_x)
    crop_y_max = min(orig_height, combined_y_max + pad_y)
    
    crop_bbox = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
    print(f"  裁剪区域: {crop_bbox} (原始尺寸: {orig_width}x{orig_height})")
    
    # 隔帧采样
    total_frames = len(image_files)
    sampled_indices = list(range(0, total_frames, sample_interval))
    print(f"  原始帧数: {total_frames}, 采样间隔: {sample_interval}, 采样后: {len(sampled_indices)} 帧")
    
    # 处理采样的图像
    processed_count = 0
    for i, img_path in enumerate(image_files):
        if i not in sampled_indices:
            continue
        
        try:
            img = Image.open(img_path)
            
            # 裁剪
            cropped = img.crop(crop_bbox)
            
            # resize到512x512
            resized = cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)
            
            # 增强对比度
            enhanced = enhance_contrast(resized, contrast_factor)
            
            # 保存
            output_name = f"kidney_{i:03d}.jpg"
            output_path = output_case_dir / output_name
            enhanced.save(output_path, quality=95)
            processed_count += 1
            
        except Exception as e:
            print(f"  [警告] 处理 {img_path.name} 失败: {e}")
    
    print(f"  [完成] 处理了 {processed_count} 张图像")
    
    # 保存处理参数供参考
    params_file = OUTPUT_DIR / case_name / "process_params.json"
    with open(params_file, 'w') as f:
        json.dump({
            "case_name": case_name,
            "original_size": [orig_width, orig_height],
            "crop_bbox": list(crop_bbox),
            "output_size": output_size,
            "sample_interval": sample_interval,
            "contrast_factor": contrast_factor,
            "total_frames": total_frames,
            "sampled_frames": len(sampled_indices),
            "processed_frames": processed_count
        }, f, indent=2)


def main():
    # 创建输出根目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理所有病例
    case_dirs = sorted(
        path for path in glob.glob(str(DATASET_DIR / "功能_*"))
        + glob.glob(str(DATASET_DIR / "机械_*"))
        + glob.glob(str(DATASET_DIR / "混合_*"))
        if os.path.isdir(path)
        and os.path.basename(path) not in {"results", "images_fused_26"}
    )
    
    # 只处理功能_18作为测试
    test_case = DATASET_DIR / "功能_18"
    
    print("=" * 60)
    print("图像预处理开始")
    print(f"输入目录: {DATASET_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 处理所有病例
    for case_dir in case_dirs:
        case_path = Path(case_dir)
        if case_path.exists():
            process_case(case_path)
    
    print("\n" + "=" * 60)
    print("全部处理完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
