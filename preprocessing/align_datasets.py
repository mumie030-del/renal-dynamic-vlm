"""
将 datasets_12 的 kidney_crop 修改为与 dataset_new 一致的格式：
1. 先创建 images_fused_26（26 张融合图）
2. 再重新生成 kidney_crop（11 张，对比度增强 1.5 倍）
"""
import os
import json
import glob
import shutil
import re
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

DATASET_DIR = Path("/root/autodl-tmp/LLM/datasets_12")

# ========== Step 1: fused image config ==========
FRAME_RANGES = [
    (1, 30), (31, 34), (35, 38), (39, 42), (43, 47), (48, 51),
    (52, 55), (56, 59), (60, 63), (64, 67), (68, 71), (72, 75),
    (76, 79), (80, 83), (84, 87), (88, 91), (92, 95), (96, 99),
    (100, 103), (104, 107), (108, 111), (112, 115), (116, 119),
    (120, 123), (124, 127), (128, 130),
]

def natural_key(path):
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)]

def list_raw_images(raw_images_dir):
    paths = sorted(
        glob.glob(os.path.join(raw_images_dir, "*.jpg")) +
        glob.glob(os.path.join(raw_images_dir, "*.jpeg")) +
        glob.glob(os.path.join(raw_images_dir, "*.png")),
        key=natural_key
    )
    return paths

def fuse_images(image_paths, output_path):
    if not image_paths:
        print(f"  [警告] 无图像可融合: {output_path}")
        return
    first_img = Image.open(image_paths[0])
    width, height = first_img.size
    sum_img = np.zeros((height, width), dtype=np.float64)
    for path in image_paths:
        img = Image.open(path).convert('L')
        img_array = np.array(img, dtype=np.float64)
        sum_img += img_array
    avg_img = sum_img / len(image_paths)
    result = Image.fromarray(avg_img.astype(np.uint8))
    result.save(output_path, quality=95)

def create_fused_images(raw_images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = list_raw_images(raw_images_dir)
    total_frames = len(image_paths)
    if total_frames == 0:
        return False
    for i, (start, end) in enumerate(FRAME_RANGES, start=1):
        start_idx = start - 1
        end_idx = end
        if start_idx < 0:
            start_idx = 0
        if end_idx > total_frames:
            end_idx = total_frames
        group_paths = image_paths[start_idx:end_idx]
        if not group_paths:
            continue
        output_name = f"fused_{i:02d}_{start}-{end}.jpg"
        output_path = os.path.join(output_dir, output_name)
        fuse_images(group_paths, output_path)
    return True

# ========== Step 2: kidney_crop config ==========
SELECTED_FRAMES = [0, 3, 6, 9, 12, 14, 16, 18, 21, 24, 25]  # 0-indexed (匹配 dataset_new)

def load_roi_from_label(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    left_roi = right_roi = None
    for shape in data['shapes']:
        points = shape['points']
        if shape['label'] == 'l':
            left_roi = points
        elif shape['label'] == 'r':
            right_roi = points
    return left_roi, right_roi

def get_bounding_box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def enhance_contrast(img, factor=1.5):
    return ImageEnhance.Contrast(img).enhance(factor)

def generate_kidney_crop(case_dir, output_size=512, contrast_factor=2.2, padding=0.4):
    case_name = case_dir.name

    # labels
    labels_dir = case_dir / "labels"
    label_files = list(labels_dir.glob("*.json"))
    if not label_files:
        print(f"  [跳过] 未找到 label 文件")
        return False
    label_path = label_files[0]

    left_roi, right_roi = load_roi_from_label(label_path)
    if not left_roi or not right_roi:
        print(f"  [跳过] ROI 数据不完整")
        return False

    # 图像目录（使用 fused 26 帧）
    images_dir = case_dir / "images_fused_26"
    if not images_dir.exists():
        print(f"  [跳过] images_fused_26 目录不存在")
        return False

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print(f"  [跳过] 未找到图像文件")
        return False

    # 获取原始图像尺寸
    first_img = Image.open(image_files[0])
    orig_width, orig_height = first_img.size

    # 合并左右肾 ROI 边界框 + padding
    left_bbox = get_bounding_box(left_roi)
    right_bbox = get_bounding_box(right_roi)

    combined_x_min = min(left_bbox[0], right_bbox[0])
    combined_y_min = min(left_bbox[1], right_bbox[1])
    combined_x_max = max(left_bbox[2], right_bbox[2])
    combined_y_max = max(left_bbox[3], right_bbox[3])

    width = combined_x_max - combined_x_min
    height = combined_y_max - combined_y_min
    pad_x = width * padding
    pad_y = height * padding

    crop_x_min = max(0, combined_x_min - pad_x)
    crop_y_min = max(0, combined_y_min - pad_y)
    crop_x_max = min(orig_width, combined_x_max + pad_x)
    crop_y_max = min(orig_height, combined_y_max + pad_y)

    crop_bbox = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)

    # 使用指定帧索引
    sampled_indices = [i for i in SELECTED_FRAMES if i < len(image_files)]
    print(f"  裁剪区域: {crop_bbox} | 选择 {len(sampled_indices)} 帧: {sampled_indices}")

    # 创建输出目录
    output_dir = case_dir / "kidney_crop"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for i, img_path in enumerate(image_files):
        if i not in sampled_indices:
            continue
        img = Image.open(img_path)
        # 灰度图转 RGB（保持与 dataset_new 一致）
        if img.mode == 'L':
            img = img.convert('RGB')
        cropped = img.crop(crop_bbox)
        resized = cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)
        enhanced = enhance_contrast(resized, contrast_factor)
        output_name = f"kidney_{i:03d}.jpg"
        enhanced.save(output_dir / output_name, quality=95)
        processed_count += 1

    print(f"  [完成] 处理了 {processed_count} 张图像")
    return True


def process_case(case_dir):
    case_name = case_dir.name
    print(f"\n{'='*60}")
    print(f"处理病例: {case_name}")
    print(f"{'='*60}")

    # Step 1: 生成 images_fused_26
    raw_images_dir = str(case_dir / "images")
    fused_output_dir = str(case_dir / "images_fused_26")
    print(f"[Step 1] 生成融合图像...")
    create_fused_images(raw_images_dir, fused_output_dir)

    # Step 2: 重新生成 kidney_crop
    print(f"[Step 2] 生成 kidney_crop (11 张, 对比度 2.2x)...")
    generate_kidney_crop(case_dir)


def main():
    # 扫描所有病例目录（排除 results_*）
    case_dirs = sorted(
        [Path(p) for p in glob.glob(str(DATASET_DIR / "*"))
         if Path(p).is_dir() and not Path(p).name.startswith("results")],
        key=lambda d: natural_key(str(d))
    )

    print(f"数据集目录: {DATASET_DIR}")
    print(f"共 {len(case_dirs)} 个病例")
    print("=" * 60)

    for case_dir in case_dirs:
        process_case(case_dir)

    print(f"\n{'='*60}")
    print("全部处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
