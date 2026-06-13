"""
为 dataset_new/ 生成 kidney_crop（按自定义帧索引采样）

格式说明：
- 从 images_fused_26/ 读取融合图像（跳过 all_merged.jpg）
- 按指定索引采样：0, 3, 6, 9, 12, 14, 16, 18, 21, 24, 26 → 11 张
- 根据 label ROI 合并双肾边界框 + 40% padding 裁剪
- Resize 到 512x512 + 对比度增强 2.2
- 输出 kidney_XXX.jpg
"""
import os
import json
import glob
import shutil
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

DATASET_DIR = Path("/root/autodl-tmp/LLM/150-200例")
SELECTED_INDICES = [0, 3, 6, 9, 12, 14, 16, 18, 21, 24, 25]
OUTPUT_SIZE = 512
CONTRAST_FACTOR = 2.2
PADDING = 0.4


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


def enhance_contrast(img, factor=2.0):
    return ImageEnhance.Contrast(img).enhance(factor)


def process_case(case_dir):
    case_name = case_dir.name
    print(f"\n处理病例: {case_name}")

    # labels
    labels_dir = case_dir / "labels"
    label_files = list(labels_dir.glob("*.json"))
    if not label_files:
        print(f"  [跳过] 未找到 label 文件")
        return
    label_path = label_files[0]

    try:
        left_roi, right_roi = load_roi_from_label(label_path)
        if not left_roi or not right_roi:
            print(f"  [跳过] ROI 数据不完整")
            return
    except Exception as e:
        print(f"  [错误] 加载 ROI 失败: {e}")
        return

    # 输出目录（直接写在 case 目录下）
    output_dir = case_dir / "kidney_crop"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 融合图像目录
    images_dir = case_dir / "images_fused_26"
    if not images_dir.exists():
        print(f"  [跳过] images_fused_26 目录不存在")
        return

    # 只取 fused_XX 文件，跳过 all_merged.jpg
    image_files = sorted(images_dir.glob("fused_*.jpg"))
    if not image_files:
        print(f"  [跳过] 未找到 fused 图像文件")
        return

    print(f"  找到 {len(image_files)} 张 fused 图像")

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
    pad_x = width * PADDING
    pad_y = height * PADDING

    crop_x_min = max(0, combined_x_min - pad_x)
    crop_y_min = max(0, combined_y_min - pad_y)
    crop_x_max = min(orig_width, combined_x_max + pad_x)
    crop_y_max = min(orig_height, combined_y_max + pad_y)

    crop_bbox = (int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max))
    print(f"  裁剪区域: {crop_bbox} (原始尺寸: {orig_width}x{orig_height})")

    # 按指定索引采样
    total_frames = len(image_files)
    sampled_indices = [i for i in SELECTED_INDICES if i < total_frames]
    print(f"  原始帧数: {total_frames}, 采样 {len(sampled_indices)} 帧: {sampled_indices}")

    processed_count = 0
    for i, img_path in enumerate(image_files):
        if i not in sampled_indices:
            continue
        try:
            img = Image.open(img_path)
            # 灰度图转 RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            cropped = img.crop(crop_bbox)
            resized = cropped.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)
            enhanced = enhance_contrast(resized, CONTRAST_FACTOR)
            output_name = f"kidney_{i:03d}.jpg"
            enhanced.save(output_dir / output_name, quality=95)
            processed_count += 1
        except Exception as e:
            print(f"  [警告] 处理 {img_path.name} 失败: {e}")

    print(f"  [完成] 处理了 {processed_count} 张图像")

    # 保存参数
    with open(case_dir / "process_params.json", 'w') as f:
        json.dump({
            "case_name": case_name,
            "original_size": [orig_width, orig_height],
            "crop_bbox": list(crop_bbox),
            "output_size": OUTPUT_SIZE,
            "selected_indices": SELECTED_INDICES,
            "contrast_factor": CONTRAST_FACTOR,
            "padding": PADDING,
            "total_frames": total_frames,
            "sampled_frames": len(sampled_indices),
            "processed_frames": processed_count
        }, f, indent=2)


def main():
    case_dirs = sorted(
        Path(p) for p in glob.glob(str(DATASET_DIR / "*"))
        if os.path.isdir(p) and Path(p).name.isdigit()
    )

    print("=" * 60)
    print(f"为 {DATASET_DIR.name} 生成 kidney_crop（按自定义索引采样：{SELECTED_INDICES}）")
    print(f"数据集目录: {DATASET_DIR}")
    print(f"共 {len(case_dirs)} 个病例")
    print("=" * 60)

    for case_dir in case_dirs:
        process_case(Path(case_dir))

    print("\n" + "=" * 60)
    print("全部处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
