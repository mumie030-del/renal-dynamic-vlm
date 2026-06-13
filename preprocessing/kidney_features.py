"""
为 VLM 生成改进的肾脏特征图像：
1. 左右肾分开裁剪（tight crop, ~15% padding, 512x512, 单独展示）
2. 全帧原始分辨率叠加图（64x64→256x256 NEAREST + ROI 轮廓）

用法：
    python preprocess_kidney_features.py [dataset_dir]
"""
import os
import sys
import json
import glob
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

OUTPUT_SIZE_CROP = 512        # 裁剪图输出尺寸
OUTPUT_SIZE_OVERLAY = 256     # 叠加图输出尺寸
CONTRAST_FACTOR = 2.0         # 对比度增强
CROP_PADDING = 0.15           # 单肾裁剪 padding 15%
SELECTED_FRAMES = [0, 3, 6, 9, 12, 14, 16, 18, 21, 24, 25]


def load_roi_from_label(label_path):
    """加载 label JSON，返回多边形点列表和图像尺寸"""
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    left_points = None
    right_points = None
    for shape in data['shapes']:
        points = [(p[0], p[1]) for p in shape['points']]
        if shape['label'] == 'l':
            left_points = points
        elif shape['label'] == 'r':
            right_points = points
    width = data.get('imageWidth', 0)
    height = data.get('imageHeight', 0)
    return left_points, right_points, width, height


def get_bounding_box_from_points(points):
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def enhance_contrast(img, factor=CONTRAST_FACTOR):
    return ImageEnhance.Contrast(img).enhance(factor)


def draw_mask_overlay(base_img, points, color, line_width=2):
    """在 PIL Image 上绘制多边形轮廓"""
    if points is None:
        return base_img
    # 如果是灰度图，先转 RGB
    if base_img.mode == 'L':
        base_img = base_img.convert('RGB')
    draw = ImageDraw.Draw(base_img)
    draw.polygon(points, outline=color, width=line_width)
    return base_img


def process_case(case_dir):
    case_dir = Path(case_dir)
    case_name = case_dir.name

    # 输出目录
    output_dir = case_dir / "kidney_features"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查 labels
    labels_dir = case_dir / "labels"
    label_files = list(labels_dir.glob("*.json"))
    if not label_files:
        print(f"  [{case_name}] 跳过：无 label 文件")
        return

    label_path = label_files[0]
    left_points, right_points, orig_width, orig_height = load_roi_from_label(label_path)
    if not left_points or not right_points:
        print(f"  [{case_name}] 跳过：ROI 不完整")
        return

    left_bbox = get_bounding_box_from_points(left_points)
    right_bbox = get_bounding_box_from_points(right_points)

    # 检查 fused 图像
    fused_dir = case_dir / "images_fused_26"
    if not fused_dir.exists():
        print(f"  [{case_name}] 跳过：无 images_fused_26")
        return

    image_files = sorted(fused_dir.glob("fused_*.jpg"))
    if not image_files:
        print(f"  [{case_name}] 跳过：无 fused 图像")
        return

    print(f"\n  [{case_name}] 图像 {orig_width}x{orig_height}, fused {len(image_files)} 帧")
    print(f"    左肾 bbox: {left_bbox}, 右肾 bbox: {right_bbox}")

    sampled_indices = [i for i in SELECTED_FRAMES if i < len(image_files)]
    processed = 0

    for i, img_path in enumerate(image_files):
        if i not in sampled_indices:
            continue

        img_gray = Image.open(img_path).convert('L')

        # ── 方案3: 左右肾分开裁剪 ──
        for points, bbox, side in [
            (left_points, left_bbox, 'l'),
            (right_points, right_bbox, 'r'),
        ]:
            x_min, y_min, x_max, y_max = bbox
            w = x_max - x_min
            h = y_max - y_min
            pad_x = int(w * CROP_PADDING)
            pad_y = int(h * CROP_PADDING)

            cx1 = max(0, x_min - pad_x)
            cy1 = max(0, y_min - pad_y)
            cx2 = min(orig_width, x_max + pad_x)
            cy2 = min(orig_height, y_max + pad_y)

            cropped = img_gray.crop((cx1, cy1, cx2, cy2))
            resized = cropped.resize((OUTPUT_SIZE_CROP, OUTPUT_SIZE_CROP), Image.Resampling.LANCZOS)
            enhanced = enhance_contrast(resized, CONTRAST_FACTOR)
            if enhanced.mode != 'RGB':
                enhanced = enhanced.convert('RGB')
            enhanced.save(output_dir / f"{side}_{i:03d}.jpg", quality=95)

        # ── 方案4: 全帧叠加图（反转 + ROI 轮廓 + NEAREST 放大）──
        arr = np.array(img_gray, dtype=np.uint8)
        inverted = 255 - arr  # 反转使肾脏变亮
        overlay_img = Image.fromarray(inverted).convert('RGB')
        overlay_img = draw_mask_overlay(overlay_img, left_points, color=(255, 0, 0), line_width=2)
        overlay_img = draw_mask_overlay(overlay_img, right_points, color=(0, 100, 255), line_width=2)
        # NEAREST 保持像素块可见
        overlay_resized = overlay_img.resize((OUTPUT_SIZE_OVERLAY, OUTPUT_SIZE_OVERLAY), Image.Resampling.NEAREST)
        overlay_resized.save(output_dir / f"ov_{i:03d}.jpg", quality=95)

        processed += 1

    print(f"    -> 生成 {processed} 帧 x 3 视图 = {processed * 3} 张")


def main():
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/LLM/dataset_new"

    case_dirs = sorted(
        [Path(p) for p in glob.glob(os.path.join(dataset_dir, "*"))
         if os.path.isdir(p) and Path(p).name.isdigit()],
        key=lambda d: int(d.name)
    )

    print(f"找到 {len(case_dirs)} 个病例")
    for case_dir in case_dirs:
        process_case(case_dir)
    print("\n全部完成!")


if __name__ == "__main__":
    main()
