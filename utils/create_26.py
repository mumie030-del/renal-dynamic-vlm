"""
将130张原始动态图像按照指定帧范围融合为26张融合图
帧范围：1-30, 31-34, 35-38, 39-42, 43-47, 48-51, 52-55, 56-59, 60-63, 64-67, 68-71, 72-75, 76-79, 80-83, 84-87, 88-91, 92-95, 96-99, 100-103, 104-107, 108-111, 112-115, 116-119, 120-123, 124-127, 128-130
"""
import os
import numpy as np
from PIL import Image
from glob import glob
import re


def natural_key(path: str) -> list:
    """自然排序key函数"""
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)]


def list_raw_images(raw_images_dir: str) -> list:
    """获取所有原始图像路径，按自然顺序排序"""
    paths = sorted(
        glob(os.path.join(raw_images_dir, "*.jpg"))
        + glob(os.path.join(raw_images_dir, "*.jpeg"))
        + glob(os.path.join(raw_images_dir, "*.png")),
        key=natural_key
    )
    return paths


def fuse_images(image_paths: list, output_path: str) -> None:
    """
    将多张图像融合为一张
    融合方式：对齐后取平均值
    """
    if not image_paths:
        print(f"[警告] 无图像可融合: {output_path}")
        return

    # 读取第一张图像获取尺寸
    first_img = Image.open(image_paths[0])
    width, height = first_img.size

    # 初始化累加器
    sum_img = np.zeros((height, width), dtype=np.float64)

    # 累加所有图像
    for path in image_paths:
        img = Image.open(path).convert('L')  # 转为灰度
        img_array = np.array(img, dtype=np.float64)
        sum_img += img_array

    # 计算平均值
    avg_img = sum_img / len(image_paths)

    # 转换回uint8并保存
    result = Image.fromarray(avg_img.astype(np.uint8))
    result.save(output_path, quality=95)
    print(f"融合完成: {output_path} ({len(image_paths)}帧)")


def create_fused_images_custom_ranges(
    raw_images_dir: str,
    output_dir: str,
    frame_ranges: list
) -> None:
    """
    按照自定义帧范围融合图像

    Args:
        raw_images_dir: 原始图像目录
        output_dir: 输出目录
        frame_ranges: 帧范围列表，每个元素为(start, end)，1-indexed
                      例如：[(1, 30), (31, 34), (35, 38), ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有原始图像
    image_paths = list_raw_images(raw_images_dir)
    total_frames = len(image_paths)
    print(f"找到 {total_frames} 张原始图像")

    # 逐组融合
    for i, (start, end) in enumerate(frame_ranges, start=1):
        # 转换为0-indexed
        start_idx = start - 1
        end_idx = end  # Python切片end是不包含的

        # 边界检查
        if start_idx < 0:
            start_idx = 0
        if end_idx > total_frames:
            end_idx = total_frames

        # 获取该范围的图像路径
        group_paths = image_paths[start_idx:end_idx]

        if not group_paths:
            print(f"[警告] 第{i}组 ({start}-{end}) 无图像")
            continue

        # 输出文件名
        output_name = f"fused_{i:02d}_{start}-{end}.jpg"
        output_path = os.path.join(output_dir, output_name)

        # 融合
        fuse_images(group_paths, output_path)

    print(f"\n融合完成！共生成 {len(frame_ranges)} 张融合图像，保存至: {output_dir}")


# 定义帧范围 (1-indexed)
FRAME_RANGES = [
    (1, 30),
    (31, 34),
    (35, 38),
    (39, 42),
    (43, 47),
    (48, 51),
    (52, 55),
    (56, 59),
    (60, 63),
    (64, 67),
    (68, 71),
    (72, 75),
    (76, 79),
    (80, 83),
    (84, 87),
    (88, 91),
    (92, 95),
    (96, 99),
    (100, 103),
    (104, 107),
    (108, 111),
    (112, 115),
    (116, 119),
    (120, 123),
    (124, 127),
    (128, 130),
]


def process_all_cases(dataset_dir: str, output_subdir: str = "images_fused_26") -> None:
    """
    处理dataset_dir下所有病例，生成自定义融合图像

    Args:
        dataset_dir: 数据集根目录
        output_subdir: 输出子目录名
    """
    # 获取所有病例目录
    case_dirs = []
    for path in sorted(glob(os.path.join(dataset_dir, "*")), key=natural_key):
        if not os.path.isdir(path):
            continue
        if os.path.basename(path).startswith("results"):
            continue

        case_name = os.path.basename(path)
        raw_images_dir = os.path.join(path, "images")
        if os.path.isdir(raw_images_dir) and len(list_raw_images(raw_images_dir)) > 0:
            case_dirs.append(path)

    print(f"找到 {len(case_dirs)} 个有效病例")

    # 处理每个病例
    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        raw_images_dir = os.path.join(case_dir, "images")
        output_dir = os.path.join(case_dir, output_subdir)

        print(f"\n{'='*60}")
        print(f"处理病例: {case_name}")
        print(f"{'='*60}")

        create_fused_images_custom_ranges(
            raw_images_dir=raw_images_dir,
            output_dir=output_dir,
            frame_ranges=FRAME_RANGES
        )


if __name__ == "__main__":
    # 数据集根目录
    DATASET_DIR = "/root/autodl-tmp/LLM/50例图像-第三批-5.30"

    # 处理所有病例
    process_all_cases(DATASET_DIR, output_subdir="images_fused_26")

    # 也可以单独处理某个病例，取消下面这行注释并修改路径
    # create_fused_images_custom_ranges(
    #     raw_images_dir="/root/autodl-tmp/LLM/dataset/功能_2/images",
    #     output_dir="/root/autodl-tmp/LLM/dataset/功能_2/images_fused_26",
    #     frame_ranges=FRAME_RANGES
    # )
