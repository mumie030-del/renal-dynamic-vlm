"""
将150-200例中50个病例的130帧flow图像融合为26帧
"""
import os
import sys
import glob
import re
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_fused_images_custom import fuse_images, FRAME_RANGES


def natural_key(path: str) -> list:
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)]


def process_image_folder(image_dir: str) -> None:
    case_dirs = sorted(
        [os.path.join(image_dir, d) for d in os.listdir(image_dir)
         if os.path.isdir(os.path.join(image_dir, d))],
        key=natural_key
    )

    print(f"找到 {len(case_dirs)} 个病例目录")

    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        print(f"\n{'='*60}")
        print(f"处理: {case_name}")

        flow_files = sorted(
            glob.glob(os.path.join(case_dir, "flow*.jpg")),
            key=natural_key
        )

        total_frames = len(flow_files)
        print(f"  找到 {total_frames} 帧flow图像")

        if total_frames == 0:
            print(f"  [跳过] 无flow图像")
            continue

        output_dir = os.path.join(case_dir, "images_fused_26")
        os.makedirs(output_dir, exist_ok=True)

        for i, (start, end) in enumerate(FRAME_RANGES, start=1):
            start_idx = start - 1
            end_idx = end

            if start_idx < 0:
                start_idx = 0
            if end_idx > total_frames:
                end_idx = total_frames

            group_paths = flow_files[start_idx:end_idx]
            if not group_paths:
                print(f"  [警告] 第{i}组 ({start}-{end}) 无图像")
                continue

            output_name = f"fused_{i:02d}_{start}-{end}.jpg"
            output_path = os.path.join(output_dir, output_name)
            fuse_images(group_paths, output_path)

        print(f"  [完成] 融合图像保存至: {output_dir}")


if __name__ == "__main__":
    IMAGE_DIR = "/root/autodl-tmp/LLM/150-200例"
    process_image_folder(IMAGE_DIR)
