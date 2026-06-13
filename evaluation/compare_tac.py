"""
用 UNet+LTAE 预测的 ROI 掩码替换医生标注，提取 TAC 并对比。
"""
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass, label as connected_components
from torch.utils.data import DataLoader

from datasets import Data3Dataset
from module import UnetWithLTAE

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoints/best_model.pth"
SAVE_DIR = "./tac_comparison"
os.makedirs(SAVE_DIR, exist_ok=True)

# 在原始 64x64 坐标上提取 TAC（因为 raw images 是 64x64）
TAC_SIZE = (64, 64)
MODEL_SIZE = (256, 256)  # 模型输入/输出尺寸


def natural_key(path: str):
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def load_doctor_mask(labels_dir: str, target_size) -> dict:
    """读取医生标注的 JSON 多边形，返回 {side: binary_mask}"""
    label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.json")), key=natural_key)
    if not label_paths:
        raise FileNotFoundError(f"未找到标注文件: {labels_dir}")

    with open(label_paths[0], "r", encoding="utf-8") as f:
        label_data = json.load(f)

    original_w = label_data["imageWidth"]
    original_h = label_data["imageHeight"]
    masks = {}

    for shape in label_data.get("shapes", []):
        label = shape.get("label")
        if label not in {"l", "r"}:
            continue
        mask_img = Image.new("L", (original_w, original_h), 0)
        polygon = [tuple(point) for point in shape["points"]]
        ImageDraw.Draw(mask_img).polygon(polygon, fill=1)
        # 缩放到 target_size
        mask_img = mask_img.resize(target_size, Image.NEAREST)
        masks[label] = np.array(mask_img, dtype=bool)

    if "l" not in masks or "r" not in masks:
        print(f"  [警告] 标注中缺少 l/r 肾区，跳过")
        return None
    return masks


def extract_tac(raw_images_dir: str, mask: np.ndarray, target_size) -> np.ndarray:
    """从原始帧图像中提取 TAC：对每帧计算 mask 内平均像素值"""
    frame_paths = sorted(
        glob.glob(os.path.join(raw_images_dir, "*.jpg"))
        + glob.glob(os.path.join(raw_images_dir, "*.jpeg"))
        + glob.glob(os.path.join(raw_images_dir, "*.png")),
        key=natural_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"未找到原始动态图像: {raw_images_dir}")

    values = []
    for path in frame_paths:
        arr = np.array(Image.open(path).convert("L").resize(target_size), dtype=np.float32)
        values.append(float((255.0 - arr)[mask].mean()))
    return np.array(values, dtype=np.float32)


def normalize_curve(values: np.ndarray) -> np.ndarray:
    max_v = values.max()
    if max_v <= 1e-8:
        return np.zeros_like(values)
    return values / max_v


def smooth_curve(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def split_predicted_mask(pred_binary: np.ndarray) -> dict:
    """
    将模型预测的二值掩码分离为左右肾。
    Returns: {"l": mask, "r": mask} 或 None
    """
    labeled, num_features = connected_components(pred_binary)
    if num_features == 0:
        return None
    if num_features == 1:
        # 只有一个连通域，按重心 x 坐标分左右：左半归 l，右半归 r
        cy, cx = center_of_mass(pred_binary)
        h, w = pred_binary.shape
        l_mask = np.zeros_like(pred_binary, dtype=bool)
        r_mask = np.zeros_like(pred_binary, dtype=bool)
        # 中线左边归 l，右边归 r
        mid = int(cx)
        l_mask[:, :mid] = pred_binary[:, :mid]
        r_mask[:, mid:] = pred_binary[:, mid:]
        return {"l": l_mask, "r": r_mask}

    # 多个连通域：取面积最大的两个
    sizes = [(idx, np.sum(labeled == idx)) for idx in range(1, num_features + 1)]
    sizes.sort(key=lambda x: x[1], reverse=True)
    comp1 = sizes[0][0]
    comp2 = sizes[1][0]

    c1 = center_of_mass(labeled == comp1)
    c2 = center_of_mass(labeled == comp2)

    # center[1] = x 坐标；x 小 → 图像左侧 = 患者右肾 = "l"
    if c1[1] < c2[1]:
        l_idx, r_idx = comp1, comp2
    else:
        l_idx, r_idx = comp2, comp1

    return {
        "l": (labeled == l_idx).astype(bool),
        "r": (labeled == r_idx).astype(bool),
    }


def plot_tac_comparison(case_name, doctor_tac, model_tac, side_label, save_path):
    """绘制 TAC 对比图（医生 vs 模型）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, (label, curves) in zip(axes, [("Raw", (doctor_tac, model_tac)), ("Normalized", (
        normalize_curve(doctor_tac), normalize_curve(model_tac)))]):
        ax.plot(curves[0], linewidth=1.5, label="Doctor mask", alpha=0.8)
        ax.plot(curves[1], linewidth=1.5, label="Model mask", alpha=0.8)
        ax.axvspan(80, 90, alpha=0.1, color="gray")
        ax.axvline(85, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Intensity" if label == "Raw" else "Normalized intensity")
        ax.set_title(f"{side_label} TAC ({label})")
        ax.legend(fontsize=8)

    plt.suptitle(f"{case_name} - TAC Comparison: Doctor vs Model", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存对比图: {save_path}")


@torch.no_grad()
def main():
    # 1. 加载模型
    model = UnetWithLTAE(in_channels=26, out_channels=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"模型加载完成 (Epoch {checkpoint.get('epoch', '?')})")

    # 2. 加载数据集来获取路径信息
    dataset = Data3Dataset(data_root="../data3", target_size=MODEL_SIZE, num_channels=26)

    if len(sys.argv) > 1:
        case_filter = sys.argv[1]
        filtered_indices = [i for i, (_, jp) in enumerate(dataset.samples) if case_filter in str(jp)]
    else:
        filtered_indices = range(len(dataset))
        # 默认只跑前 10 个避免太久
        filtered_indices = list(filtered_indices)[:10]

    print(f"将处理 {len(filtered_indices)} 个病例")

    # 3. 建立 data3 到原始数据集的映射
    # data3/n1 ← dataset_new/1  → 需找 images/ 和 labels/ 在 dataset_new/1/
    # data3/功能_1 ← datasets_12/功能_1 → 在 datasets_12/功能_1/
    data3_to_original = {}
    for images_fnames, json_path in dataset.samples:
        # images_fnames = [..., "../data3/功能_1/0.jpg", ...]
        # 从中提取 case 名
        first_img = images_fnames[0]
        # "../data3/功能_1/0.jpg" → "功能_1"
        case_key = os.path.basename(os.path.dirname(first_img))
        json_path_str = json_path

        # 判断来自哪个原始数据集
        if case_key.startswith("n"):
            orig_id = case_key[1:]  # n1 → 1
            candidate = os.path.join("/root/autodl-tmp/LLM/dataset_new", orig_id)
            if os.path.isdir(candidate):
                data3_to_original[case_key] = candidate
        else:
            candidate = os.path.join("/root/autodl-tmp/LLM/datasets_12", case_key)
            if os.path.isdir(candidate):
                data3_to_original[case_key] = candidate

    # 4. 逐样本处理
    for idx in filtered_indices:
        images_fnames, json_path = dataset.samples[idx]
        case_key = os.path.basename(os.path.dirname(images_fnames[0]))
        orig_dir = data3_to_original.get(case_key)

        if orig_dir is None:
            print(f"[跳过] {case_key}: 找不到原始数据集路径")
            continue

        raw_images_dir = os.path.join(orig_dir, "images")
        labels_dir = os.path.join(orig_dir, "labels")

        if not os.path.isdir(raw_images_dir) or not os.path.isdir(labels_dir):
            print(f"[跳过] {case_key}: 缺 images 或 labels")
            continue

        print(f"\n========== {case_key} ==========")

        # A. 模型预测
        img, _ = dataset[idx]
        img_batch = img.unsqueeze(0).to(DEVICE)  # (1, 26, 256, 256)
        output = model(img_batch)
        probs = torch.sigmoid(output)
        pred_binary = (probs > 0.5).float()

        # B. 分离左右肾（在 256x256 坐标上）
        pred_masks_256 = split_predicted_mask(pred_binary[0, 0].cpu().numpy().astype(bool))
        if pred_masks_256 is None:
            print(f"  模型未预测到肾区，跳过")
            continue

        # C. 缩放到 TAC 尺寸 (64x64)
        pred_masks = {
            side: np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(TAC_SIZE, Image.NEAREST), dtype=bool)
            for side, m in pred_masks_256.items()
        }

        # D. 读取医生标注掩码 (64x64)
        doctor_masks = load_doctor_mask(labels_dir, TAC_SIZE)
        if doctor_masks is None:
            continue

        # E. 提取 TAC
        doctor_tac = {
            side: extract_tac(raw_images_dir, doctor_masks[side], TAC_SIZE)
            for side in ["l", "r"]
        }
        model_tac = {
            side: extract_tac(raw_images_dir, pred_masks[side], TAC_SIZE)
            for side in ["l", "r"]
        }

        # F. 可视化对比
        side_names = {"l": "Right kidney (image-left)", "r": "Left kidney (image-right)"}
        for side in ["l", "r"]:
            save_path = os.path.join(SAVE_DIR, f"{case_key}_tac_{side}.png")
            plot_tac_comparison(
                case_name=case_key,
                doctor_tac=doctor_tac[side],
                model_tac=model_tac[side],
                side_label=side_names[side],
                save_path=save_path,
            )

        # G. 计算相似度指标
        for side, side_cn in [("l", "右肾"), ("r", "左肾")]:
            d = normalize_curve(doctor_tac[side])
            m = normalize_curve(model_tac[side])
            if len(d) != len(m):
                continue
            # 皮尔逊相关系数
            corr = np.corrcoef(d, m)[0, 1]
            # 均方误差
            mse = np.mean((d - m) ** 2)
            # 峰值差异
            peak_diff = abs(d.argmax() - m.argmax())
            print(f"  {side_cn}: Corr={corr:.4f}, MSE={mse:.6f}, PeakDiff={peak_diff}帧")

    print(f"\n🎉 完成！结果保存在 {SAVE_DIR}")


if __name__ == "__main__":
    main()
