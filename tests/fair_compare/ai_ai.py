import base64
import glob
import io
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from openai import OpenAI
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass, label as connected_components

# ── UNet+LTAE 模型（用于 TAC 掩码预测）──
sys.path.insert(0, os.path.dirname(__file__))
from module import UnetWithLTAE

MODEL_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
_model = None
_device = None

DATASET_DIR = "/root/autodl-tmp/LLM/datasets_12"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_ai_ai_unetltae")
os.makedirs(RESULTS_DIR, exist_ok=True)

API_KEY = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("请提供 API Key 作为命令行参数，或设置 DASHSCOPE_API_KEY 环境变量")

CASE_GLOB = sys.argv[2] if len(sys.argv) > 2 else "*"
MODEL_NAME = os.getenv("VLM_MODEL", "qwen3.6-plus")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

ALLOWED_STAGE1 = {
    "early_visualization": {"及时", "轻度延迟", "明显延迟", "难以判断"},
    "peak_timing": {"早期达峰", "中期达峰", "晚期达峰", "难以判断"},
    "pre_diuretic_trend": {
        "已开始自然下降",
        "基本平台或轻度滞留",
        "持续上升或进行性积聚",
        "难以判断",
    },
    "post_diuretic_response": {
        "明显持续改善",
        "有一定改善但不充分",
        "几乎无明显改善",
        "难以判断",
    },
    "late_phase_retention": {"低", "中", "高", "难以判断"},
    "overall_pattern": {
        "整体排空协调",
        "排空延迟但可改善",
        "持续高位滞留",
        "介于可改善与持续滞留之间",
        "难以判断",
    },
    "image_tac_consistency": {"高", "中", "低"},
    "confidence": {"高", "中", "低"},
}

DEFAULT_STAGE1 = {
    "early_visualization": "难以判断",
    "peak_timing": "难以判断",
    "pre_diuretic_trend": "难以判断",
    "post_diuretic_response": "难以判断",
    "late_phase_retention": "难以判断",
    "overall_pattern": "难以判断",
    "reasoning_focus": "证据不足",
}

FINAL_STAGE2_ALLOWED = {
    "final_label": {"正常肾脏", "功能性梗阻", "机械性梗阻", "混合性梗阻", "难以判断"},
    "confidence": {"高", "中", "低"},
    "overall_confidence": {"高", "中", "低"},
    "image_tac_consistency": {"高", "中", "低"},
}


def natural_key(path: str):
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


# ── UNet+LTAE 模型辅助函数 ──

def _load_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = UnetWithLTAE(in_channels=26, out_channels=1).to(_device)
    ckpt = torch.load(MODEL_CHECKPOINT, map_location=_device)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    print(f"  [模型] UNet+LTAE 已加载 (Epoch {ckpt.get('epoch', '?')})")
    return _model, _device


def _load_fused_images(fused_dir: str) -> np.ndarray:
    """加载 26 张融合图像 → 归一化到 [0,1] → 返回 (26, 256, 256)"""
    frame_paths = sorted(
        glob.glob(os.path.join(fused_dir, "*.jpg"))
        + glob.glob(os.path.join(fused_dir, "*.jpeg"))
        + glob.glob(os.path.join(fused_dir, "*.png")),
        key=natural_key,
    )
    if len(frame_paths) != 26:
        raise ValueError(f"融合图像数量不为26: {len(frame_paths)} ({fused_dir})")

    stack = []
    for path in frame_paths:
        arr = np.array(
            Image.open(path).convert("L").resize((256, 256), Image.BILINEAR),
            dtype=np.float32,
        )
        stack.append(arr)

    imgs = np.stack(stack, axis=0)  # (26, 256, 256)
    max_val = imgs.max()
    imgs = imgs / 65535.0 if max_val > 255 else imgs / 255.0
    return imgs


def _split_predicted_mask(pred_binary: np.ndarray) -> dict | None:
    """将模型预测的二值掩码分离为 {"l": ..., "r": ...}"""
    labeled, num_features = connected_components(pred_binary)
    if num_features == 0:
        return None
    if num_features == 1:
        cy, cx = center_of_mass(pred_binary)
        h, w = pred_binary.shape
        mid = int(cx)
        l_mask = np.zeros_like(pred_binary, dtype=bool)
        r_mask = np.zeros_like(pred_binary, dtype=bool)
        l_mask[:, :mid] = pred_binary[:, :mid]
        r_mask[:, mid:] = pred_binary[:, mid:]
        return {"l": l_mask, "r": r_mask}

    sizes = [(idx, np.sum(labeled == idx)) for idx in range(1, num_features + 1)]
    sizes.sort(key=lambda x: x[1], reverse=True)
    comp1, comp2 = sizes[0][0], sizes[1][0]
    c1, c2 = center_of_mass(labeled == comp1), center_of_mass(labeled == comp2)
    l_idx, r_idx = (comp1, comp2) if c1[1] < c2[1] else (comp2, comp1)
    return {
        "l": (labeled == l_idx).astype(bool),
        "r": (labeled == r_idx).astype(bool),
    }


def predict_masks_from_model(case_dir: str, target_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """用 UNet+LTAE 从 images_fused_26 预测 ROI 掩码，返回 {"l": mask, "r": mask} 坐标在 target_size"""
    model, device = _load_model()
    fused_dir = os.path.join(case_dir, "images_fused_26")
    if not os.path.isdir(fused_dir):
        raise FileNotFoundError(f"缺少融合图像目录: {fused_dir}")

    imgs = _load_fused_images(fused_dir)                         # (26, 256, 256)
    inp = torch.from_numpy(imgs).float().unsqueeze(0).to(device) # (1, 26, 256, 256)

    with torch.no_grad():
        out = model(inp)
        probs = torch.sigmoid(out)
        pred = (probs > 0.5).float()[0, 0].cpu().numpy().astype(bool)

    splitted = _split_predicted_mask(pred)
    if splitted is None:
        raise RuntimeError(f"模型未预测到任何肾区: {case_dir}")

    # 缩放到 TAC 提取所需的分辨率
    return {
        side: np.array(
            Image.fromarray((m * 255).astype(np.uint8)).resize(target_size, Image.NEAREST),
            dtype=bool,
        )
        for side, m in splitted.items()
    }


def _get_raw_image_size(raw_images_dir: str) -> Tuple[int, int]:
    """从第一张原始帧获取图像尺寸"""
    candidates = (
        glob.glob(os.path.join(raw_images_dir, "*.jpg"))
        + glob.glob(os.path.join(raw_images_dir, "*.jpeg"))
        + glob.glob(os.path.join(raw_images_dir, "*.png"))
    )
    if not candidates:
        raise FileNotFoundError(f"未找到原始动态图像: {raw_images_dir}")
    return Image.open(sorted(candidates, key=natural_key)[0]).size


def load_roi_masks(labels_dir: str) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
    label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.json")), key=natural_key)
    if not label_paths:
        raise FileNotFoundError(f"未找到ROI标注文件: {labels_dir}")

    with open(label_paths[0], "r", encoding="utf-8") as f:
        label_data = json.load(f)

    width = label_data["imageWidth"]
    height = label_data["imageHeight"]
    masks: Dict[str, np.ndarray] = {}

    for shape in label_data.get("shapes", []):
        label = shape.get("label")
        if label not in {"l", "r"}:
            continue
        mask_img = Image.new("L", (width, height), 0)
        polygon = [tuple(point) for point in shape["points"]]
        drawer = ImageDraw.Draw(mask_img)
        drawer.polygon(polygon, fill=1)
        masks[label] = np.array(mask_img, dtype=bool)

    if "l" not in masks or "r" not in masks:
        raise ValueError("ROI标注中缺少 l / r 肾区")
        

    return masks, (width, height)


def load_tac_series(raw_images_dir: str, mask: np.ndarray, target_size: Tuple[int, int]) -> List[float]:
    frame_paths = sorted(
        glob.glob(os.path.join(raw_images_dir, "*.jpg"))
        + glob.glob(os.path.join(raw_images_dir, "*.jpeg"))
        + glob.glob(os.path.join(raw_images_dir, "*.png")),
        key=natural_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"未找到原始动态图像: {raw_images_dir}")

    values: List[float] = []
    for path in frame_paths:
        image = Image.open(path).convert("L").resize(target_size)
        arr = np.array(image, dtype=np.float32)
        values.append(float((255.0 - arr)[mask].mean()))
    return values


def normalize_curve(values: List[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    max_v = float(arr.max())
    if max_v <= 1e-8:
        return np.zeros_like(arr)
    return arr / max_v


def smooth_curve(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def compute_curve_summary(values: List[float]) -> Dict[str, object]:
    raw = np.array(values, dtype=np.float32)
    norm = smooth_curve(normalize_curve(values), window=5)
    if norm.size == 0:
        return {
            "peak_frame": None,
            "start_level": None,
            "end_level": None,
            "post_diuretic_drop": None,
            "late_mean": None,
        }

    n = len(norm)
    pre_end = min(83, n - 1)
    late_start = min(max(n - 15, 0), n - 1)

    peak_frame = int(np.argmax(norm)) + 1
    start_level = float(norm[min(5, n - 1)])
    end_level = float(norm[-1])
    late_mean = float(norm[late_start:].mean())
    pre_ref = float(norm[pre_end])
    post_ref = float(norm[-1])
    post_diuretic_drop = float(pre_ref - post_ref)

    return {
        "peak_frame": peak_frame,
        "start_level": round(start_level, 4),
        "end_level": round(end_level, 4),
        "post_diuretic_drop": round(post_diuretic_drop, 4),
        "late_mean": round(late_mean, 4),
        "curve": norm.tolist(),
        "raw_curve": raw.tolist(),
    }


def tac_summary_text(right_values: List[float], left_values: List[float]) -> str:
    right_summary = compute_curve_summary(right_values)
    left_summary = compute_curve_summary(left_values)
    return f"""
已提供双肾TAC曲线图。TAC只作辅助验证，不替代图像主判。
可参考以下辅助信息，但不要机械依赖单个数字：
- 右肾TAC辅助摘要：peak_frame={right_summary['peak_frame']}, post_diuretic_drop={right_summary['post_diuretic_drop']}, late_mean={right_summary['late_mean']}
- 左肾TAC辅助摘要：peak_frame={left_summary['peak_frame']}, post_diuretic_drop={left_summary['post_diuretic_drop']}, late_mean={left_summary['late_mean']}
重点只看三件事：大致达峰时相、利尿后是否持续下降、末期残留是否仍明显。
""".strip()


def build_tac_plot_content(right_values: List[float], left_values: List[float]) -> Dict[str, object]:
    right_norm = smooth_curve(normalize_curve(right_values), window=5)
    left_norm = smooth_curve(normalize_curve(left_values), window=5)
    x = np.arange(1, len(right_norm) + 1)

    fig = plt.figure(figsize=(10, 4), dpi=180)
    plt.plot(x, right_norm, linewidth=2, label="Right kidney")
    plt.plot(x, left_norm, linewidth=2, label="Left kidney")
    plt.axvspan(80, 90, alpha=0.15)
    plt.axvline(85, linestyle="--", linewidth=1)
    plt.text(86, 0.95, "Diuretic", fontsize=8)
    plt.xlabel("Frame")
    plt.ylabel("Normalized intensity")
    plt.title("Kidney TAC curves (smoothed)")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
    }


def build_image_contents(images_dir: str) -> List[Dict[str, object]]:
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png")),
        key=natural_key,
    )
    if not image_paths:
        raise FileNotFoundError(f"未找到关键帧ROI图像: {images_dir}")

    image_contents: List[Dict[str, object]] = []
    for path in image_paths:
        with open(path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        image_contents.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_base64}"},
            }
        )
    return image_contents


def build_stage1_prompt(case_name: str, tac_hint_text: str) -> str:
    return f"""
你是一位有经验的核医学科医师。下面是一个99mTc-EC肾动态显像病例，病例名：{case_name}。

你将看到：
1. 按时间顺序排列的关键帧肾脏ROI放大图（已经裁剪、放大、增强对比度）；
2. 一张双肾TAC曲线图。

你的任务不是直接下最终病名，而是先分别提取右肾和左肾的动态证据。
请严格根据“图像主判，TAC辅助验证”的原则进行。

【总原则】
1. 以关键帧ROI图作为主要依据，TAC仅用于辅助验证，不替代图像主判。
2. 图像左侧对应患者右肾，图像右侧对应患者左肾。
3. 不要直接输出“正常/功能性/机械性/混合性”。
4. 不要机械依赖单一帧、单一时间点或固定阈值，要根据整个动态演变过程综合判断。
5. 若图像与TAC轻度不一致，以图像整体动态模式为主；若高度一致，可提高一致性与置信度。
6. 80-90帧附近为利尿剂相关时相。
7. 你的输出必须严格使用下面给定字段名，不能改字段名，不能新增字段。

【你需要提取的动态证据】
请对每个肾分别判断以下 7 项：

1. early_visualization（早期显影）
   - 及时
   - 轻度延迟
   - 明显延迟
   - 难以判断

2. peak_timing（达峰时相）
   - 早期达峰
   - 中期达峰
   - 晚期达峰
   - 难以判断

3. pre_diuretic_trend（利尿前趋势）
   - 已开始自然下降
   - 基本平台或轻度滞留
   - 持续上升或进行性积聚
   - 难以判断

4. post_diuretic_response（利尿后反应）
   - 明显持续改善
   - 有一定改善但不充分
   - 几乎无明显改善
   - 难以判断

5. late_phase_retention（末期残留）
   - 低
   - 中
   - 高
   - 难以判断

6. overall_pattern（整体动态模式）
   - 整体排空协调
   - 排空延迟但可改善
   - 持续高位滞留
   - 介于可改善与持续滞留之间
   - 难以判断

7. reasoning_focus（简要理由）
   - 用不超过60字说明最关键的动态依据
   - 优先描述：达峰大致时相、利尿前趋势、利尿后变化、末期残留
   - 不要写空泛总结

【证据提取时的区分重点】

- 正常更常见的证据组合：
  整体排空协调；可早期或中期达峰；利尿前可已自然下降，后续排空顺畅；末期残留低。若整体排空协调，且利尿前已存在自然下降，且末期残留低，即使利尿后进一步下降，也优先考虑正常，而不是功能性。

- 功能性梗阻更常见的证据组合：
  排空延迟但可改善；利尿前下降不充分或有短暂滞留；利尿后明显持续改善；末期残留通常低到中。
  核心是“排得慢，但能明显排出去”。

- 机械性梗阻更常见的证据组合：
  持续高位滞留；利尿前常持续上升或明显积聚；利尿后无改善；末期残留高。
  核心是“固定性受阻，排不出去”。

- 混合性梗阻更常见的证据组合：
  介于可改善与持续滞留之间；利尿后有一定改善，但改善不足；末期残留中到高。
  核心是“不是完全不改善，但也远未达到顺畅排空”。只有当“存在可见改善，但改善始终不充分，且末期仍有明确残留，整体排空未恢复至协调模式”时，才优先考虑混合型。

【特别提醒：最容易混淆的边界】
正常 vs 功能性
若整体排空协调，达峰后可自然下降，末期残留低或不明显，即使达峰略晚，也优先判断为正常。只有当早中期滞留较明确，自然排空不充分，并且主要依赖利尿后才出现明显排空改善时，才更偏功能性梗阻。

功能性 vs 混合性
功能性梗阻表现为“排得慢，但利尿后能比较充分地排出去”。其特点是利尿后改善明确、连续，末期残留明显减轻，末期残留低，整体排空趋势接近协调。
混合性梗阻表现为“能排一部分，但排不彻底”。其特点是利尿后可见一定下降，但下降幅度不足，排空不充分，末期仍有明确中到高残留，整体仍处于滞留状态。

混合性 vs 机械性
若利尿后仍有可见改善，但改善不充分、末期仍有中到高残留，更偏混合性梗阻。
若利尿后几乎完全看不到改善，放射性积聚持续高位，末期持续高残留，更偏机械性梗阻。
若出现以下情况，应优先判断为混合性梗阻：
1. 利尿后放射性浓度或范围较前下降；
2. 下降具有一定方向性或连续性，而不是单纯随机波动；
3. 末期仍有中到高残留；
4. 整体表现为“能排一部分，但排不彻底”。

若出现以下情况，才优先判断为机械性梗阻：
1. 利尿后几乎没有下降；
2. 即使有变化，也只是极轻微、短暂、无方向性的波动；
3. 肾盂/集合系统放射性浓度和范围未见实质性减轻；
4. 末期持续高位残留；
5. 整体表现为“利尿后仍基本排不出去”。

判断时不要把“末期高残留”直接等同于机械性梗阻。
末期高残留 + 利尿后可识别部分下降 = 更支持混合性梗阻。
末期高残留 + 利尿后无有效下降 = 更支持机械性梗阻。

如果无法判断下降是否有效，不要给出高置信度机械性梗阻；应选择混合性梗阻低置信度或难以判断。

判断优先级
判断时应优先看利尿后改善程度和末期残留程度，而不是单纯看利尿前是否滞留。
有自然协调下降、末期低残留：优先正常；
利尿后明显改善、末期低残留：优先功能性；
利尿后部分改善、末期中高残留：优先混合性；
利尿后几乎无改善、末期比利尿时刻残留都高：优先机械性。

  补充说明：
- 只要利尿后稍微下降一点点，即使末期高位滞留，也要优先考虑混合型梗阻，不要误记为机械性梗阻或功能性梗阻。当机械型特征存有但利尿之后仍然有下降趋势，即使非常小也是混合型梗阻。

【判断顺序】

先判断：
正常肾脏
非正常/存在梗阻倾向
难以判断

如果是“非正常”，再细分：
机械性
功能性

如果判断偏向功能性：
再看下降是否明显，从而判断更趋向混合性还是功能性。

{tac_hint_text}

输出格式必须是严格JSON，不要输出任何额外文字：
{{
  "right_kidney": {{
    "early_visualization": "及时 或 轻度延迟 或 明显延迟 或 难以判断",
    "peak_timing": "早期达峰 或 中期达峰 或 晚期达峰 或 难以判断",
    "pre_diuretic_trend": "已开始自然下降 或 基本平台或轻度滞留 或 持续上升或进行性积聚 或 难以判断",
    "post_diuretic_response": "明显持续改善 或 有一定改善但不充分 或 几乎无明显改善 或 难以判断",
    "late_phase_retention": "低 或 中 或 高 或 难以判断",
    "overall_pattern": "整体排空协调 或 排空延迟但可改善 或 持续高位滞留 或 介于可改善与持续滞留之间 或 难以判断",
    "reasoning_focus": "不超过60字"
  }},
  "left_kidney": {{
    "early_visualization": "及时 或 轻度延迟 或 明显延迟 或 难以判断",
    "peak_timing": "早期达峰 或 中期达峰 或 晚期达峰 或 难以判断",
    "pre_diuretic_trend": "已开始自然下降 或 基本平台或轻度滞留 或 持续上升或进行性积聚 或 难以判断",
    "post_diuretic_response": "明显持续改善 或 有一定改善但不充分 或 几乎无明显改善 或 难以判断",
    "late_phase_retention": "低 或 中 或 高 或 难以判断",
    "overall_pattern": "整体排空协调 或 排空延迟但可改善 或 持续高位滞留 或 介于可改善与持续滞留之间 或 难以判断",
    "reasoning_focus": "不超过60字"
  }},
  "image_tac_consistency": "高 或 中 或 低",
  "confidence": "高 或 中 或 低"
}}
""".strip()


def build_stage2_prompt(case_name: str, stage1_obj: dict) -> str:
    return f"""
你是一位有经验的核医学科医师。下面是病例 {case_name} 的结构化动态证据提取结果。

{json.dumps(stage1_obj, ensure_ascii=False, indent=2)}

你的任务是：仅根据上述结构化动态证据，对右肾和左肾分别做最终梗阻分型。

【总体要求】
1. Stage1 已经完成证据提取；你现在不要重新抽取证据，而是结合这些已提取结果做最终分型。
2. 不允许使用作者预设打分、阈值、固定加权或硬编码规则。
3. 请像临床医师一样，对所有证据做整体综合判断，而不是逐项投票。
4. 需要重点综合以下信息：早期显影、达峰时相、利尿前趋势、利尿后变化、末期残留、整体动态模式，以及图像-TAC一致性。
5. 若证据间存在轻度不一致，请以整体动态演变模式为核心进行解释。
6. 若图像与TAC一致性高，可提高置信度；若存在冲突或证据不足，应降低置信度。
7. 不要为了输出某一类别而强行归类；若证据不足或明显冲突，可输出“难以判断”。
8. 看的是放射性浓度的变化和放射性的范围变化，而不是TAC曲线。

【最终分型的临床含义】
- 正常肾脏：整体更接近生理性显影、达峰和排泄过程，没有充分证据支持明显梗阻。
- 功能性梗阻：整体更接近“排泄延迟但仍可改善”，提示排空受影响，但不支持固定性重度受阻。
- 机械性梗阻：整体更接近“持续性高位滞留、不排空，无改善”，提示固定性受阻可能更大。
- 混合性梗阻：整体介于功能性与机械性之间，既不是顺畅改善，也不是完全固定不变。
- 难以判断：证据冲突明显，或证据不足以支持稳定分型。

【判读原则】
1. 必须做整体临床综合判断，而不是将任一单一字段视为绝对决定因素。
2. 不要输出中间推理过程，不要输出打分过程。
3. reasoning 用一句简短中文概括最关键的综合依据。
4. 每个肾都必须单独给出 final_label、confidence、reasoning。

请严格输出 JSON，不要输出任何额外文字：
{{
  "right_kidney": {{
    "final_label": "正常肾脏 或 功能性梗阻 或 机械性梗阻 或 混合性梗阻 或 难以判断",
    "confidence": "高 或 中 或 低",
    "reasoning": "不超过80字，概括最关键综合依据"
  }},
  "left_kidney": {{
    "final_label": "正常肾脏 或 功能性梗阻 或 机械性梗阻 或 混合性梗阻 或 难以判断",
    "confidence": "高 或 中 或 低",
    "reasoning": "不超过80字，概括最关键综合依据"
  }},
  "image_tac_consistency": "高 或 中 或 低",
  "overall_confidence": "高 或 中 或 低"
}}
""".strip()


def safe_extract_json(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def validate_stage1_json(obj: dict) -> dict:
    def clean_kidney(k: dict) -> dict:
        out = {}
        for field in [
            "early_visualization",
            "peak_timing",
            "pre_diuretic_trend",
            "post_diuretic_response",
            "late_phase_retention",
            "overall_pattern",
        ]:
            value = str(k.get(field, "")).strip()
            out[field] = value if value in ALLOWED_STAGE1[field] else DEFAULT_STAGE1[field]
        reason = str(k.get("reasoning_focus", DEFAULT_STAGE1["reasoning_focus"]))
        out["reasoning_focus"] = reason.strip().replace("\n", " ")[:60] or DEFAULT_STAGE1["reasoning_focus"]
        return out

    right = clean_kidney(obj.get("right_kidney", {}))
    left = clean_kidney(obj.get("left_kidney", {}))
    consistency = str(obj.get("image_tac_consistency", "中")).strip()
    confidence = str(obj.get("confidence", "中")).strip()

    return {
        "right_kidney": right,
        "left_kidney": left,
        "image_tac_consistency": consistency if consistency in ALLOWED_STAGE1["image_tac_consistency"] else "中",
        "confidence": confidence if confidence in ALLOWED_STAGE1["confidence"] else "中",
    }


def validate_stage2_json(obj: dict) -> dict:
    def clean_kidney(k: dict) -> dict:
        final_label = str(k.get("final_label", "难以判断")).strip()
        confidence = str(k.get("confidence", "中")).strip()
        reasoning = str(k.get("reasoning", "证据不足")).strip().replace("\n", " ")
        return {
            "final_label": final_label if final_label in FINAL_STAGE2_ALLOWED["final_label"] else "难以判断",
            "confidence": confidence if confidence in FINAL_STAGE2_ALLOWED["confidence"] else "中",
            "reasoning": reasoning[:80] if reasoning else "证据不足",
        }

    consistency = str(obj.get("image_tac_consistency", "中")).strip()
    overall_confidence = str(obj.get("overall_confidence", "中")).strip()

    return {
        "right_kidney": clean_kidney(obj.get("right_kidney", {})),
        "left_kidney": clean_kidney(obj.get("left_kidney", {})),
        "image_tac_consistency": consistency if consistency in FINAL_STAGE2_ALLOWED["image_tac_consistency"] else "中",
        "overall_confidence": overall_confidence if overall_confidence in FINAL_STAGE2_ALLOWED["overall_confidence"] else "中",
    }


def classify_from_stage1_llm(case_name: str, stage1_obj: dict) -> Tuple[str, dict]:
    stage2_prompt = build_stage2_prompt(case_name, stage1_obj)

    response2 = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": stage2_prompt}]}
        ],
        extra_body={"enable_thinking": False},
    )
    stage2_raw = (response2.choices[0].message.content or "").strip()
    stage2_obj = safe_extract_json(stage2_raw)
    if not stage2_obj:
        raise ValueError(f"Stage2 JSON解析失败:\n{stage2_raw}")
    return stage2_raw, validate_stage2_json(stage2_obj)


def format_final_report(classification: dict) -> str:
    right = classification["right_kidney"]
    left = classification["left_kidney"]
    return (
        f"右肾：{right['final_label']}；{right['reasoning']}\n"
        f"左肾：{left['final_label']}；{left['reasoning']}\n"
        f"图像TAC一致性：{classification['image_tac_consistency']}\n"
        f"置信度：{classification['overall_confidence']}"
    )


def process_case(case_dir: str) -> None:
    case_name = os.path.basename(case_dir)
    raw_images_dir = os.path.join(case_dir, "images")
    roi_images_dir = os.path.join(case_dir, "kidney_crop")
    labels_dir = os.path.join(case_dir, "labels")

    if not os.path.isdir(raw_images_dir):
        print(f"[跳过] {case_name}: 缺少 images")
        return
    if not os.path.isdir(roi_images_dir):
        print(f"[跳过] {case_name}: 缺少 kidney_crop")
        return
    if not os.path.isdir(labels_dir):
        print(f"[跳过] {case_name}: 缺少 labels")
        return

    print(f"\n========== 开始处理 {case_name} ==========")

    # ── 用 UNet+LTAE 预测掩码替代医生标注 ──
    target_size = _get_raw_image_size(raw_images_dir)
    masks = predict_masks_from_model(case_dir, target_size)

    # 图像左侧 = 患者右肾 -> masks['l']
    # 图像右侧 = 患者左肾 -> masks['r']
    right_values = load_tac_series(raw_images_dir, masks["l"], target_size)
    left_values = load_tac_series(raw_images_dir, masks["r"], target_size)

    tac_hint_text = tac_summary_text(right_values, left_values)
    stage1_prompt = build_stage1_prompt(case_name, tac_hint_text)
    image_contents = build_image_contents(roi_images_dir)
    tac_plot = build_tac_plot_content(right_values, left_values)

    response1 = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": image_contents + [tac_plot] + [{"type": "text", "text": stage1_prompt}],
            }
        ],
        extra_body={"enable_thinking": False},
    )
    stage1_raw = (response1.choices[0].message.content or "").strip()
    stage1_obj = safe_extract_json(stage1_raw)
    if not stage1_obj:
        raise ValueError(f"Stage1 JSON解析失败:\n{stage1_raw}")



    stage1_obj = validate_stage1_json(stage1_obj)

    stage2_raw, classification = classify_from_stage1_llm(case_name, stage1_obj)
    final_report = format_final_report(classification)

    result_txt_path = os.path.join(RESULTS_DIR, f"{case_name}.txt")
    result_json_path = os.path.join(RESULTS_DIR, f"{case_name}.json")

    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write("【Stage1 原始输出】\n")
        f.write(stage1_raw)
        f.write("\n\n【Stage1 校验后结构化证据】\n")
        f.write(json.dumps(stage1_obj, ensure_ascii=False, indent=2))
        f.write("\n\n【Stage2 原始输出】\n")
        f.write(stage2_raw)
        f.write("\n\n【Stage2 校验后最终分型】\n")
        f.write(json.dumps(classification, ensure_ascii=False, indent=2))
        f.write("\n\n【最终回答】\n")
        f.write(final_report)

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_name": case_name,
                "stage1_raw": stage1_raw,
                "stage1_validated": stage1_obj,
                "stage2_raw": stage2_raw,
                "stage2_validated": classification,
                "final_report": final_report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========== Stage1 校验后结构化动态证据 ==========")
    print(json.dumps(stage1_obj, ensure_ascii=False, indent=2))
    print("\n========== Stage2 校验后最终分型 ==========")
    print(json.dumps(classification, ensure_ascii=False, indent=2))
    print("\n========== 最终回答 ==========")
    print(final_report)
    print("\n========== 统计信息 ==========")
    print({"stage1_usage": getattr(response1, "usage", None)})


def main() -> None:
    case_dirs = sorted(
        path
        for path in glob.glob(os.path.join(DATASET_DIR, CASE_GLOB))
        if os.path.isdir(path)
        and os.path.basename(path) not in {"results", os.path.basename(RESULTS_DIR)}
    )

    if not case_dirs:
        raise FileNotFoundError(f"未找到病例文件夹: {os.path.join(DATASET_DIR, CASE_GLOB)}")

    for case_dir in case_dirs:
        try:
            process_case(case_dir)
        except Exception as exc:
            print(f"[失败] {os.path.basename(case_dir)}: {exc}")


if __name__ == "__main__":
    main()