import base64
import glob
import io
import json
import os
import re
import sys
import time
import httpx
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from openai import RateLimitError
from PIL import Image, ImageDraw

MAX_RETRIES = 2        # 429 限流时的最大重试次数
API_DELAY = 3          # 重试退避基数（秒）

DATASET_DIR = "/root/autodl-tmp/LLM/datasets_12"
RESULTS_DIR = os.path.join(DATASET_DIR, "results_ai_ai_claude_baseline")
os.makedirs(RESULTS_DIR, exist_ok=True)

API_KEY = sys.argv[1] if len(sys.argv) > 1 else os.getenv("ANTHROPIC_AUTH_TOKEN")
if not API_KEY:
    raise RuntimeError("请提供 API Key 作为命令行参数，或设置 ANTHROPIC_AUTH_TOKEN 环境变量")

CASE_GLOB = sys.argv[2] if len(sys.argv) > 2 else "*"
BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://timicc.com")
MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=httpx.Timeout(120),
)


def api_call_with_retry(**kwargs):
    """Call client.chat.completions.create with retry on errors."""
    last_exc = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                delay = API_DELAY * (2 ** attempt)
                print(f"[限流] 等待 {delay}s 后重试 (第{attempt+1}次)")
                time.sleep(delay)
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                delay = API_DELAY * (2 ** attempt)
                print(f"[重试] {type(e).__name__}: {e} -> 等待 {delay}s 后重试 (第{attempt+1}次)")
                time.sleep(delay)
    raise last_exc


ALLOWED_STAGE1 = {
    "early_visualization": {"及时", "轻度延迟", "明显延迟", "难以判断"},
    "peak_timing": {"早期达峰", "中期达峰", "晚期达峰", "难以判断"},
    "pre_diuretic_trend": {
        "已开始自然下降",
        "基本平台或轻度滞留",
        "持续上升或进行性积聚",
        "难以判断",
    },
    "post_drop_presence": {
        "明确连续下降",
        "轻微或不稳定下降",
        "无明确下降或继续上升",
        "难以判断",
    },
    "post_diuretic_response": {
        "明显持续改善",
        "有一定改善但不充分",
        "几乎无明显改善",
        "难以判断",
    },
    "late_phase_retention": {"低", "中", "高", "难以判断"},
    "overall_dynamic_pattern": {
        "排空协调",
        "排空延迟但后期改善充分",
        "排空延迟且后期改善不充分",
        "持续滞留无实质改善",
        "难以判断",
    },
    "image_tac_consistency": {"高", "中", "低"},
    "confidence": {"高", "中", "低"},
}

DEFAULT_STAGE1 = {
    "early_visualization": "难以判断",
    "peak_timing": "难以判断",
    "pre_diuretic_trend": "难以判断",
    "post_drop_presence": "难以判断",
    "post_diuretic_response": "难以判断",
    "late_phase_retention": "难以判断",
    "overall_dynamic_pattern": "难以判断",
    "reasoning_focus": "证据不足",
}

FINAL_STAGE2_ALLOWED = {
    "final_label": {"正常肾脏", "功能性梗阻", "机械性梗阻", "混合性梗阻", "难以判断"},
    "confidence": {"高", "中", "低"},
    "image_tac_consistency": {"高", "中", "低"},
    "overall_confidence": {"高", "中", "低"},
}

def natural_key(path: str):
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


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
你是一位有经验的核医学科医师。下面是一个 99mTc-EC 肾动态显像病例，病例名：{case_name}。

你将看到：
1. 按时间顺序排列的关键帧肾脏 ROI 放大图；
2. 一张双肾 TAC 曲线图。

你的任务是分别提取右肾和左肾的动态证据，不要给出最终分型或诊断名称。

【总原则】
1. 以关键帧ROI图作为主要依据，TAC仅用于辅助验证，不替代图像主判。
2. 图像左侧对应患者右肾，图像右侧对应患者左肾。
3. 不要直接输出”正常/功能性/机械性/混合性”。
4. 不要机械依赖单一帧、单一时间点或固定阈值，要根据整个动态演变过程综合判断。
5. 若图像与TAC轻度不一致，以图像整体动态模式为主；若高度一致，可提高一致性与置信度。
6. 你的输出必须严格使用下面给定字段名，不能改字段名，不能新增字段。

【11张ROI图的时相对应关系】（非常重要）
下面 11 张图是按时间顺序从 26 帧融合图中采样得到的，每张图对应的原始帧范围如下：
- 第1张（融合索引0）：原始帧 1-30 —— 早期显影阶段
- 第2张（融合索引3）：原始帧 39-42 —— 早期显影
- 第3张（融合索引6）：原始帧 52-55 —— 中期
- 第4张（融合索引9）：原始帧 64-67 —— 中期
- 第5张（融合索引12）：原始帧 76-79 —— 利尿前即刻
- 第6张（融合索引14）：原始帧 84-87 —— 利尿后早期（利尿剂约在原始帧80-90给药）
- 第7张（融合索引16）：原始帧 92-95 —— 利尿后
- 第8张（融合索引18）：原始帧 100-103 —— 利尿后期
- 第9张（融合索引21）：原始帧 112-115 —— 末期
- 第10张（融合索引24）：原始帧 124-127 —— 末期
- 第11张（融合索引25）：原始帧 128-130 —— 末期
判断利尿后反应时，重点对比第5张（利尿前即刻）与第6-11张（利尿后各阶段）的肾脏放射性变化。

【需要分别提取的动态证据】

1. early_visualization（早期显影）
可选项：
- 及时
- 轻度延迟
- 明显延迟
- 难以判断

2. peak_timing（达峰时相）
可选项：
- 早期达峰
- 中期达峰
- 晚期达峰
- 难以判断

3. pre_diuretic_trend（利尿前趋势）
可选项：
- 已开始自然下降
- 基本平台或轻度滞留
- 持续上升或进行性积聚
- 难以判断

4. post_drop_presence（利尿后是否存在可信下降）
可选项：
- 明确连续下降
- 轻微或不稳定下降
- 无明确下降或继续上升
- 难以判断

判断标准：
- 明确连续下降：利尿后多个连续时相可见放射性浓度或范围下降，方向一致。
- 轻微或不稳定下降：仅少数帧变化，幅度小、方向不稳定，或可能为图像波动。
- 无明确下降或继续上升：利尿后整体无下降，或仍继续积聚。
- 难以判断：图像质量、时相或 TAC 信息不足以确认。

5. post_emptying_sufficiency（利尿后排空是否充分）
可选项：
- 排空较充分
- 排空不充分
- 基本未排空
- 难以判断

判断标准：
- 排空较充分：利尿后残留明显减轻，末期残留低或接近低，整体接近协调排空。
- 排空不充分：利尿后虽有下降，但末期仍有明确中到高残留。
- 基本未排空：利尿后仍持续高位滞留，整体未见实质性排空。
- 难以判断：无法稳定判断排空充分程度。

6. collecting_system_change（肾盂或集合系统放射性范围变化）
可选项：
- 范围明显缩小
- 范围部分缩小
- 范围无明显缩小
- 难以判断

判断标准：
- 范围明显缩小：利尿后集合系统放射性范围较前明显减少。
- 范围部分缩小：利尿后范围有减少，但仍有明确残留。
- 范围无明显缩小：利尿后范围基本不变或继续扩大。
- 难以判断：图像无法可靠判断范围变化。

7. post_diuretic_response（利尿后总体反应）
可选项：
- 明显持续改善
- 有一定改善但不充分
- 几乎无明显改善
- 难以判断

判断标准：
- 明显持续改善：利尿后浓度或范围连续下降，末期残留明显减轻，整体排空趋势较好。
- 有一定改善但不充分：利尿后存在明确方向性下降，但下降后仍有明显残留。
- 几乎无明显改善：利尿后无实质性下降，或仅有轻微、不稳定、单帧波动。
- 难以判断：改善程度无法可靠判断。

8. late_phase_retention（末期残留）
可选项：
- 低
- 中
- 高
- 难以判断

判断标准：
- 低：末期肾盂或集合系统残留不明显。
- 中：末期仍可见残留，但程度不高。
- 高：末期仍有明显高位放射性残留。
- 难以判断：无法稳定判断末期残留程度。

9. late_retention_relative_to_diuretic（末期相对利尿时相变化）
可选项：
- 明显低于利尿时相
- 略低于利尿时相
- 接近或高于利尿时相
- 难以判断

判断标准：
- 明显低于利尿时相：末期残留较利尿时相明显减轻。
- 略低于利尿时相：末期较利尿时相有一定减轻，但减轻幅度有限。
- 接近或高于利尿时相：末期与利尿时相相近，或较前更高。
- 难以判断：无法可靠比较末期与利尿时相。

10. overall_dynamic_pattern（整体动态模式）
可选项：
- 排空协调
- 排空延迟但后期改善充分
- 排空延迟且后期改善不充分
- 持续滞留无实质改善
- 难以判断

判断标准：
- 排空协调：显影、达峰、下降及末期排空过程整体协调。
- 排空延迟但后期改善充分：早中期排空较慢，但后期明显改善，末期残留较低。
- 排空延迟且后期改善不充分：后期有改善，但排空仍不充分，末期仍有明确残留。
- 持续滞留无实质改善：利尿后仍持续高位滞留，未见实质性排空。
- 难以判断：证据不足或图像与 TAC 明显冲突。

11. reasoning_focus（简要理由）
要求：
- 用不超过 60 字说明最关键的动态依据。
- 优先描述：达峰时相、利尿前趋势、利尿后下降是否可信、排空是否充分、末期残留。
- 不要写最终病名或最终分型。
- 不要写空泛总结。

【重要提醒】
1. 不要把末期残留高直接等同于无排空。
2. 不要把轻微、短暂、单帧或方向不稳定的变化直接视为可信下降。
3. 判断利尿后变化时，应同时观察浓度变化和范围变化。
4. 若利尿后有下降，但末期仍有明显残留，应分别体现在 post_drop_presence、post_emptying_sufficiency 和 late_phase_retention 中。
5. 若利尿后几乎无下降，且末期仍明显残留，应体现在 post_drop_presence 和 post_emptying_sufficiency 中。
6. Stage1 只做动态证据提取，不做最终分型。

【特别提醒：最容易混淆的边界】
正常 vs 功能性
若整体排空协调，达峰后可自然下降，末期残留低或不明显，即使达峰略晚，也优先判断为正常。
只有当早中期滞留较明确，自然排空不充分，并且主要依赖利尿后才出现明显排空改善时，才更偏功能性梗阻。

功能性 vs 混合性
功能性梗阻表现为“排得慢，但利尿后能比较充分地排出去”。其特点是利尿后改善明确、连续，末期残留明显减轻，末期残留中偏低，整体排空趋势接近协调。
混合性梗阻表现为“能排一部分，但排不彻底”。其特点是利尿后可见一定下降，但下降幅度不足，排空不充分，末期仍有明确中偏高残留，整体仍处于滞留状态。

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
- 只要利尿后稍微下降一点点，即使末期高位滞留，也要优先考虑混合型梗阻，不要误记为机械性梗阻。
- 当机械型特征存有但利尿之后仍然有下降趋势，即使非常小也是混合型梗阻。
- 混合性梗阻和功能性梗阻很像，趋势是一样的，所以要根据末期放射性浓度聚集程度区分。高滞留是混合性，中低滞留是功能。

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

输出格式必须是严格 JSON，不要输出任何额外文字：
{{
  "right_kidney": {{
    "early_visualization": "及时 或 轻度延迟 或 明显延迟 或 难以判断",
    "peak_timing": "早期达峰 或 中期达峰 或 晚期达峰 或 难以判断",
    "pre_diuretic_trend": "已开始自然下降 或 基本平台或轻度滞留 或 持续上升或进行性积聚 或 难以判断",
    "post_drop_presence": "明确连续下降 或 轻微或不稳定下降 或 无明确下降或继续上升 或 难以判断",
    "post_emptying_sufficiency": "排空较充分 或 排空不充分 或 基本未排空 或 难以判断",
    "collecting_system_change": "范围明显缩小 或 范围部分缩小 或 范围无明显缩小 或 难以判断",
    "post_diuretic_response": "明显持续改善 或 有一定改善但不充分 或 几乎无明显改善 或 难以判断",
    "late_phase_retention": "低 或 中 或 高 或 难以判断",
    "late_retention_relative_to_diuretic": "明显低于利尿时相 或 略低于利尿时相 或 接近或高于利尿时相 或 难以判断",
    "overall_dynamic_pattern": "排空协调 或 排空延迟但后期改善充分 或 排空延迟且后期改善不充分 或 持续滞留无实质改善 或 难以判断",
    "reasoning_focus": "不超过60字"
  }},
  "left_kidney": {{
    "early_visualization": "及时 或 轻度延迟 或 明显延迟 或 难以判断",
    "peak_timing": "早期达峰 或 中期达峰 或 晚期达峰 或 难以判断",
    "pre_diuretic_trend": "已开始自然下降 或 基本平台或轻度滞留 或 持续上升或进行性积聚 或 难以判断",
    "post_drop_presence": "明确连续下降 或 轻微或不稳定下降 或 无明确下降或继续上升 或 难以判断",
    "post_emptying_sufficiency": "排空较充分 或 排空不充分 或 基本未排空 或 难以判断",
    "collecting_system_change": "范围明显缩小 或 范围部分缩小 或 范围无明显缩小 或 难以判断",
    "post_diuretic_response": "明显持续改善 或 有一定改善但不充分 或 几乎无明显改善 或 难以判断",
    "late_phase_retention": "低 或 中 或 高 或 难以判断",
    "late_retention_relative_to_diuretic": "明显低于利尿时相 或 略低于利尿时相 或 接近或高于利尿时相 或 难以判断",
    "overall_dynamic_pattern": "排空协调 或 排空延迟但后期改善充分 或 排空延迟且后期改善不充分 或 持续滞留无实质改善 或 难以判断",
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
- 正常肾脏：最核心的特征是利尿前趋势为"已开始自然下降"，证明肾脏具备自主排空能力。peak_timing无论是"早期达峰"还是"中期达峰"，只要利尿前已开始自然下降，就应优先判断为正常肾脏。"post_drop_presence": "明确连续下降"进一步佐证排空通畅。末期残留受多种因素影响，不能单独用作排除正常肾脏的依据。
- 功能性梗阻：利尿前未达峰，利尿后排空明显，整体更接近“排泄延迟但仍可改善”，提示排空受影响，但不支持固定性重度受阻。
- 机械性梗阻：整体更接近“持续性高位滞留、不排空，无改善”，提示固定性受阻可能更大。
- 混合性梗阻：整体介于功能性与机械性之间，既不是顺畅改善，也不是完全固定不变。
- 难以判断：证据冲突明显，或证据不足以支持稳定分型。
- 混合性梗阻和功能性梗阻很像，趋势是一样的，所以当有利尿后峰值下降的情况下，要根据末期放射性浓度聚集程度区分：高滞留是混合性，中低滞留是功能。
- 正常性梗阻是利尿前就已经达峰了，符合自然生理过程。


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
            "post_drop_presence",
            "post_diuretic_response",
            "late_phase_retention",
            "overall_dynamic_pattern",
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

    response2 = api_call_with_retry(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": stage2_prompt}]}
        ],
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

    masks, target_size = load_roi_masks(labels_dir)

    # 图像左侧 = 患者右肾 -> masks['l']
    # 图像右侧 = 患者左肾 -> masks['r']
    right_values = load_tac_series(raw_images_dir, masks["l"], target_size)
    left_values = load_tac_series(raw_images_dir, masks["r"], target_size)

    tac_hint_text = tac_summary_text(right_values, left_values)
    stage1_prompt = build_stage1_prompt(case_name, tac_hint_text)
    image_contents = build_image_contents(roi_images_dir)
    tac_plot = build_tac_plot_content(right_values, left_values)

    response1 = api_call_with_retry(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": image_contents + [tac_plot] + [{"type": "text", "text": stage1_prompt}],
            }
        ],
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
