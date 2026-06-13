"""
肾动态显像 VLM 诊断评估工具
=============================

按论文 §2.7 评估指标与统计方法，从预测 CSV 计算：
  - 总体指标：Accuracy, Cohen's Kappa (linear), Quadratic Weighted Kappa, Macro F1
  - 每类指标：Sensitivity, Specificity, PPV, NPV, F1
  - 混淆矩阵
  - 95% Bootstrap 置信区间（以患者为单位重采样）
  - 多方法配对 McNemar 检验（Holm 校正）

输入格式: case_id, kidney_side, true_label, pred_label
支持中文标签（正常肾脏/功能性梗阻/机械性梗阻/混合性梗阻）和数字标签（0-3）。

用法:
    python evaluate.py eval_ai_ai.csv                          # 单个文件
    python evaluate.py *.csv                                   # 多个文件，自动做 McNemar
    python evaluate.py --method-name "医生ROI" eval1.csv ...   # 自定义方法名
    python evaluate.py --gold-csv 患者信息.csv --results-dir results_xxx/  # 原始格式
    python evaluate.py --gold-csv 患者信息.csv --results-dir results_xxx/ \\
        --method-name "方法名"                                  # 自定义方法名
"""

import sys
import re
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from scipy import stats

# ---------- 标签映射 ----------

# 中文 → 数字
CN_TO_IDX = {
    "正常肾脏": 0,
    "功能性梗阻": 1,
    "机械性梗阻": 2,
    "混合性梗阻": 3,
    "正常梗阻": 0,  # 旧标签，等同于正常肾脏
}

# 数字 → 中文
IDX_TO_CN = {0: "正常肾脏", 1: "功能性梗阻", 2: "机械性梗阻", 3: "混合性梗阻"}

# 固定类别顺序
CLASSES = ["正常肾脏", "功能性梗阻", "机械性梗阻", "混合性梗阻"]

# 金标准简称 → 全称映射（用于 0-100 / 100-200 患者信息.csv）
SHORT_TO_FULL = {
    "机械": "机械性梗阻",
    "功能": "功能性梗阻",
    "混合": "混合性梗阻",
    "正常": "正常肾脏",
    "排泄不明显": "功能性梗阻",  # 排空不明显 → 归入功能性梗阻
}


# ---------- 数据加载 ----------

def load_csv(path: str) -> pd.DataFrame:
    """读取评估 CSV，自动检测标签格式并统一为数字编码。"""
    df = pd.read_csv(path)

    # 检查必需列
    for col in ["case_id", "true_label", "pred_label"]:
        if col not in df.columns:
            raise ValueError(f"{path}: 缺少列 '{col}'")

    # 检测标签类型并转换
    for col in ["true_label", "pred_label"]:
        sample = str(df[col].iloc[0])
        if sample in CN_TO_IDX or any(c in sample for c in CN_TO_IDX):
            df[col] = df[col].astype(str).map(CN_TO_IDX)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 过滤无效标签
    valid_mask = df["true_label"].between(0, 3) & df["pred_label"].between(0, 3)
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        print(f"  [{path}] 丢弃 {n_dropped} 行无效标签")

    return df[valid_mask].copy()


def load_from_raw_format(
    gold_csv: str,
    results_dir: str,
    case_id_col: int = 0,
    left_kidney_col: int = 3,
    right_kidney_col: int = 4,
    sep: str = "\t",
    skip_header: bool = True,
    swap_lr: bool = True,
    row_offset: int = 0,
) -> pd.DataFrame:
    """从金标准 CSV + 预测 JSON 目录构建评估 DataFrame。

    金标准 CSV 格式（Tab 分隔）:
        case_id, 性别, 年龄, 左肾, 右肾
    标签使用简称：机械/功能/混合/正常/排泄不明显。

    预测 JSON 格式（一个文件一个 case）:
        stage2_validated.left_kidney.final_label
        stage2_validated.right_kidney.final_label
    标签使用全称：机械性梗阻/功能性梗阻/混合性梗阻/正常肾脏。

    row_offset: 结果 case_id 到金标准行号的偏移量。
        例：结果 case_id=1 对应金标准第 100 行 → row_offset=99

    Returns:
        DataFrame with columns: case_id, kidney_side, true_label, pred_label
    """
    # 读取金标准
    gold_df = pd.read_csv(gold_csv, sep=sep, header=None, dtype=str)

    if skip_header:
        gold_df = gold_df.iloc[1:]  # 跳过标题行

    # 去除首尾空白
    gold_df = gold_df.map(lambda x: x.strip() if isinstance(x, str) else x)

    results_dir = Path(results_dir)
    rows = []

    # 建立金标准索引：case_id → (left_label, right_label)
    gold_by_id = {}
    for _, row in gold_df.iterrows():
        cid = str(row[case_id_col]).strip()
        if not cid:
            continue
        left_raw = str(row[left_kidney_col]).strip() if pd.notna(row[left_kidney_col]) else ""
        right_raw = str(row[right_kidney_col]).strip() if pd.notna(row[right_kidney_col]) else ""
        gold_by_id[cid] = (left_raw, right_raw)

    # 获取所有预测 JSON
    json_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []

    for json_path in json_files:
        json_case_id = json_path.stem

        # 计算金标准 case_id
        try:
            gold_case_id = str(int(json_case_id) + row_offset)
        except ValueError:
            gold_case_id = json_case_id  # 非数字直接匹配

        if gold_case_id not in gold_by_id:
            continue

        case_id = gold_case_id
        left_label_raw, right_label_raw = gold_by_id[gold_case_id]
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    pred_data = json.load(f)
                sv = pred_data.get("stage2_validated", {})
                pred_left = (
                    sv.get("left_kidney", {}).get("final_label", "").strip()
                )
                pred_right = (
                    sv.get("right_kidney", {}).get("final_label", "").strip()
                )
            except (json.JSONDecodeError, KeyError):
                pass

        # 构建行：仅当金标准和预测都存在时才加入
        # ⚠️ 预测 JSON 可能按图像位置命名（left_kidney = 图像左侧 = 患者右肾）
        #    swap_lr=True 时交换以匹配金标准的解剖学命名
        if swap_lr:
            pairs = [
                ("left", left_label_raw, pred_right),   # 左肾(解剖) → right_kidney(图像右)
                ("right", right_label_raw, pred_left),  # 右肾(解剖) → left_kidney(图像左)
            ]
        else:
            pairs = [
                ("left", left_label_raw, pred_left),
                ("right", right_label_raw, pred_right),
            ]
        for side, true_raw, pred_label in pairs:
            if not pred_label:
                continue  # 无预测则跳过
            # 空白金标准 → 正常肾脏
            true_label = SHORT_TO_FULL.get(true_raw, true_raw) if true_raw else "正常肾脏"
            rows.append({
                "case_id": case_id,
                "kidney_side": side,
                "true_label": true_label,
                "pred_label": pred_label,
            })

    if not rows:
        raise ValueError(
            f"{gold_csv} + {results_dir}: 没有匹配到任何有效评估样本"
        )

    df = pd.DataFrame(rows)

    # 将标签转换为数字编码
    for col in ["true_label", "pred_label"]:
        df[col] = df[col].astype(str).map(CN_TO_IDX)

    # 过滤无效标签
    valid_mask = df["true_label"].between(0, 3) & df["pred_label"].between(0, 3)
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        print(f"  [{gold_csv} + {results_dir}] 丢弃 {n_dropped} 行无效标签")

    return df[valid_mask].copy()


def load_from_type2_format(
    gold_csv: str,
    results_dir: str,
    swap_lr: bool = True,
) -> pd.DataFrame:
    """从 1-52 格式金标准 CSV + 预测 JSON 目录构建评估 DataFrame。

    金标准 CSV 格式（逗号分隔）:
        case_name, kidney_side, true_label
    例: 功能_1, left, 功能性梗阻

    预测 JSON 格式: {case_name}.json
        stage2_validated.left_kidney.final_label
        stage2_validated.right_kidney.final_label

    Returns:
        DataFrame with columns: case_id, kidney_side, true_label, pred_label
    """
    gold_df = pd.read_csv(gold_csv)
    gold_df = gold_df.map(lambda x: x.strip() if isinstance(x, str) else x)

    for col in ["case_name", "kidney_side", "true_label"]:
        if col not in gold_df.columns:
            raise ValueError(f"{gold_csv}: 缺少列 '{col}'")

    results_dir = Path(results_dir)
    rows = []

    for _, row in gold_df.iterrows():
        case_name = str(row["case_name"]).strip()
        gold_side = str(row["kidney_side"]).strip().lower()  # left/right
        true_label = str(row["true_label"]).strip()

        json_path = results_dir / f"{case_name}.json"
        if not json_path.exists():
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            sv = pred_data.get("stage2_validated", {})
            pred_left = sv.get("left_kidney", {}).get("final_label", "").strip()
            pred_right = sv.get("right_kidney", {}).get("final_label", "").strip()
        except (json.JSONDecodeError, KeyError):
            continue

        if not pred_left or not pred_right:
            continue

        # swap_lr: left_kidney(图像) → 患者右肾, right_kidney(图像) → 患者左肾
        if swap_lr:
            pred_map = {"left": pred_right, "right": pred_left}
        else:
            pred_map = {"left": pred_left, "right": pred_right}

        pred_label = pred_map.get(gold_side, "")
        if not pred_label:
            continue

        rows.append({
            "case_id": case_name,
            "kidney_side": gold_side,
            "true_label": true_label,
            "pred_label": pred_label,
        })

    if not rows:
        raise ValueError(
            f"{gold_csv} + {results_dir}: 没有匹配到任何有效评估样本"
        )

    df = pd.DataFrame(rows)

    for col in ["true_label", "pred_label"]:
        df[col] = df[col].astype(str).map(CN_TO_IDX)

    valid_mask = df["true_label"].between(0, 3) & df["pred_label"].between(0, 3)
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        print(f"  [{gold_csv} + {results_dir}] 丢弃 {n_dropped} 行无效标签")

    return df[valid_mask].copy()


# ---------- Bootstrap ----------

def bootstrap_metrics(
    df: pd.DataFrame,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """以患者为单位 Bootstrap 重采样，计算所有指标的 95% CI。"""
    rng = np.random.default_rng(seed)
    patients = df["case_id"].unique()
    n_patients = len(patients)

    y_true_all = df["true_label"].values
    y_pred_all = df["pred_label"].values
    patient_indices = {p: df["case_id"] == p for p in patients}

    # 存储每轮 bootstrap 的指标
    metrics_pool = defaultdict(list)

    for _ in range(n_iter):
        # 有放回抽取患者
        sampled_patients = rng.choice(patients, size=n_patients, replace=True)
        mask = np.zeros(len(df), dtype=bool)
        for p in sampled_patients:
            mask |= patient_indices[p]

        yt, yp = y_true_all[mask], y_pred_all[mask]

        # 总体指标
        metrics_pool["accuracy"].append(accuracy_score(yt, yp))
        metrics_pool["kappa_linear"].append(
            cohen_kappa_score(yt, yp, weights="linear")
        )
        metrics_pool["kappa_quadratic"].append(
            cohen_kappa_score(yt, yp, weights="quadratic")
        )
        metrics_pool["macro_f1"].append(f1_score(yt, yp, average="macro"))
        metrics_pool["weighted_f1"].append(f1_score(yt, yp, average="weighted"))

        # 每类指标
        for i, cls_name in enumerate(CLASSES):
            tp = ((yt == i) & (yp == i)).sum()
            fp = ((yt != i) & (yp == i)).sum()
            fn = ((yt == i) & (yp != i)).sum()
            tn = ((yt != i) & (yp != i)).sum()

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0.0

            metrics_pool[f"{cls_name}_sensitivity"].append(sens)
            metrics_pool[f"{cls_name}_specificity"].append(spec)
            metrics_pool[f"{cls_name}_ppv"].append(ppv)
            metrics_pool[f"{cls_name}_npv"].append(npv)
            metrics_pool[f"{cls_name}_f1"].append(f1)

    # 用原始数据计算点估计
    yt, yp = y_true_all, y_pred_all

    result = {}
    result["accuracy"] = _make_entry(float(accuracy_score(yt, yp)), metrics_pool["accuracy"])
    result["kappa_linear"] = _make_entry(
        float(cohen_kappa_score(yt, yp, weights="linear")), metrics_pool["kappa_linear"]
    )
    result["kappa_quadratic"] = _make_entry(
        float(cohen_kappa_score(yt, yp, weights="quadratic")), metrics_pool["kappa_quadratic"]
    )
    result["macro_f1"] = _make_entry(
        float(f1_score(yt, yp, average="macro")), metrics_pool["macro_f1"]
    )
    result["weighted_f1"] = _make_entry(
        float(f1_score(yt, yp, average="weighted")), metrics_pool["weighted_f1"]
    )

    for i, cls_name in enumerate(CLASSES):
        tp = ((yt == i) & (yp == i)).sum()
        fp = ((yt != i) & (yp == i)).sum()
        fn = ((yt == i) & (yp != i)).sum()
        tn = ((yt != i) & (yp != i)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1v = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0.0

        result[f"{cls_name}_sensitivity"] = _make_entry(float(sens), metrics_pool[f"{cls_name}_sensitivity"])
        result[f"{cls_name}_specificity"] = _make_entry(float(spec), metrics_pool[f"{cls_name}_specificity"])
        result[f"{cls_name}_ppv"] = _make_entry(float(ppv), metrics_pool[f"{cls_name}_ppv"])
        result[f"{cls_name}_npv"] = _make_entry(float(npv), metrics_pool[f"{cls_name}_npv"])
        result[f"{cls_name}_f1"] = _make_entry(float(f1v), metrics_pool[f"{cls_name}_f1"])

    return result


def _make_entry(point: float, bootstrap_values: list) -> dict:
    """构建 {point, ci_low, ci_high} 条目。"""
    arr = np.array(bootstrap_values)
    return {
        "point": point,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


# ---------- 打印 ----------

def fmt_pct(point, ci_low, ci_high, decimals=1):
    """格式化: 80.8 (72.2–87.2)"""
    m = 10**decimals
    p = round(point * 100 * m) / m
    lo = round(ci_low * 100 * m) / m
    hi = round(ci_high * 100 * m) / m
    return f"{p:.{decimals}f} ({lo:.{decimals}f}–{hi:.{decimals}f})"


def print_overall_table(name: str, df: pd.DataFrame, metrics: dict):
    """打印总体指标表 (Table A)。"""
    n_kidneys = len(df)
    n_cases = df["case_id"].nunique()
    n_errors = int((df["true_label"] != df["pred_label"]).sum())

    print(f"\n{'─'*70}")
    print(f"  {name}  (n={n_cases} cases, {n_kidneys} kidneys)")
    print(f"{'─'*70}")
    print(
        f"  {'Metric':<22} {'Point':>8}  {'95% CI':>18}"
    )
    print(f"  {'─'*50}")

    rows = [
        ("Accuracy", "accuracy"),
        ("Cohen's Kappa (linear)", "kappa_linear"),
        ("Quadratic Weighted Kappa", "kappa_quadratic"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
    ]
    for label, key in rows:
        m = metrics[key]
        print(
            f"  {label:<22} {m['point']:>8.4f}  "
            + fmt_pct(m["point"], m["ci_low"], m["ci_high"], decimals=2)
        )
    print(f"  {'Misclassifications':<22} {n_errors:>8}")


def print_per_class_table(metrics: dict):
    """打印每类指标表 (Table B)。"""
    print(f"\n  {'─'*90}")
    print(
        f"  {'Class':<14} {'Sens':>14} {'Spec':>14} {'PPV':>14} {'NPV':>14} {'F1':>18}"
    )
    print(f"  {'─'*86}")

    for cls_name in CLASSES:
        s = metrics[f"{cls_name}_sensitivity"]
        sp = metrics[f"{cls_name}_specificity"]
        p = metrics[f"{cls_name}_ppv"]
        n = metrics[f"{cls_name}_npv"]
        f = metrics[f"{cls_name}_f1"]

        print(
            f"  {cls_name:<14}"
            + f" {fmt_pct(s['point'], s['ci_low'], s['ci_high'])}"
            + f" {fmt_pct(sp['point'], sp['ci_low'], sp['ci_high'])}"
            + f" {fmt_pct(p['point'], p['ci_low'], p['ci_high'])}"
            + f" {fmt_pct(n['point'], n['ci_low'], n['ci_high'])}"
            + f" {fmt_pct(f['point'], f['ci_low'], f['ci_high'])}"
        )


def print_confusion_matrix(df: pd.DataFrame, name: str):
    """打印 4×4 混淆矩阵。"""
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=[0, 1, 2, 3])
    print(f"\n  Confusion Matrix — {name}")
    header = " " * 14 + "".join(f"{c:>8}" for c in CLASSES)
    print(f"  {header}")
    for i, cls in enumerate(CLASSES):
        row = "".join(f"{cm[i, j]:>8}" for j in range(4))
        print(f"  {cls:<14}{row}")


# ---------- McNemar 检验 ----------

def mcnemar_paired(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """配对 McNemar 检验（以患者为单位对齐）。"""
    # 按 case_id + kidney_side 对齐
    merged = df_a.merge(
        df_b,
        on=["case_id", "kidney_side"],
        suffixes=("_a", "_b"),
    )
    if len(merged) == 0:
        return 1.0

    correct_a = merged["true_label_a"] == merged["pred_label_a"]
    correct_b = merged["true_label_b"] == merged["pred_label_b"]

    b = (correct_a & ~correct_b).sum()  # A 对 B 错
    c = (~correct_a & correct_b).sum()  # A 错 B 对

    n_discordant = b + c
    if n_discordant == 0:
        return 1.0

    # 精确二项检验
    p = stats.binomtest(min(b, c), n=n_discordant, p=0.5, alternative="two-sided")
    return float(p.pvalue)


def holm_correction(pvalues: dict) -> dict:
    """Holm 校正 p 值。"""
    items = sorted(pvalues.items(), key=lambda x: x[1])
    n = len(items)
    corrected = {}
    for rank, (key, p) in enumerate(items):
        corrected[key] = min(p * (n - rank), 1.0)
    return corrected


def print_mcnemar_table(names: list, dataframes: list):
    """打印 McNemar 检验 p 值矩阵。"""
    print(f"\n{'─'*70}")
    print(f"  McNemar Paired Tests (Holm-corrected p-values)")
    print(f"{'─'*70}")

    # 计算原始 p 值
    raw_p = {}
    for (i, j) in combinations(range(len(names)), 2):
        raw_p[(i, j)] = mcnemar_paired(dataframes[i], dataframes[j])
        raw_p[(j, i)] = raw_p[(i, j)]

    # Holm 校正
    all_pairs = {}
    for (i, j), p in raw_p.items():
        all_pairs[f"{names[i]} vs {names[j]}"] = p
    corrected = holm_correction(all_pairs)

    # 打印矩阵
    max_name = max(len(n) for n in names)
    header = " " * (max_name + 2) + "".join(
        f"{n:>{max_name + 2}}" for n in names
    )
    print(f"  {header}")
    for i, ni in enumerate(names):
        row = f"  {ni:<{max_name}}"
        for j, nj in enumerate(names):
            if i == j:
                row += f" {'—':>{max_name + 1}}"
            else:
                p = corrected.get(f"{ni} vs {nj}", 1.0)
                stars = (
                    "***"
                    if p < 0.001
                    else "**"
                    if p < 0.01
                    else "*"
                    if p < 0.05
                    else ""
                )
                row += f" {p:.4f}{stars:>{4-len(stars)}}"
        print(row)
    print(f"  *** p<0.001  ** p<0.01  * p<0.05")


# ---------- 主入口 ----------

def evaluate_single(path: str, method_name: str = None) -> tuple:
    """评估单个 CSV 文件，返回 (name, df, metrics)。"""
    if method_name is None:
        method_name = Path(path).stem

    df = load_csv(path)
    metrics = bootstrap_metrics(df)

    print_overall_table(method_name, df, metrics)
    print_per_class_table(metrics)
    print_confusion_matrix(df, method_name)

    return method_name, df, metrics


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # 解析参数
    args = sys.argv[1:]
    method_names = []
    paths = []
    raw_tasks = []  # list of (gold_csv, results_dir, is_type2)
    swap_lr = True  # 默认启用左右交换

    i = 0
    while i < len(args):
        if args[i] == "--method-name" and i + 1 < len(args):
            method_names.append(args[i + 1])
            i += 2
        elif args[i] == "--no-swap":
            swap_lr = False
            i += 1
        elif args[i] == "--gold-csv" and i + 1 < len(args):
            gold_csv = args[i + 1]
            i += 2
            # 可能跟随 --results-dir DIR 或直接跟在 gold-csv 后面
            if i < len(args) and args[i] == "--results-dir" and i + 1 < len(args):
                results_dir = args[i + 1]
                i += 2
            elif i < len(args) and not args[i].startswith("--"):
                results_dir = args[i]
                i += 1
            else:
                print(f"ERROR: --gold-csv 需要指定 --results-dir 或预测目录路径")
                sys.exit(1)
            # 自动检测格式：Type 2 有 case_name + kidney_side 列
            _is_type2 = False
            try:
                header = pd.read_csv(gold_csv, nrows=0).columns.tolist()
                if "case_name" in header and "kidney_side" in header:
                    _is_type2 = True
            except Exception:
                pass
            raw_tasks.append((gold_csv, results_dir, _is_type2))
        elif args[i] in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            # 支持通配符（shell 已展开）
            p = Path(args[i])
            if p.exists():
                paths.append(str(p))
                method_names.append(p.stem)
            i += 1

    if not paths and not raw_tasks:
        print("ERROR: 没有找到 CSV 文件或 --gold-csv 参数")
        sys.exit(1)

    print("=" * 70)
    print("  肾动态显像 VLM 诊断评估")
    print("=" * 70)

    all_names = []
    all_dfs = []

    # 处理标准 CSV 格式
    for path, name in zip(paths, method_names):
        try:
            method_name, df, metrics = evaluate_single(path, name)
            all_names.append(method_name)
            all_dfs.append(df)
        except Exception as e:
            print(f"  [ERROR] {path}: {e}")
            import traceback
            traceback.print_exc()

    # 处理原始格式（金标准 CSV + 预测 JSON 目录）
    for gold_csv, results_dir, is_type2 in raw_tasks:
        try:
            name = Path(results_dir).name
            print(f"\n  [加载] 金标准: {gold_csv}")
            print(f"  [加载] 预测目录: {results_dir}")
            if is_type2:
                df = load_from_type2_format(gold_csv, results_dir, swap_lr=swap_lr)
            else:
                df = load_from_raw_format(gold_csv, results_dir, swap_lr=swap_lr)
            metrics = bootstrap_metrics(df)

            print_overall_table(name, df, metrics)
            print_per_class_table(metrics)
            print_confusion_matrix(df, name)

            all_names.append(name)
            all_dfs.append(df)
        except Exception as e:
            print(f"  [ERROR] {gold_csv} + {results_dir}: {e}")
            import traceback
            traceback.print_exc()

    # 多方法配对 McNemar（仅当比较不同方法时）
    if len(all_dfs) >= 2:
        print_mcnemar_table(all_names, all_dfs)

    # 多数据集聚合（raw_tasks ≥ 2 时自动合并计算总指标）
    if len(raw_tasks) >= 2:
        print(f"\n{'='*70}")
        print(f"  聚合评估 — 合并 {len(raw_tasks)} 个数据集")
        print(f"{'='*70}")

        agg_dfs = []
        for gold_csv, results_dir, is_type2 in raw_tasks:
            try:
                if is_type2:
                    df = load_from_type2_format(gold_csv, results_dir, swap_lr=swap_lr)
                else:
                    df = load_from_raw_format(gold_csv, results_dir, swap_lr=swap_lr)
                # 为每个数据集加前缀避免 case_id 冲突
                prefix = Path(results_dir).parent.name + "/"
                df["case_id"] = prefix + df["case_id"].astype(str)
                agg_dfs.append(df)
            except Exception as e:
                print(f"  [SKIP] {results_dir}: {e}")

        if len(agg_dfs) >= 2:
            df_agg = pd.concat(agg_dfs, ignore_index=True)
            metrics_agg = bootstrap_metrics(df_agg)
            print_overall_table("AGGREGATE (合并)", df_agg, metrics_agg)
            print_per_class_table(metrics_agg)
            print_confusion_matrix(df_agg, "AGGREGATE")

    print(f"\n{'=' * 70}")
    print("  评估完成")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
