"""
对比 results_ai_ai_gpt_baseline_12 (医生ROI) vs results_ai_ai_unetltae (UNet+LTAE预测掩码)
"""
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

GROUND_TRUTH_CSV = "/root/autodl-tmp/LLM/datasets_12/标注模板.csv"
BASELINE_DIR = "/root/autodl-tmp/LLM/datasets_12/results_ai_ai_gpt_baseline_12"
UNETLTAE_DIR = "/root/autodl-tmp/LLM/datasets_12/results_ai_ai_unetltae"

# Label mapping: CSV uses "正常梗阻", model outputs "正常肾脏"
LABEL_MAP = {"正常肾脏": "正常梗阻", "正常梗阻": "正常梗阻"}

SIDE_MAP = {"left_kidney": "left", "right_kidney": "right"}


def load_ground_truth(csv_path: str) -> dict:
    """Returns {(case_name, side): true_label}"""
    truth = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row["case_name"].strip()
            side = row["kidney_side"].strip()
            label = row["true_label"].strip()
            truth[(case, side)] = label
    return truth


def load_predictions(results_dir: str, truth: dict) -> dict:
    """Returns {(case_name, side): pred_label} for cases where both truth and prediction exist"""
    preds = {}
    for case_name, side in truth:
        json_path = os.path.join(results_dir, f"{case_name}.json")
        if not os.path.isfile(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        side_key = "left_kidney" if side == "left" else "right_kidney"
        kidney_data = data.get("stage2_validated", {}).get(side_key, {})
        pred_label = kidney_data.get("final_label", "")

        if not pred_label:
            continue

        # Map "正常肾脏" → "正常梗阻"
        if pred_label in LABEL_MAP:
            pred_label = LABEL_MAP[pred_label]

        preds[(case_name, side)] = pred_label
    return preds


def quadratic_kappa(y_true, y_pred, labels):
    """Compute quadratic weighted Cohen's Kappa"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = cm.sum()
    if n == 0:
        return 0.0

    # Expected agreement by chance
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    outer = np.outer(row_sums, col_sums)
    expected = outer / n

    # Quadratic weights
    k = len(labels)
    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i, j] = 1.0 - ((i - j) ** 2) / ((k - 1) ** 2)

    # Weighted observed agreement
    o = cm.astype(float) / n
    e = expected.astype(float) / n

    w_observed = (o * weights).sum()
    w_expected = (e * weights).sum()

    if w_expected == 1.0:
        return 0.0
    return (w_observed - w_expected) / (1.0 - w_expected)


def strided_confusion_matrix_annotations(cm, labels, title="Confusion Matrix"):
    """Print annotated confusion matrix"""
    max_w = max(len(l) for l in labels)
    fmt_num = max(4, len(str(cm.max())))
    header = f"{'':>{max_w}}  " + "  ".join(f"{l:>{fmt_num}}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:>{max_w}}  " + "  ".join(f"{cm[i, j]:>{fmt_num}}" for j in range(len(labels)))
        print(row)


def evaluate(truth_dict: dict, pred_dict: dict, label_order: list, name: str):
    aligned_true, aligned_pred = [], []
    for key in sorted(truth_dict.keys()):
        if key in pred_dict:
            aligned_true.append(truth_dict[key])
            aligned_pred.append(pred_dict[key])

    y_true = np.array(aligned_true)
    y_pred = np.array(aligned_pred)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred, labels=label_order)
    qk = quadratic_kappa(y_true, y_pred, label_order)

    n_total = len(aligned_true)
    n_correct = (y_true == y_pred).sum()

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  总样本: {n_total}  ({n_correct} 正确, {n_total - n_correct} 错误)")
    print(f"  准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    print(f"  Quadratic Weighted Kappa: {qk:.4f}")
    print()

    print("  混淆矩阵 (行=真实, 列=预测):")
    strided_confusion_matrix_annotations(cm, label_order)
    print()

    print(f"  {'类别':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'支持数':<8}")
    print(f"  {'-'*50}")
    for i, label in enumerate(label_order):
        print(f"  {label:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {int(cm[i].sum()):<8}")

    return {"accuracy": acc, "kappa": kappa, "quadratic_kappa": qk, "cm": cm, "f1": f1}


def main():
    print("加载金标准...")
    truth = load_ground_truth(GROUND_TRUTH_CSV)

    print(f"  金标准: {len(truth)} 条记录")
    print()

    label_order = ["正常梗阻", "功能性梗阻", "机械性梗阻", "混合性梗阻"]

    # 加载 baseline (医生 ROI)
    baseline_preds = load_predictions(BASELINE_DIR, truth)
    print(f"Baseline 预测: {len(baseline_preds)} 条 (来自 {BASELINE_DIR})")

    # 加载 UNet+LTAE 预测
    unet_preds = load_predictions(UNETLTAE_DIR, truth)
    print(f"UNet+LTAE 预测: {len(unet_preds)} 条 (来自 {UNETLTAE_DIR})")

    # 评估 baseline
    baseline_metrics = evaluate(truth, baseline_preds, label_order, "BASELINE: 医生ROI掩码")

    # 评估 UNet+LTAE
    unet_metrics = evaluate(truth, unet_preds, label_order, "UNET+LTAE: 模型预测掩码")

    # ── 逐病例对比，找差异 ──
    print(f"\n{'='*70}")
    print("  逐病例差异对比 (两套方案预测结果不同的病例)")
    print(f"{'='*70}")
    diff_count = 0
    baseline_correct_more = 0
    unet_correct_more = 0
    both_wrong = 0

    for key in sorted(truth.keys()):
        if key not in baseline_preds or key not in unet_preds:
            continue
        true_label = truth[key]
        base_label = baseline_preds[key]
        unet_label = unet_preds[key]
        if base_label != unet_label:
            diff_count += 1
            base_correct = base_label == true_label
            unet_correct = unet_label == true_label
            case, side = key
            side_cn = "左肾" if side == "left" else "右肾"
            marker = ""
            if base_correct and not unet_correct:
                marker = "← Baseline正确, UNet+LTAE错误"
                baseline_correct_more += 1
            elif unet_correct and not base_correct:
                marker = "← UNet+LTAE正确, Baseline错误"
                unet_correct_more += 1
            else:
                marker = "← 两套都错"
                both_wrong += 1
            print(f"  {case} {side_cn}: GT={true_label:<8} "
                  f"Baseline={base_label:<8} UNetLTAE={unet_label:<8} {marker}")

    print(f"\n  预测不同的病例总数: {diff_count}")
    print(f"  Baseline 正确、UNet+LTAE 错误: {baseline_correct_more}")
    print(f"  UNet+LTAE 正确、Baseline 错误: {unet_correct_more}")
    print(f"  两套都错: {both_wrong}")

    # ── 汇总对比 ──
    print(f"\n{'='*70}")
    print("  汇总对比")
    print(f"{'='*70}")
    print(f"  {'指标':<25} {'Baseline(医生ROI)':<20} {'UNet+LTAE':<20}")
    print(f"  {'-'*65}")
    print(f"  {'准确率':<25} {baseline_metrics['accuracy']:<20.4f} {unet_metrics['accuracy']:<20.4f}")
    print(f"  {'Cohen Kappa':<25} {baseline_metrics['kappa']:<20.4f} {unet_metrics['kappa']:<20.4f}")
    print(f"  {'Quadratic W Kappa':<25} {baseline_metrics['quadratic_kappa']:<20.4f} {unet_metrics['quadratic_kappa']:<20.4f}")

    for i, label in enumerate(label_order):
        print(f"  {'F1-' + label:<25} {baseline_metrics['f1'][i]:<20.4f} {unet_metrics['f1'][i]:<20.4f}")


if __name__ == "__main__":
    main()
