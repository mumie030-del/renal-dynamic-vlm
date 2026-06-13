import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SCRIPT_DIR = os.path.dirname(__file__)
DATASET_DIR = "/root/autodl-tmp/LLM/datasets_12"
GROUND_TRUTH_FILE = os.path.join(DATASET_DIR, "标注模板.csv")

EXPERIMENTS = {
    "baseline (doctor ROI)": os.path.join(SCRIPT_DIR, "results_ai_ai_baseline"),
    "notac (no TAC)":       os.path.join(SCRIPT_DIR, "results_ai_ai_notac"),
    "unetltae":             os.path.join(SCRIPT_DIR, "results_ai_ai_unetltae"),
    "tac_only":             os.path.join(SCRIPT_DIR, "results_ai_ai_tac_only"),
}

LABEL_MAP = {
    "正常肾脏": "正常肾脏",
    "正常梗阻": "正常肾脏",
    "功能性梗阻": "功能性梗阻",
    "机械性梗阻": "机械性梗阻",
    "混合性梗阻": "混合性梗阻",
}

SIDES = {"left": "left_kidney", "right": "right_kidney"}


def load_ground_truth(path):
    gt = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row["case_name"].strip()
            side = row["kidney_side"].strip()
            label = row["true_label"].strip()
            mapped = LABEL_MAP.get(label)
            if mapped is None:
                print(f"  [警告] 跳过未知标签 '{label}' 在 {case} {side}")
                continue
            gt[(case, side)] = mapped
    return gt


def load_predictions(results_dir):
    preds = {}
    if not os.path.isdir(results_dir):
        print(f"  [错误] 结果目录不存在: {results_dir}")
        return preds
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        case = data.get("case_name", fname.replace(".json", ""))
        sv = data.get("stage2_validated", {})
        for side_key, side_label in [("left_kidney", "left"), ("right_kidney", "right")]:
            kid = sv.get(side_key, {})
            label = kid.get("final_label")
            if label:
                mapped = LABEL_MAP.get(label)
                if mapped:
                    preds[(case, side_label)] = mapped
    return preds


def evaluate(gt, preds, name):
    y_true, y_pred = [], []
    skipped = []
    for key in sorted(gt):
        if key in preds:
            y_true.append(gt[key])
            y_pred.append(preds[key])
        else:
            skipped.append(key)
    if not y_true:
        print(f"\n  {name}: 无有效预测")
        return None
    classes = ["正常肾脏", "功能性梗阻", "机械性梗阻", "混合性梗阻"]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cr = classification_report(y_true, y_pred, labels=classes, target_names=classes, digits=4)
    n = len(y_true)
    print(f"\n{'='*60}")
    print(f"  {name}  (n={n})")
    print(f"{'='*60}")
    if skipped:
        print(f"  [警告] {len(skipped)} 条无预测: {skipped[:3]}...")
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>18}  " + "  ".join(f"{c[:4]:>6}" for c in classes))
    for i, row_label in enumerate(classes):
        row_vals = "  ".join(f"{v:>6}" for v in cm[i])
        print(f"  {row_label:>18}  {row_vals}")
    print(f"\n  Classification Report:")
    for line in cr.split("\n"):
        print(f"    {line}")
    return {"name": name, "accuracy": acc, "n": n, "cm": cm, "report": cr}


def main():
    print("加载 ground truth...")
    gt = load_ground_truth(GROUND_TRUTH_FILE)
    print(f"  {len(gt)} 条标注")

    results = []
    for name, res_path in EXPERIMENTS.items():
        print(f"\n加载 {name} ({res_path})...")
        preds = load_predictions(res_path)
        print(f"  {len(preds)} 条预测")
        r = evaluate(gt, preds, name)
        if r:
            results.append(r)

    if results:
        print(f"\n\n{'='*60}")
        print(f"  Accuracy 对比汇总")
        print(f"{'='*60}")
        print(f"  {'Method':<30} {'n':>5}  {'Accuracy':>10}")
        print(f"  {'-'*30} {'-'*5}  {'-'*10}")
        for r in sorted(results, key=lambda x: -x["accuracy"]):
            print(f"  {r['name']:<30} {r['n']:>5}  {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")


if __name__ == "__main__":
    main()
