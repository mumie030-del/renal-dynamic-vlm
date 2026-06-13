"""批量评估：自动匹配金标准与所有预测结果目录，输出汇总对比表。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (
    load_from_raw_format,
    load_from_type2_format,
    CLASSES,
    IDX_TO_CN,
)

BASE = Path(__file__).resolve().parent.parent / "data"


def quick_metrics(df):
    """快速计算点估计（无 bootstrap），返回 dict。"""
    yt = df["true_label"].values
    yp = df["pred_label"].values

    result = {
        "accuracy": accuracy_score(yt, yp),
        "kappa_linear": cohen_kappa_score(yt, yp, weights="linear"),
        "kappa_quadratic": cohen_kappa_score(yt, yp, weights="quadratic"),
        "macro_f1": f1_score(yt, yp, average="macro"),
        "weighted_f1": f1_score(yt, yp, average="weighted"),
    }

    for i, cls_name in enumerate(CLASSES):
        tp = ((yt == i) & (yp == i)).sum()
        fp = ((yt != i) & (yp == i)).sum()
        fn = ((yt == i) & (yp != i)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1v = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0.0
        result[f"{cls_name}_f1"] = f1v

    return result


def run_one(name, gold_csv, results_dir, is_type2, swap_lr, row_offset=0, left_col=3, right_col=4):
    if is_type2:
        df = load_from_type2_format(gold_csv, results_dir, swap_lr=swap_lr)
    else:
        df = load_from_raw_format(gold_csv, results_dir, swap_lr=swap_lr,
                                  row_offset=row_offset,
                                  left_kidney_col=left_col, right_kidney_col=right_col)

    m = quick_metrics(df)
    n_errors = int((df["true_label"] != df["pred_label"]).sum())
    return {
        "name": name,
        "n_cases": df["case_id"].nunique(),
        "n_kidneys": len(df),
        "n_errors": n_errors,
        **m,
    }


def main():
    tasks = []

    # 0-100 test set (Type 1) — swap_lr=True, row_offset=0, 左=col3 右=col4
    gold_0_100 = str(BASE / "0-100（测试集）/0-100患者信息.csv")
    for d in sorted((BASE / "0-100（测试集）").glob("results_*")):
        tasks.append((f"0-100/{d.name}", gold_0_100, str(d), False, True, 0, 3, 4))

    # 1-52 (Type 2) — swap_lr=False
    gold_1_52 = str(BASE / "1-52/1-52患者信息.csv")
    for d in sorted((BASE / "1-52").glob("results_*")):
        tasks.append((f"1-52/{d.name}", gold_1_52, str(d), True, False, 0, 3, 4))

    # 100-150 (Type 1) — swap_lr=True, row_offset=0 (新金标 case_id 直接对应)
    gold_100_200 = str(BASE / "100-150/100-200患者信息.csv")
    for d in sorted((BASE / "100-150").glob("results_*")):
        tasks.append((f"100-150/{d.name}", gold_100_200, str(d), False, True, 0, 3, 4))

    # 150-200 (Type 1) — 同上
    for d in sorted((BASE / "150-200").glob("results_*")):
        tasks.append((f"150-200/{d.name}", gold_100_200, str(d), False, True, 0, 3, 4))

    print(f"{'='*120}")
    print(f"  批量评估 — {len(tasks)} 个结果集")
    print(f"{'='*120}")

    results = []
    for label, gold_csv, results_dir, is_type2, swap_lr, row_offset, left_col, right_col in tasks:
        n_json = len(list(Path(results_dir).glob("*.json")))
        try:
            r = run_one(label, gold_csv, results_dir, is_type2, swap_lr=swap_lr,
                        row_offset=row_offset, left_col=left_col, right_col=right_col)
            results.append(r)
            acc = r["accuracy"]
            kappa = r["kappa_linear"]
            print(f"  {label:<55}  n={r['n_kidneys']:>3d}  Acc={acc:.1%}  κ={kappa:.3f}  err={r['n_errors']}")
        except Exception as e:
            print(f"  {label:<55}  SKIP: {e}")

    # 打印汇总表
    print(f"\n{'='*120}")
    print(f"  汇总对比表")
    print(f"{'='*120}")
    header = f"  {'方法':<50} {'N':>4} {'Acc%':>7} {'κ_lin':>7} {'κ_quad':>7} {'MacroF1':>7} {'Err':>4}"
    print(header)
    print(f"  {'-'*95}")

    for r in results:
        print(
            f"  {r['name']:<50}"
            f" {r['n_kidneys']:>4}"
            f" {r['accuracy']*100:>6.1f}"
            f" {r['kappa_linear']:>7.3f}"
            f" {r['kappa_quadratic']:>7.3f}"
            f" {r['macro_f1']:>7.3f}"
            f" {r['n_errors']:>4}"
        )

    # Per-class F1 表
    print(f"\n{'='*120}")
    print(f"  每类 F1 对比")
    print(f"{'='*120}")
    header2 = f"  {'方法':<50} " + " ".join(f"{c:>8}" for c in CLASSES)
    print(header2)
    print(f"  {'-'*90}")
    for r in results:
        f1s = " ".join(f"{r[f'{c}_f1']*100:>7.1f}" for c in CLASSES)
        print(f"  {r['name']:<50} {f1s}")

    print(f"\n{'='*120}")
    print(f"  评估完成")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
