# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research project for **renal dynamic scintigraphy diagnosis** using vision-language models (VLMs). Classifies kidneys from 99mTc-EC dynamic image sequences into four categories: 正常肾脏 (normal), 功能性梗阻 (functional obstruction), 机械性梗阻 (mechanical obstruction), 混合性梗阻 (mixed obstruction).

## Environment

- No virtual environment, package manager, or dependency file. Install dependencies with `pip install openai pillow matplotlib numpy scikit-learn pandas tqdm torch`.
- API: `openai` Python SDK → `https://dashscope.aliyuncs.com/compatible-mode/v1` (DashScope OpenAI compatible mode).
- **API key**: `DASHSCOPE_API_KEY` env var, or `sys.argv[1]` in some ablation scripts.
- Primary model: `qwen3.6-plus` (via `VLM_MODEL` env var).

## Repository structure

```
ablation/        — VLM diagnostic ablation experiments (various models, TAC configs)
preprocessing/   — Image preprocessing: frame fusion, kidney cropping, feature extraction
segmentation/    — UNet+LTAE kidney ROI segmentation (model, train, test)
evaluation/      — Evaluation: Bootstrap CI, McNemar test, batch comparison
diagnosis/       — Diagnostic comparison tools
tests/fair_compare/ — Full fair-comparison pipeline (UNet ROI vs doctor ROI)
utils/           — Utility scripts (dataset checker, image re-generation)
published/       — Public: gold standard CSVs + model prediction JSONs (26 experiments)
results/         — Evaluation result CSVs and comparison charts
paper/           — Manuscript drafts and patent documents
docs/            — Methodology docs, doctor guidelines
data/            — Raw medical images (local only, not in git)
```

## Dataset

Raw medical images (~50K files, 300 MB) are in `data/` (gitignored). Public evaluation data is in `published/`:

| Dataset | Cases | Kidneys | Naming | Gold CSV |
|---------|-------|---------|--------|----------|
| 0-100 (test) | 97 | 194 | Numbers 1–100 | `published/gold/0-100患者信息.csv` (Type 1, tab-separated) |
| 1-52 (train/dev) | 52 | 104 | `功能_1`, `机械_1`... | `published/gold/1-52患者信息.csv` (Type 2, comma-separated) |
| 100-150 | 50 | 100 | Numbers 1–50 | `published/gold/100-200患者信息.csv` (Type 1) |
| 150-200 | 50 | 98 | Numbers 51–69, 100 | Same as above |

Data is received in batches from clinicians and is still being added.

### Gold standard formats

**Type 1 (patient-level, tab-separated)**: `case_id`, `性别`, `年龄`, `左肾`, `右肾`. Short labels (`机械`→机械性梗阻, `功能`→功能性梗阻, `混合`→混合性梗阻, `正常`→正常肾脏, `排泄不明显`→功能性梗阻). Empty cell = normal.

**Type 2 (kidney-level, comma-separated)**: `case_name`, `kidney_side`, `true_label`. Full labels.

### Case directory structure (local only, in data/)

```
case_dir/
├── images/              # 130 raw dynamic frames
├── images_fused_26/     # 26 fused images
├── labels/              # JSON ROI annotations
├── kidney_crop/         # Cropped kidney regions (512×512)
└── process_params.json  # Processing parameters
```

## Key pipelines

### ablation/ — Main active pipelines

Two-stage architecture: Stage 1 (VLM image evidence extraction) → Stage 2 (LLM text-based classification):

| Script | Description |
|--------|-------------|
| `baseline_qwen.py` | Qwen baseline (most frequently used) |
| `baseline_gpt.py` | GPT baseline |
| `baseline_claude.py` | Claude baseline |
| `full_vlm.py` | Full pipeline (image + TAC) |
| `no_tac.py` | Ablation: no TAC assistance |
| `tac_only.py` | Ablation: TAC only (no images) |
| `points_130.py` | Uses all 130 frames instead of fused 26 |

All use `DATASET_DIR` and `RESULTS_DIR` constants that may need updating per run.

### tests/fair_compare/ — Fair comparison pipeline

Complete pipeline for controlled comparison between methods (doctor-annotated ROI vs UNet-predicted ROI).

## Preprocessing chain

```
130 raw frames
    │
    ├─→ preprocessing/fuse_frames.py
    │   Grouped averaging → 26 fused images
    │
    └─→ preprocessing/crop_kidney.py
        Load ROI → crop dual kidneys → 40% padding → 512×512
```

## Evaluation

**Primary tool**: `evaluation/evaluate.py`

Computes: Accuracy, Cohen's Kappa (linear + quadratic), Macro/Weighted F1, per-class Sensitivity/Specificity/PPV/NPV/F1, confusion matrix, 95% Bootstrap CI (patient-level resampling, n=1000), McNemar paired test with Holm correction.

```bash
# Single evaluation
python evaluation/evaluate.py \
  --gold-csv published/gold/0-100患者信息.csv \
  --results-dir published/results/0-100_test/results_ai_ai_baseline_gpt_test1

# Type 2 format
python evaluation/evaluate.py \
  --gold-csv published/gold/1-52患者信息.csv \
  --results-dir published/results/1-52/results_ai_ai

# Batch evaluation
python evaluation/batch_evaluate.py
```

Two gold standard formats are auto-detected. `--no-swap` disables left/right kidney swap (image-left = patient's right kidney is the default convention).

## Key design patterns

- **Temperature = 0.0** — deterministic output for research reproducibility.
- **JSON retry logic** — `MAX_JSON_RETRIES` with regex fallback when VLM output isn't valid JSON.
- **Strict output vocabularies** — `ALLOWED_*` sets with whitelist validation, fallback to defaults on mismatch.
- **Diuretic timing prior** — Frame 80–90 in the 130-frame sequence is the diuretic injection point.
- **Left/right convention** — image-left = patient's right kidney, image-right = patient's left kidney. Prediction JSONs use image-space naming so `evaluate.py` auto-swaps by default.
