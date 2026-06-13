# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research project for **renal dynamic scintigraphy diagnosis** using vision-language models (VLMs). The goal is to classify kidneys from 99mTc-EC dynamic image sequences into four categories: 正常肾脏 (normal), 功能性梗阻 (functional obstruction), 机械性梗阻 (mechanical obstruction), 混合性梗阻 (mixed obstruction).

## Environment

- No virtual environment, package manager, or dependency file. Install dependencies ad-hoc with `pip install openai pillow matplotlib numpy scikit-learn pandas tqdm`.
- API calls use the `openai` Python SDK pointed at Alibaba DashScope's compatible-mode endpoint (`https://dashscope.aliyuncs.com/compatible-mode/v1`).
- **API key**: Set via `DASHSCOPE_API_KEY` environment variable, or passed as `sys.argv[1]` in some scripts. Several older scripts have hardcoded keys — never commit those.
- Primary model: `qwen3.6-plus` (set via `VLM_MODEL` env var in some scripts).

## Pipeline evolution (newest → oldest)

The repo contains multiple iterations of the diagnostic pipeline. The main active ones:

1. **`renal_dynamic_report_pipeline.py`** / **`renal_dynamic_report_pipeline (1).py`** — Structured two-stage pipeline (Stage 1: image-based assessment with uncertainty → Stage 2: LLM review/synthesis). Works on `datasets_12/` (60 cases). Outputs structured JSON reports.

2. **`ai_ai.py`** / **`ai_ai_1.py`** — Variants of the two-stage pipeline with Stage 2 acting as a reviewer/critic (LLM reviews Stage 1 outputs). Similar to the report pipeline but with different prompt engineering.

3. **`stable_fusion_pipeline_v2.py`** — Three-view fusion: VLM primary read + TAC-based phenotype classifier + clinical feature summary, followed by a joint synthesis step. Temperature set to 0.0 across all stages for deterministic output.

4. **`renal_multimodal_joint_reasoning_v1.py`** — Three-branch multimodal approach: independent image evidence branch + TAC evidence branch + joint reasoning synthesis. Each branch runs at temperature 0.0.

5. **`clean_kidney_pipeline.py`** — Uncertainty-driven framework: Stage 1 (VLM image read with self-assessment), Stage 2 (rule-based uncertainty quantification), Stage 3 (conditional TAC arbitration only when uncertainty is high).

6. **`api.py`** / **`api_std.py`** — Simpler single-pass VLM pipelines (direct image → diagnosis prompt with no multi-stage reasoning).

7. **`fixed.py`** — Similar to the stable fusion pipeline, a fixed/revised version.

## Dataset structure

Two main dataset directories:

- **`dataset/`** — Original 60 cases: `功能_1` through `功能_20`, `机械_1` through `机械_20`, `混合_1` through `混合_20`.
- **`datasets_12/`** — Expanded 60-case set with a different numbering scheme (used by the newer pipelines).

Each case directory contains:
- `images/` — Raw 130-frame sequence images
- `images_fused_26/` — 26 fused images (130 raw frames merged into 26 time steps per standard protocol)
- `images_fused_5/` — 5 fused images (coarser temporal aggregation)
- `labels/` — JSON ROI annotations (left/right kidney bounding regions)
- `kidney_crop/` — Cropped kidney regions from preprocessing
- `process_params.json` — Per-case processing parameters

## Running a pipeline

Most pipeline scripts are self-contained and run end-to-end on all cases:

```bash
# Set API key first
export DASHSCOPE_API_KEY="your-key"

# Run the main structured pipeline (newest)
python renal_dynamic_report_pipeline.py

# Some scripts accept a case glob as second arg
python renal_dynamic_report_pipeline.py "$API_KEY" "机械_*"

# Run older single-pass pipeline on a single case
python api.py
```

Each pipeline writes results into a `results_*` subdirectory under its configured `DATASET_DIR`.

## Evaluation

**`ratio.py`** — Computes accuracy, confusion matrix, and per-class precision/recall/F1 from `eval.csv`. Uses sklearn metrics with fixed label order: 正常肾脏 (3), 功能性梗阻 (0), 混合性梗阻 (2), 机械性梗阻 (1).

**`eval.csv`** — Ground truth + prediction comparison table with columns `true_label` and `pred_label` (numeric codes 0-3).

## Utility scripts

- **`create_fused_images_custom.py`** — Fuses 130 raw frames into 26 fused images using configurable frame ranges.
- **`preprocess_images.py`** — Crops kidney ROIs from images based on label JSON annotations, enhances contrast.
- **`copy_images.py`** / **`copy_labels.py`** — One-off scripts to copy `images/` and `labels/` directories from `dataset/` to `datasets_12/`.
- **`check_dataset.py`** — Inspects dataset directory structure and case naming.

## Key design patterns

- **Temperature = 0.0** is used in newer pipelines for deterministic/reproducible research outputs. Earlier pipelines used higher temperatures (0.3–0.6).
- **JSON retry logic**: Most pipelines include `MAX_JSON_RETRIES` with regex fallback parsing when the VLM doesn't produce valid JSON.
- **Strict output vocabularies**: Each pipeline defines `ALLOWED_*` sets and validates VLM outputs against them, falling back to defaults (typically "难以判断" / "uncertain") on mismatch.
- **Diuretic timing prior**: Frame ~80–90 in the original 130-frame sequence is the diuretic administration point. This is a fixed protocol constant used across pipelines.
- **Left/right convention**: In the images, image-left = patient's right kidney, image-right = patient's left kidney.
