# 肾动态显像 AI 辅助诊断系统

基于视觉语言模型（VLM）的 99mTc-EC 肾动态显像自动诊断。将肾脏功能状态分为四类：**正常肾脏**、**功能性梗阻**、**机械性梗阻**、**混合性梗阻**。

## 项目结构

```
.
├── ablation/          # 消融实验（不同 VLM / TAC 配置对比）
├── preprocessing/     # 数据预处理（帧融合、肾区裁剪、特征提取）
├── segmentation/      # UNet 分割模型
├── evaluation/        # 评估工具（Bootstrap CI + McNemar）
├── diagnosis/         # 诊断对比分析
├── tests/             # 公平对比实验 pipeline
├── utils/             # 辅助脚本
├── published/         # 金标准 + 预测结果（可公开）
├── results/           # 评估结果 CSV / 图表
├── paper/             # 论文草稿 / 专利
└── docs/              # 方法论文档
```

## 数据

原始医学影像（>5 万张，约 300 MB）因患者隐私不在仓库中，仅提供：

| 内容 | 位置 | 说明 |
|------|------|------|
| 金标准标签 | [`published/gold/`](published/gold/) | 3 个 CSV，249 例 / 496 肾 |
| 模型预测 | [`published/results/`](published/results/) | 26 组实验 JSON |
| 评估汇总 | [`results/`](results/) | 各实验 eval CSV |

详见 [`published/README.md`](published/README.md)。

### 数据概况

| 子集 | 病例 | 肾脏 | 用途 | 命名 |
|------|------|------|------|------|
| 1-52 | 52 | 104 | 训练/开发 | `功能_1`, `机械_1`... |
| 0-100 | 97 | 194 | 测试集 | 数字 1–100（有缺失） |
| 100-150 | 50 | 100 | 扩展测试 | 数字 1–50 |
| 150-200 | 50 | 98 | 扩展测试 | 数字 51–69, 100 |

数据分批次从医生处获取，持续更新中。

### 金标准格式

**Type 1 — 患者级（Tab 分隔）**：`0-100患者信息.csv`、`100-200患者信息.csv`

| case_id | 性别（男1女0） | 年龄 | 左肾 | 右肾 |
|---------|---------------|------|------|------|
| 1 | 1 | 8d | 机械 | |
| 12 | 1 | 2 | 混合 | 功能 |

简称映射：`机械`→机械性梗阻, `功能`→功能性梗阻, `混合`→混合性梗阻, `正常`→正常肾脏, `排泄不明显`→功能性梗阻。空单元格=正常肾脏。

**Type 2 — 肾脏级（逗号分隔）**：`1-52患者信息.csv`

| case_name | kidney_side | true_label |
|-----------|-------------|------------|
| 功能_1 | left | 功能性梗阻 |

## 各模块

### ablation/ — 消融实验

两阶段 VLM 诊断 pipeline（Stage 1: 影像证据提取 → Stage 2: LLM 分类综合），通过不同配置对比关键设计选择：

| 脚本 | 说明 |
|------|------|
| `baseline_qwen.py` | Qwen 基线 |
| `baseline_gpt.py` | GPT 基线 |
| `baseline_claude.py` | Claude 基线 |
| `full_vlm.py` | 完整 pipeline（影像 + TAC） |
| `no_tac.py` | 消融：不看 TAC |
| `tac_only.py` | 消融：仅 TAC（不看图像） |
| `points_130.py` | 使用全部 130 帧（非融合 26 帧） |

### preprocessing/ — 数据预处理

| 脚本 | 说明 |
|------|------|
| `fuse_frames.py` | 130 帧 → 26 帧融合 |
| `fuse_batch.py` | 批量融合 |
| `crop_kidney.py` | ROI 标注 → 肾区裁剪（512×512） |
| `kidney_features.py` | 肾脏特征图生成 |
| `align_datasets.py` | 跨数据集格式对齐 |

### segmentation/ — UNet 分割

| 脚本 | 说明 |
|------|------|
| `model.py` | UNet 定义 |
| `train.py` | 训练 |
| `test.py` | 推理 |
| `dataset.py` | DataLoader |
| `loss.py` | 损失函数 |
| `prepare_data.py` | 数据准备 |

用于自动分割肾脏 ROI，或作为 VLM 诊断前的预处理步骤。

### evaluation/ — 评估

| 脚本 | 说明 |
|------|------|
| `evaluate.py` | 主评估：Accuracy / Kappa / F1 / Bootstrap 95% CI / McNemar + Holm |
| `batch_evaluate.py` | 批量评估多个实验 |
| `compare_tac.py` | TAC 曲线对比 |

### tests/fair_compare/ — 公平对比

完整的两阶段 pipeline，用于不同方法间的公平比较实验。含 UNet ROI 预测对比（医生标注 vs 模型预测）。

## 环境

```bash
pip install openai pillow matplotlib numpy scikit-learn pandas tqdm torch

export DASHSCOPE_API_KEY="your-key"
export VLM_MODEL="qwen3.6-plus"
```

API 端点：`https://dashscope.aliyuncs.com/compatible-mode/v1`（DashScope OpenAI 兼容模式）。

## 运行

```bash
# 消融实验
python ablation/baseline_qwen.py "$DASHSCOPE_API_KEY"

# 评估单个结果
python evaluation/evaluate.py \
  --gold-csv published/gold/0-100患者信息.csv \
  --results-dir published/results/0-100_test/results_ai_ai_baseline_gpt_test1

# 批量评估
python evaluation/batch_evaluate.py
```

## 临床背景

- 注射 99mTc-EC 后连续采集 130 帧肾脏动态图像
- 图像左侧 = 患者右肾，图像右侧 = 患者左肾（评估时需注意交换）
- 第 80–90 帧为利尿剂注射窗口
- TAC（Time-Activity Curve）反映肾区放射性随时间变化，是区分梗阻类型的关键

## 输出类别

| 类别 | 临床意义 |
|------|---------|
| 正常肾脏 | 功能正常，排泄通畅 |
| 功能性梗阻 | 功能受损，无器质性梗阻 |
| 机械性梗阻 | 存在器质性梗阻 |
| 混合性梗阻 | 功能 + 机械因素并存 |
