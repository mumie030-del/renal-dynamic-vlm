# Published Evaluation Data

肾动态显像 VLM 诊断实验的金标准标签和模型预测结果。原始医学影像不包含在内。

## 目录

```
published/
├── gold/          # 金标准标签 CSV（3 个文件）
└── results/       # 模型预测结果 JSON（26 组实验）
    ├── 0-100_test/    # 测试集
    ├── 1-52/          # 训练/开发集
    ├── 100-150/       # 扩展测试集
    └── 150-200/       # 扩展测试集
```

## 金标准格式

### Type 1 — 患者级（Tab 分隔）

`0-100患者信息.csv` 和 `100-200患者信息.csv`：

| 列 | 说明 |
|----|------|
| case_id | 患者编号 |
| 性别 | 1=男, 0=女 |
| 年龄 | 数值（d=天, m=月, y=岁） |
| 左肾 | 诊断标签（简称） |
| 右肾 | 诊断标签（简称） |

标签简称映射：`机械`→机械性梗阻, `功能`→功能性梗阻, `混合`→混合性梗阻, `正常`→正常肾脏, `排泄不明显`→功能性梗阻。空单元格=正常肾脏。

### Type 2 — 肾脏级（逗号分隔）

`1-52患者信息.csv`：

| 列 | 说明 |
|----|------|
| case_name | 病例名（如 `功能_1`） |
| kidney_side | left / right |
| true_label | 诊断标签（全称） |

## 预测结果格式

每个 `results_*/` 目录包含每个病例的 JSON 文件（如 `1.json`），结构如下：

```json
{
  "case_name": "1",
  "stage1_validated": {
    "right_kidney": {
      "early_visualization": "及时",
      "peak_timing": "晚期达峰",
      "post_diuretic_response": "几乎无明显改善",
      ...
    },
    "left_kidney": { ... },
    "confidence": "中"
  },
  "stage2_validated": {
    "right_kidney": {
      "final_label": "机械性梗阻",
      "confidence": "高",
      "reasoning": "..."
    },
    "left_kidney": { ... }
  },
  "final_report": "右肾：机械性梗阻；..."
}
```

四类诊断标签：`正常肾脏` / `功能性梗阻` / `机械性梗阻` / `混合性梗阻`。

## 使用评估工具

```bash
# 对单个数据集评估
python evaluation/evaluate.py \
  --gold-csv published/gold/0-100患者信息.csv \
  --results-dir published/results/0-100_test/results_ai_ai_baseline_gpt_test1

# Type 2 格式（1-52）
python evaluation/evaluate.py \
  --gold-csv published/gold/1-52患者信息.csv \
  --results-dir published/results/1-52/results_ai_ai

# 对比多个实验结果（自动 McNemar 检验）
python evaluation/evaluate.py \
  --gold-csv published/gold/0-100患者信息.csv \
  --results-dir published/results/0-100_test/results_ai_ai_baseline_gpt_test1 \
  --gold-csv published/gold/0-100患者信息.csv \
  --results-dir published/results/0-100_test/results_ai_ai_baseline_qwen_5.30_test3
```
