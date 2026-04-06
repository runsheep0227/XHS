# 评论分析模块说明

本目录实现「小红书 AIGC 相关评论」的三分类情感流水线：原始数据合并与清洗、大模型（可选）打伪标签、MacBERT 类预训练模型微调、测试集评估与可视化、以及对全量清洗评论的批量推理与统计。

**标签语义（业务极性）**

| `sentiment_polarity`（与原 `label` 一致） | 含义 |
| --- | --- |
| `1` | 正向（赞美、期待、求带等） |
| `0` | 中性（客观描述、工具性问题等） |
| `-1` | 负向（抵触、吐槽、反讽等） |

**模型内部类别 id**（`sentiment_class_id` / HuggingFace `config.id2label`）

| 模型输出索引 | 对应极性 |
| --- | --- |
| `0` | `-1` |
| `1` | `0` |
| `2` | `1` |

---

## 当前仓库中的数据与指标（仅供参考）

以下数字来自本仓库内已生成的报告文件，**随你重新跑清洗、划分或训练会变化**。

- **二次清洗**（`bert_data/clean_data_report.txt`）：输入 `cleaned_com.json` 约 76,895 条 → 有效约 **75,245** 条，留存率约 97.85%。
- **全量预测统计**（`comment_results/prediction_stats.txt`）：已对约 **75,245** 条打标；极性占比约 负 25.52% / 中 35.28% / 正 39.21%；涉及约 **10,851** 个不同 `note_id`。
- **测试集评估**（`results/evaluation_report.txt`）：Accuracy **0.7450**，Macro-F1 **0.7384**（测试集 800 条，与当前 `test.csv` 一致）。

默认微调基座为 **`hfl/chinese-macbert-large`**（见 `train_roberta.py` 中 `DEFAULT_MODEL_NAME`），可通过 `--model_name` 改为 `hfl/chinese-roberta-wwm-ext` 等中文分类常用模型。

---

## 处理流程（推荐顺序）

1. **`clean_com.py`**：读取 `rawdata/search_comments_*.json`，字段提取、去 `@`、表情占位符转文字、质量过滤与去重 → **`cleaned_com.json`**（与脚本同级）。
2. **`clean_data.py`**：读取 `cleaned_com.json`，符号与长度规则、无意义词表过滤、再次去重 → **`bert_data/final_cleaned_comments.json`**、**`bert_data/bert_train_ready.csv`**、**`bert_data/clean_data_report.txt`**。
3. **`random_sample.py`**：从 `final_cleaned_comments.json` 随机抽样（脚本内 `SAMPLE_SIZE`，当前为 **8000**）→ **`bert_data.llm_sample_data.json`**。
4. **`llm_annotate.py`**：调用兼容 OpenAI API 的本地服务（如 LM Studio），对抽样评论打 `-1/0/1`（失败行可能为 `2` 或 `-2`）→ **`bert_data/llm_labeled_result.csv`** 与 **`bert_data/llm_labeled_result_stats.csv`**。  
   **注意**：脚本内 `OUTPUT_DIR` 曾写死为绝对路径，换机器时请改为基于 `__file__` 的目录，与输入 `bert_data` 一致。
5. **`split_dataset.py`**：读取 `llm_labeled_result.csv`，只保留标签 `{-1,0,1}` 且非空 `content`，按 **8:1:1** 分层划分；可选 **`--train_neg_multiplier`** 仅对训练集负样本过采样。输出 **`bert_data/train.csv`**、`val.csv`、`test.csv`。
6. **`train_roberta.py`**：读取 `train.csv` / `val.csv`，训练三分类模型；训练过程 checkpoint 在 **`checkpoint_temp/`**（或带 `run_name` 的子目录），验证最优模型导出到 **`saved_model/`**（默认 `--export_dir saved_model`）。支持类权重、Focal、早停、`run_summary.json` 与 `checkpoint_eval_summary.json` 等。
7. **`evaluate_model.py`**：在 **`bert_data/test.csv`** 上评估 **`saved_model/`**，生成 **`results/`** 下报告与多张图（混淆矩阵、ROC、误判 CSV 等）。
8. **`predict_all.py`**：对 **`bert_data/final_cleaned_comments.json`**（或可切换为含 `txt`/`content` 的 CSV）批量推理，写入 **`comment_results/predicted_comments.json`**，每条记录增加 `sentiment_class_id`、`sentiment_polarity`、`sentiment_text`；支持断点续跑（`*.predict_ckpt.json`）。
9. **`summarize_predictions.py`**（可选）：汇总 `predicted_comments.json` → **`comment_results/prediction_stats.txt`**、`_polarity.csv`、`_class.csv`、`_notes.csv`。
10. **`recalc_llm_label_stats.py`**（可选）：在你手工修正 **`llm_labeled_result.csv`** 后，重算 **`llm_labeled_result_stats.csv`**，并可导出失败行列表 **`llm_labeled_failure_ids.txt`**、**`llm_labeled_failures_report.csv`**。

---

## 目录与文件说明

### 脚本（`comment/` 根目录）

| 文件 | 作用 |
| --- | --- |
| `clean_com.py` | 合并 `rawdata`，一级清洗 → `cleaned_com.json` |
| `clean_data.py` | 二级清洗 → `bert_data/` 内 JSON/CSV/报告 |
| `meaningless_word.txt` | 低信息熵过滤用词表（`#` 行为注释） |
| `random_sample.py` | 构造 LLM 标注用子集 |
| `llm_annotate.py` | 本地 API 批量标注 |
| `recalc_llm_label_stats.py` | 修正标注表后重算统计与失败报告 |
| `split_dataset.py` | 划分 train/val/test，可选负样本过采样 |
| `train_roberta.py` | 微调与导出 |
| `evaluate_model.py` | 测试集评估与图表 |
| `predict_all.py` | 全量推理 |
| `summarize_predictions.py` | 预测结果分布与按笔记聚合 |

### `rawdata/`

按日抓取的 **`search_comments_YYYY-MM-DD.json`**。结构支持「根为数组」或「根为对象且列表在 `data` 字段」。

### `bert_data/`（清洗与训练数据）

| 文件 | 说明 |
| --- | --- |
| `final_cleaned_comments.json` | 清洗后全量记录：`note_id`、`nickname`、`content`、`ip_location`、`original_content` 等 |
| `bert_train_ready.csv` | 两列 `note_id`, `txt`（无标签，供预测或二次合并标签） |
| `clean_data_report.txt` | 二级清洗统计（**注意**：报告正文里个别文件名描述可能与脚本实际输出名不一致，以脚本为准） |
| `llm_sample_data.json` | 供 LLM 标注的抽样 |
| `llm_labeled_result.csv` | LLM 标注结果（含 `label`、`raw_response` 等） |
| `llm_labeled_result_stats.csv` | 标注分布统计 |
| `train.csv` / `val.csv` / `test.csv` | 划分后训练与评估集（保留 `split_dataset` 写入的列） |

手工修正标注后可运行 `recalc_llm_label_stats.py`；若存在异常标签，可能额外生成 `llm_labeled_failure_ids.txt`、`llm_labeled_failures_report.csv`。

### `saved_model/`（推理与评估加载目录）

由 `train_roberta.py` 将验证集上最优 checkpoint 导出至此。至少包含 **`config.json`**、分词器文件；完整训练中还应含 **`model.safetensors`** 或 **`pytorch_model.bin`** 等权重（若仓库中未提交大文件，需本地训练后生成）。

### `checkpoint_temp/`（训练中间产物）

各 `checkpoint-*` 子目录、`trainer_state.json`、`run_summary.json`、`checkpoint_eval_summary.json` 等；用于断点续训或对比用不同步数的验证指标。

### `results/`（离线评估产出）

| 文件 | 说明 |
| --- | --- |
| `evaluation_report.txt` | Accuracy、Macro-F1、按类 precision/recall/F1 |
| `01_confusion_matrix.png` ~ `08_metrics_radar.png` | 混淆矩阵、指标柱状图、置信度、ROC/PR、雷达图等 |
| `09_error_analysis.txt` | 误判分析摘要 |
| `misclassified_test.csv` | 误判样本明细 |

### `comment_results/`（全量预测与汇总）

| 文件 | 说明 |
| --- | --- |
| `predicted_comments.json` | 在清洗结构基础上增加 `sentiment_*` 字段 |
| `prediction_stats.txt`、`*_polarity.csv`、`*_class.csv`、`*_notes.csv` | 由 `summarize_predictions.py` 生成（前缀默认 `prediction_stats`） |

推理过程中可能短暂存在 **`predicted_comments.predict_ckpt.json`**，跑完后成功则删除。

---

## 依赖与环境

- **Python**：`torch`、`transformers`、`datasets`、`pandas`、`scikit-learn`、`tqdm`；评估与作图需要 `matplotlib`、`seaborn`。
- **LLM 标注**：`openai` 客户端 + 本地 **`BASE_URL`**（默认 `http://127.0.0.1:1234/v1`）与 **`MODEL_NAME`**；`llm_annotate.py` 中 `MAX_WORKERS` 当前为 1，可按机器与接口稳定性调高。

安装示例：

```bash
pip install torch transformers datasets pandas scikit-learn tqdm matplotlib seaborn openai
```

---

## 快速执行命令（在 `comment/` 下）

```bash
python clean_com.py
python clean_data.py
python random_sample.py
python llm_annotate.py
python split_dataset.py
python train_roberta.py
python evaluate_model.py
python predict_all.py
python summarize_predictions.py
```

`split_dataset` 示例（训练集负样本放大 1.5 倍）：

```bash
python split_dataset.py --train_neg_multiplier 1.5 --seed 42
```

`predict_all` 可指定输入与模型目录：

```bash
python predict_all.py --input bert_data/final_cleaned_comments.json --model-dir saved_model --output-dir comment_results
```

---

## 上游可视化 / 前端接入建议

- **极性饼图、柱状图**：直接读 **`comment_results/prediction_stats_polarity.csv`** 或 `predicted_comments.json` 聚合。
- **按笔记评论量分布、热力或列表**：**`comment_results/prediction_stats_notes.csv`**（`note_id` + `comment_count`）。
- **模型类 id 分布**：**`prediction_stats_class.csv`**（与 `sentiment_class_id` 一致）。
- **与真实标签对比**（演示用）：**`bert_data/test.csv`** + 同一模型对 `content` 推理结果对比。
- **训练/标注质量**：**`results/`** 下图表与 **`misclassified_test.csv`**；**`llm_labeled_result.csv`** 用于抽检 LLM 噪声。

---

## 使用注意

1. **路径可移植**：`clean_com.py`、`clean_data.py`、`split_dataset.py`、`predict_all.py` 等多基于脚本目录拼路径；`llm_annotate.py` 的 `OUTPUT_DIR` 建议改为与仓库相对路径一致。
2. **标签与 CSV**：划分与评估脚本会将 `label` 转为数值；应用 Excel 编辑后若出现非法 `label`，`split_dataset` 会剔除并打印报告。
3. **长文本**：训练与推理默认 **`max_length=512`**（`predict_all` / `train_roberta` 可改），与短于 128 的旧配置不同。
4. **全量 JSON 体积**：`predicted_comments.json` 行数大，前端宜通过后端分页或只加载聚合 CSV。

如需把 README 中的「当前指标」与某次固定实验对齐，请在复现实验后更新本节「当前仓库中的数据与指标」中的引用文件或数字。
