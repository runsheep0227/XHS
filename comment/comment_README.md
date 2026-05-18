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

以下数字来自本仓库内已生成的报告文件，**随你重新跑清洗、划分、训练或全量预测会变化**。

| 阶段 | 来源文件 | 当前约数 |
| --- | --- | --- |
| 一级清洗 | `cleaned_com.json` | **79,537** 条 |
| 二级清洗 | `bert_data/clean_data_report.txt` | 输入 79,537 → 有效 **76,401** 条（留存率 **96.06%**） |
| 全量预测 | `comment_results/prediction_stats.txt` | **75,245** 条已打标（若与 `final_cleaned_comments.json` 条数不一致，请重跑 `predict_all.py`） |
| 测试集评估 | `results/evaluation_report.txt` | Accuracy **0.7438**，Macro-F1 **0.7354**（测试集 **800** 条） |

全量预测极性占比（`prediction_stats.txt`）：负 **25.52%** / 中 **35.28%** / 正 **39.21%**；约 **10,851** 个不同 `note_id`。

默认微调基座为 **`hfl/chinese-macbert-large`**（见 `train_roberta.py` 中 `DEFAULT_MODEL_NAME`），可通过 `--model_name` 改为 `hfl/chinese-roberta-wwm-ext` 等中文分类常用模型。训练默认 **`max_length=512`**、`train_batch=12`、`grad_accum=2`（有效 batch 24），支持 bf16、可选 8bit Adam、类权重、Focal Loss、早停与 `run_summary.json` 等。

---

## 处理流程（推荐顺序）

1. **`clean_com.py`**：读取 `rawdata/search_comments_*.json`，字段提取、去 `@`、表情占位符转文字、质量过滤与去重 → **`cleaned_com.json`**（与脚本同级）。
2. **`clean_data.py`**：读取 `cleaned_com.json`，符号与长度规则、无意义词表过滤、再次去重 → **`bert_data/final_cleaned_comments.json`**、**`bert_data/bert_train_ready.csv`**、**`bert_data/clean_data_report.txt`**。
3. **`random_sample.py`**：从 `final_cleaned_comments.json` 随机抽样（脚本内 `SAMPLE_SIZE`，当前为 **8000**，`random.seed(42)`）→ **`bert_data/llm_sample_data.json`**。
4. **`llm_annotate.py`**：调用兼容 OpenAI API 的本地服务（如 LM Studio），对抽样评论打 `-1/0/1`（失败行可能为 `2` 或 `-2`）→ **`bert_data/llm_labeled_result.csv`** 与 **`bert_data/llm_labeled_result_stats.csv`**。  
   **输出**：`bert_data/llm_labeled_result.csv`（路径相对本脚本目录）。
5. **`split_dataset.py`**：读取 `llm_labeled_result.csv`，只保留标签 `{-1,0,1}` 且非空 `content`，按 **8:1:1** 分层划分；可选 **`--train_neg_multiplier`** 仅对训练集负样本过采样。输出 **`bert_data/train.csv`**、`val.csv`、`test.csv`。
6. **`train_roberta.py`**：读取 `train.csv` / `val.csv`，训练三分类模型；训练过程 checkpoint 在 **`checkpoint_temp/`**（或带 `run_name` 的子目录），验证最优模型导出到 **`saved_model/`**（默认 `--export_dir saved_model`）。
7. **`evaluate_model.py`**：在 **`bert_data/test.csv`** 上评估 **`saved_model/`**，生成 **`results/`** 下报告、多张 PNG 与 **`eval_viz_payload.json`**（供前端复现 ROC/PR 等）。
8. **`predict_all.py`**：对 **`bert_data/final_cleaned_comments.json`**（或可切换为含 `txt`/`content` 的 CSV）批量推理，写入 **`comment_results/predicted_comments.json`**，每条增加 `sentiment_class_id`、`sentiment_polarity`、`sentiment_text`；支持断点续跑（`*.predict_ckpt.json`）。
9. **`summarize_predictions.py`**（可选）：汇总 `predicted_comments.json` → **`comment_results/prediction_stats.txt`**、`_polarity.csv`、`_class.csv`、`_notes.csv`。
10. **`recalc_llm_label_stats.py`**（可选）：手工修正 **`llm_labeled_result.csv`** 后，重算 **`llm_labeled_result_stats.csv`**，并可导出失败行 **`llm_labeled_failure_ids.txt`**、**`llm_labeled_failures_report.csv`**。

---

## 目录与文件说明

### 脚本（`comment/` 根目录）

| 文件 | 作用 |
| --- | --- |
| `clean_com.py` | 合并 `rawdata`，一级清洗 → `cleaned_com.json` |
| `clean_data.py` | 二级清洗 → `bert_data/` 内 JSON/CSV/报告 |
| `meaningless_word.txt` | 低信息熵过滤用词表（`#` 行为注释） |
| `random_sample.py` | 构造 LLM 标注用子集 |
| `llm_annotate.py` | 本地 OpenAI 兼容 API 批量标注 |
| `recalc_llm_label_stats.py` | 修正标注表后重算统计与失败报告 |
| `split_dataset.py` | 划分 train/val/test，可选负样本过采样 |
| `train_roberta.py` | 微调与导出 |
| `evaluate_model.py` | 测试集评估、图表与 `eval_viz_payload.json` |
| `predict_all.py` | 全量推理（默认 `max_length=512`） |
| `summarize_predictions.py` | 预测结果分布与按笔记聚合 |

### `rawdata/`

按日抓取的 **`search_comments_YYYY-MM-DD.json`**。结构支持「根为数组」或「根为对象且列表在 `data` 字段」。

### `bert_data/`（清洗与训练数据）

| 文件 | 说明 |
| --- | --- |
| `final_cleaned_comments.json` | 清洗后全量：`note_id`、`nickname`、`content`、`original_content` 等 |
| `bert_train_ready.csv` | 两列 `note_id`, `txt`（无标签，供预测或合并标签） |
| `clean_data_report.txt` | 二级清洗统计 |
| `llm_sample_data.json` | 供 LLM 标注的抽样 |
| `llm_labeled_result.csv` | LLM 标注结果（含 `label`、`raw_response` 等） |
| `llm_labeled_result_stats.csv` | 标注分布统计 |
| `train.csv` / `val.csv` / `test.csv` | 划分后训练与评估集 |

### `saved_model/`（推理与评估加载目录）

由 `train_roberta.py` 将验证集上最优 checkpoint 导出至此。至少包含 **`config.json`**、分词器与权重（`model.safetensors` 或 `pytorch_model.bin`）。大文件可能未纳入 Git，需本地训练后生成。

### `checkpoint_temp/`（训练中间产物）

各 `checkpoint-*` 子目录、`trainer_state.json`、`run_summary.json`、`checkpoint_eval_summary.json` 等。开发态 **`judge_server.py`** 在未设置 `JUDGE_COMMENT_MODEL_DIR` 时会自动选用步号最大的 checkpoint 做在线情感推理。

### `results/`（离线评估产出）

| 文件 | 说明 |
| --- | --- |
| `evaluation_report.txt` | Accuracy、Macro-F1、按类 precision/recall/F1 |
| `01_confusion_matrix.png` ~ `08_metrics_radar.png` | 混淆矩阵、指标柱状图、置信度、ROC/PR、雷达图等 |
| `eval_viz_payload.json` | 与 PNG 同源的数值载荷，供 `visualization` 前端绘图 |
| `09_error_analysis.txt` | 误判分析摘要 |
| `misclassified_test.csv` | 误判样本明细 |

### `comment_results/`（全量预测与汇总）

| 文件 | 说明 |
| --- | --- |
| `predicted_comments.json` | 在清洗结构基础上增加 `sentiment_*` 字段 |
| `prediction_stats.txt`、`*_polarity.csv`、`*_class.csv`、`*_notes.csv` | 由 `summarize_predictions.py` 生成 |

推理过程中可能短暂存在 **`predicted_comments.predict_ckpt.json`**，成功后通常删除。

---

## 与可视化 / 在线研判的衔接

仓库根目录 **`visualization/`** 在开发模式下通过 Vite 同源挂载 `comment/` 与 `content/`（默认端口 **1306**），评论分析页主要读取：

- `comment/comment_results/prediction_stats_*.csv`
- `comment/results/evaluation_report.txt`、`eval_viz_payload.json`
- `comment/predictions` API（分页读 `predicted_comments.json`，见 `vite.config.ts`）
- `comment/training_history`（解析 `checkpoint_temp` 下 `trainer_state.json`）

在线「笔记 + 评论」研判由 **`visualization/judge_server.py`**（`npm run dev` 时自动拉起，默认 `127.0.0.1:18999`）提供：评论情感默认加载 `comment/checkpoint_temp` 最新 checkpoint 或 `JUDGE_COMMENT_MODEL_DIR` 指向的目录；与 `evaluate_model.py` 的类别 id→极性映射一致。

跨模块关联键为 **`note_id`**：前端将评论预测与 `content/bertopic_results_optimized/final_pro_topics.csv` 按 `note_id` 合并，展示笔记宏观主题与评论情感对照（见 `visualization/src/data/commentLiveData.ts`）。

---

## 依赖与环境

- **Python**：与仓库根目录 **`requirements_backup.txt`** / **`environment.yml`** 一致（`torch` 需单独按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 或 conda 安装）。
- **本模块常用包**：`transformers`、`datasets`、`pandas`、`scikit-learn`、`tqdm`、`matplotlib`、`seaborn`、`openai`。
- **LLM 标注**：`openai` 客户端 + 本地 **`BASE_URL`**（默认 `http://127.0.0.1:1234/v1`）与 **`MODEL_NAME`**；`llm_annotate.py` 中 `MAX_WORKERS` 当前为 1，可按接口稳定性调高。

安装示例（仓库根目录）：

```bash
conda env create -f environment.yml
conda activate studio
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

`train_roberta` 示例（换基座、缩短序列防 OOM）：

```bash
python train_roberta.py --model_name hfl/chinese-roberta-wwm-ext --max_length 256 --epochs 10
```

`predict_all` 可指定输入与模型目录：

```bash
python predict_all.py --input bert_data/final_cleaned_comments.json --model-dir saved_model --output-dir comment_results
```

开发研判服务（仓库根目录，需已训练模型与 content 侧 BERTopic）：

```bash
cd visualization && python judge_server.py
```

---

## 使用注意

1. **路径可移植**：各脚本输出目录多基于 `__file__` 解析；`llm_annotate.py` 写入同级的 `bert_data/`。
2. **标签与 CSV**：划分与评估脚本将 `label` 转为数值；Excel 编辑后若出现非法 `label`，`split_dataset` 会剔除并打印报告。
3. **长文本**：训练与推理默认 **`max_length=512`**，与旧版 128 配置不同，需与导出模型一致。
4. **全量 JSON 体积**：`predicted_comments.json` 行数大，前端通过分页 API 或聚合 CSV 加载，不宜整文件进浏览器。
5. **清洗与预测条数**：二级清洗后 `final_cleaned_comments.json` 条数若已更新，应重新执行 `predict_all.py` 与 `summarize_predictions.py`，再刷新前端缓存。

如需把 README 中的「当前指标」与某次固定实验对齐，请在复现实验后更新本节表格中的引用文件或数字。
