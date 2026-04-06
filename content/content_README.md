# 小红书笔记内容分析模块（BERTopic）

本目录对小红书 **AIGC 相关笔记**（标题 + 正文）做清洗、分词与 **BERTopic** 主题建模：先得到细粒度 **微观主题**（HDBSCAN 聚类），再映射到 5 个 **宏观主题**（AI 内容创作、应用与测评、学习教程、赋能工作生活、社会反思）。

**BERTopic 文档与引用**：[BERTopic 官方文档](https://maartengr.github.io/BERTopic/index.html)。实现上主要依赖 **Sentence-Transformers** 句向量、**UMAP** 降维、**HDBSCAN** 密度聚类与 **c-TF-IDF** 主题词；与早期仅用 K-Means 的实验不同，请以当前脚本为准。

---

## 推荐执行顺序

| 步骤 | 脚本 | 输入 | 主要输出 |
| --- | --- | --- | --- |
| 1 | `clean_desc.py` | `rawdata/search_contents_*.json` | `cleaned_desc.json`、`logs/extract_error_report.json` |
| 2 | `clean_data.py` | `cleaned_desc.json` | `bertopic_ready_data/bertopic_train_data.csv`、`bertopic_cleaned_data.json`；质量报告见 `logs/` |
| 3 | `bertopic_train.py` | `bertopic_train_data.csv` | `bertopic_results_optimized/` 下模型与表格、映射报告 |
| 4（可选） | `bertopic_visualize.py` | 同上目录下已训练的模型与 `final_pro_topics.csv` | `bertopic_visualizations/` 下多张图表 |

---

## 目录结构

```
content/
├── rawdata/                          # 按日 JSON：search_contents_YYYY-MM-DD.json（根节点为数组）
├── cleaned_desc.json                 # clean_desc.py：抽取 note_id / title / desc 并去重
├── aigc_xhs_stopwords.txt            # 停用词（# 开头行为注释）
├── aigc_keyword.txt                  # 领域词，注入 jieba
├── bertopic_ready_data/
│   ├── bertopic_train_data.csv       # BERTopic 主输入：note_id, cleaned_full_text, cleaned_seg_text
│   └── bertopic_cleaned_data.json    # 有效样本全字段（JSON 数组）
├── bertopic_results_optimized/       # bertopic_train.py（及 predict.py）默认输出根目录
│   ├── saved_model/                  # topic_model.save(...)：config、topics.json、ctfidf 等
│   ├── final_pro_topics.csv          # 每条笔记：微观主题、宏观名、置信度、是否噪声等
│   ├── pro_mapping_report.txt        # 人工可读映射与主题摘要
│   ├── topic_distribution_stats.csv  # 仅当运行 predict.py 时生成（见下文）
│   └── document_classification_details.csv  # 当前仓库 Python 未写入；若存在多为历史/手工导出
├── bertopic_visualizations/          # bertopic_visualize.py 输出（运行后生成）
├── hf_cache/                         # Hugging Face 缓存（如曾下载 sentence-transformers 模型）
├── logs/
│   ├── clean_desc.log
│   ├── extract_error_report.json
│   ├── data_cleaning_report.txt
│   ├── clean_quality_report.json
│   ├── bertopic_optimized.log        # bertopic_train / predict 共用日志文件名
│   ├── training_report.txt           # 历史训练摘要（内容可能对应旧流程，与现脚本不一致时以代码为准）
│   └── …
├── clean_desc.py
├── clean_data.py
├── bertopic_train.py                 # 【主训练】Guided BERTopic + 动态 HDBSCAN + 离群点回收与合并
├── predict.py                        # 【训练变体】另一套 UMAP/HDBSCAN 超参；输出目录与主脚本相同
└── bertopic_visualize.py             # 离线读模型与 CSV，出图
```

---

## 各脚本说明

### 1. `clean_desc.py`

- 遍历 `rawdata/*.json`，要求根为 **list**，每条含 `note_id`、`title`、`desc`（类型为 str）。
- 校验后保留 **title 或 desc 非空** 的笔记；按 `note_id` **去重**（保留首次出现）。
- 输出：`cleaned_desc.json`；异常与统计写入 `logs/extract_error_report.json`。
- **路径**：`WORK_DIR = Path(__file__).resolve().parent`，始终以本文件所在目录（`content/`）为项目根。

### 2. `clean_data.py`

- 读 `cleaned_desc.json`，将 `title` 与 `desc` 拼接后清洗：
  - 去 `[]` 表情占位、`#话题#`、@、链接；
  - 可选保留英文（`KEEP_ENGLISH`）；
  - 去除数字与非法控制字符；
  - **jieba** 分词，过滤停用词与单字中文；
- 有效条件：`cleaned_full_text` 长度 ≥ **20**（字符），分词后词数 ≥ **10**（`MIN_FULL_TEXT_LEN`、`MIN_SEG_WORD_COUNT`）。
- 输出：
  - `bertopic_ready_data/bertopic_train_data.csv`（三列，供训练）；
  - `bertopic_ready_data/bertopic_cleaned_data.json`；
  - `logs/clean_quality_report.json`、`logs/data_cleaning_report.txt`。
- **不会**生成旧 README 中的 `bertopic_docs.txt`、`.xlsx` 或「BERTopic数据清洗报告」等文件。

### 3. `bertopic_train.py`（当前主线）

- 读 `bertopic_train_data.csv` 的 `cleaned_full_text` / `cleaned_seg_text`；对过短文本做额外过滤（少于 5 个字符）。
- **嵌入模型**：`BAAI/bge-large-zh-v1.5`（`max_seq_length=512`），设备自动选 CUDA/CPU。
- **聚类**：UMAP + **HDBSCAN**（`min_cluster_size` 随数据量动态调整），**Guided BERTopic**（`seed_topic_list` 来自 5 类宏观锚点关键词）。
- 主题数若超过上限会 **reduce_topics**；对离群点做 **reduce_outliers**（embedding 策略）；低置信度微观主题会合并或标为噪声。
- 宏观映射：用宏观 **definition** 文本与微观主题词算嵌入相似度，并结合关键词加权（见 `compute_mappings`）。
- 写入：
  - `bertopic_results_optimized/saved_model/`（`serialization="safetensors"`）；
  - `final_pro_topics.csv`；
  - `pro_mapping_report.txt`。
- 环境变量：`HF_ENDPOINT` 默认设为 `https://hf-mirror.com` 以便国内拉模型。

### 4. `predict.py`（训练变体，非「仅推理」）

- 与 `bertopic_train.py` 类似，同样是 **完整训练 + 保存**，默认 **同一输出目录** `bertopic_results_optimized/`。
- UMAP/HDBSCAN 等超参与主脚本不同（例如更小的 `min_cluster_size`），并额外写出 **`topic_distribution_stats.csv`**。
- **注意**：与 `bertopic_train.py` **不要交替运行后混用结果**，除非你有意覆盖；名称中的 predict 表示「管线脚本」，不是加载旧模型只对「新笔记」打分的 API。

若需要对**全新笔记**批量打主题，应在业务代码中 `BERTopic.load(saved_model)` 后调用 `transform` / `transform` 系列 API（本仓库未单独提供该小脚本）。

### 5. `bertopic_visualize.py`

- 读取 `bertopic_results_optimized/saved_model`、`final_pro_topics.csv` 与 `bertopic_train_data.csv`。
- 将图表写入 **`bertopic_visualizations/`**（宏观分布、热力、降维散点等，具体以脚本内函数为准）。
- Windows 下会尝试绑定中文字体（如 `msyh.ttc`）；Linux 可依赖脚本内备选路径。

---

## 宏观主题（与代码中 `MACRO_ANCHORS` 一致）

1. **AI内容创作** — 绘画/视频/音乐/小说等生成与作品展示。  
2. **AI应用与测评** — 工具、APP、大模型、智能体等测评与对比。  
3. **AI学习教程** — 教程、提示词、入门与进阶学习。  
4. **AI赋能工作生活** — 办公提效、副业、论文、代码等场景。  
5. **AI社会反思** — 版权、失业焦虑、伦理、深度伪造、隐私等。

**分布数字**（如各主题多少条）随数据与随机种子变化；旧版 `logs/training_report.txt` 中「MiniLM + K-Means 五类」的描述与 **当前** `bertopic_train.py`（bge-large + HDBSCAN）**不一定一致**，可视化与统计请以 **`final_pro_topics.csv`** 与 **`pro_mapping_report.txt`** 为准。

---

## Python 依赖（示例）

```text
pandas numpy torch
jieba
bertopic
sentence-transformers
umap-learn
hdbscan
scikit-learn
matplotlib seaborn
```

---

## GPU 与 PyTorch（含 RTX 50 系列说明）

在 **RTX 50 系列** 等新架构上，稳定版 PyTorch  wheel 可能尚未包含对应 **CUDA 架构（如 sm_120）**，会出现 CUDA 不可用或运行报错。可选方案：

1. 卸载旧包并清缓存：  
   `pip uninstall torch torchvision torchaudio -y`  
   `pip cache purge`
2. 按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择 **Nightly** 或已声明支持新架构的 CUDA 版本安装，例如（示例，请以官网当前命令为准）：

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Nightly 可能存在不稳定因素，生产环境请锁定通过验证的版本号。

---

## 工作目录

各脚本中 **`WORK_DIR = Path(__file__).resolve().parent`**，即始终以 **`content/`** 包目录为根，数据与输出会相对该目录读写，便于克隆到其他路径或系统。`bertopic_visualize.py` 头部的 Windows 字体路径仍为系统路径，与其他 OS 无关。
