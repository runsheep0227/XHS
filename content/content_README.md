# 小红书笔记内容分析模块（BERTopic）

本目录对小红书 **AIGC 相关笔记**（标题 + 正文）做清洗、分词与 **BERTopic** 主题建模：先得到细粒度 **微观主题**（HDBSCAN 聚类），再映射到 5 个 **宏观主题**（AI 内容创作、应用与测评、学习教程、赋能工作生活、社会反思）。

**BERTopic 文档**：[BERTopic 官方文档](https://maartengr.github.io/BERTopic/index.html)。实现依赖 **Sentence-Transformers**（`BAAI/bge-large-zh-v1.5`）、**UMAP**、**HDBSCAN** 与 **c-TF-IDF**；以当前 `bertopic_train.py` 为准（非早期 K-Means 实验）。

---

## 推荐执行顺序

| 步骤 | 脚本 | 输入 | 主要输出 |
| --- | --- | --- | --- |
| 1 | `clean_desc.py` | `rawdata/search_contents_*.json` | `cleaned_desc.json`、`logs/extract_error_report.json` |
| 2 | `clean_data.py` | `cleaned_desc.json` | `bertopic_ready_data/bertopic_train_data.csv`、`bertopic_cleaned_data.json`；质量报告见 `logs/` |
| 3 | `bertopic_train.py` | `bertopic_train_data.csv` | `bertopic_results_optimized/` 下模型、`final_pro_topics.csv`、`pro_mapping_report.txt` |
| 4（可选） | `bertopic_visualize.py` | 已训练模型 + `final_pro_topics.csv` + `bertopic_train_data.csv` | `bertopic_visualizations/` 下 01–11 号图表 |

各脚本 **`WORK_DIR = Path(__file__).resolve().parent`**，均以 **`content/`** 为根目录读写。

---

## 当前仓库中的数据与指标（仅供参考）

| 阶段 | 来源 | 当前约数 |
| --- | --- | --- |
| 一级清洗 | `clean_desc.py` → `cleaned_desc.json` | 输入约 **19,097** 条（去重后，见 `logs/extract_error_report.json`） |
| 二级清洗 | `logs/clean_quality_report.json` | 有效 **18,255** 条（有效率 **95.6%**）；门控：全文 ≥20 字、分词后 ≥10 词 |
| 主题建模 | `pro_mapping_report.txt` / `final_pro_topics.csv` | 训练样本 **18,255** 条；有效分类 **17,583** 条（**96.3%**）；微观主题 **24** 个 |

宏观分布（`pro_mapping_report.txt`，随重训变化）示例：AI内容创作约 5,314 条、AI应用与测评约 3,380 条、AI学习教程约 3,058 条、AI赋能工作生活约 3,600 条、AI社会反思见报告余下章节。

随机种子：**`RANDOM_STATE = 13`**（`bertopic_train.py`）。

---

## 目录结构

```
content/
├── rawdata/                          # search_contents_YYYY-MM-DD.json（根为 list）
├── cleaned_desc.json                 # clean_desc.py 输出（clean_data.py 的输入）
├── aigc_xhs_stopwords.txt            # 停用词（# 开头为注释）
├── aigc_keyword.txt                  # 领域词，注入 jieba
├── bertopic_ready_data/
│   ├── bertopic_train_data.csv       # note_id, cleaned_full_text, cleaned_seg_text
│   └── bertopic_cleaned_data.json    # 通过门控的全字段 JSON 数组
├── bertopic_results_optimized/       # bertopic_train.py 默认输出
│   ├── saved_model/                  # topic_model.save(..., serialization="safetensors")
│   ├── final_pro_topics.csv          # 每条笔记：微观/宏观主题、置信度、is_noise 等
│   ├── pro_mapping_report.txt        # 宏观—微观映射可读报告
│   └── topic_distribution_stats.csv  # 可选；当前训练脚本不自动生成，研判服务会尝试读取
├── bertopic_visualizations/          # bertopic_visualize.py 输出（运行后生成）
├── logs/
│   ├── clean_desc.log
│   ├── extract_error_report.json
│   ├── data_cleaning_report.txt
│   ├── clean_quality_report.json
│   ├── bertopic_optimized.log
│   └── training_report.txt           # 历史摘要，与现流程不一致时以代码为准
├── clean_desc.py
├── clean_data.py
├── bertopic_train.py                 # Guided BERTopic + 动态 HDBSCAN + 离群回收
└── bertopic_visualize.py             # 离线读模型与 CSV 出图
```

> **说明**：仓库中曾存在训练变体脚本 `predict.py`，当前已移除；请勿与 `bertopic_train.py` 混用旧结果。`topic_distribution_stats.csv` 若存在，多为历史导出；`judge_server.py` 用于微观 id→宏观名查表，缺失时可从 `final_pro_topics.csv` 自行聚合。

---

## 各脚本说明

### 1. `clean_desc.py`

- 遍历 `rawdata/*.json`，根节点为 **list**，每条含 `note_id`、`title`、`desc`（str）。
- 保留 **title 或 desc 非空** 的笔记；按 `note_id` **去重**（保留首次）。
- 输出：`cleaned_desc.json`；异常与统计 → `logs/extract_error_report.json`。

### 2. `clean_data.py`

- 读 `cleaned_desc.json`，拼接 `title` 与 `desc` 后清洗：去 `[]` 表情占位、`#话题#`、@、链接；可选保留英文（`KEEP_ENGLISH=True`）；jieba 分词并滤停用词与单字。
- 有效条件：`cleaned_full_text` ≥ **20** 字符，分词词数 ≥ **10**（`MIN_FULL_TEXT_LEN`、`MIN_SEG_WORD_COUNT`）。
- 输出：`bertopic_ready_data/bertopic_train_data.csv`、`bertopic_cleaned_data.json`；`logs/clean_quality_report.json`、`logs/data_cleaning_report.txt`。
- **不**生成旧流程中的 `bertopic_docs.txt`、`.xlsx` 等。

### 3. `bertopic_train.py`（主线）

- 读 `bertopic_train_data.csv`；过滤空行与全文 &lt;5 字符。
- **嵌入**：`BAAI/bge-large-zh-v1.5`，`max_seq_length=512`，CUDA/CPU 自动选择。
- **聚类**：UMAP（`n_neighbors=30`, `n_components=15`）+ **HDBSCAN**（`min_cluster_size` 随样本量动态计算，`RANDOM_STATE=13`）；**Guided BERTopic**（`seed_topic_list` 来自 `MACRO_ANCHORS` 关键词）。
- 训练后：`reduce_topics`（上限约 30）、`reduce_outliers`（embedding 策略）、低置信微观主题合并或标噪声。
- **宏观映射**（`compute_mappings`）：微观主题词表加权（40%）+ 宏观 definition 语义相似度（60%）。
- 写入：`bertopic_results_optimized/saved_model/`、`final_pro_topics.csv`、`pro_mapping_report.txt`。
- 环境：`HF_ENDPOINT=https://hf-mirror.com`（便于国内拉模型）。

`final_pro_topics.csv` 主要列：`note_id`、`content`、`segmented_text`、`micro_topic_id`、`micro_topic_keywords`、`macro_topic_name`、`mapping_confidence`、`is_noise`。

### 4. `bertopic_visualize.py`

- 输入：`bertopic_results_optimized/saved_model`、`final_pro_topics.csv`、`bertopic_train_data.csv`。
- 输出目录 **`bertopic_visualizations/`**（`NUMBA_DISABLE_JIT=1` 默认开启，避免 Windows 下 numba 卡住）：

| 文件 | 内容 |
| --- | --- |
| `01_macro_distribution.png` | 宏观主题分布 |
| `02_micro_topic_sizes.png` | 微观主题规模 |
| `03_umap_projection.png` | UMAP 散点 |
| `04_confidence_distribution.png` | 映射置信度 |
| `05_keyword_comparison.png` | 关键词对比 |
| `06_topic_distance_heatmap.png` | 主题距离热力 |
| `07_text_length.png` | 文本长度分布 |
| `08_dashboard.png` | 总览仪表盘 |
| `09_macro_micro_breakdown.png` | 宏观—微观拆解 |
| `10_wordclouds.png` | 词云 |
| `11_topic_sankey.html` / `.png` | 桑基图（需 plotly；PNG 需 kaleido） |

Windows 默认绑定 **`C:\Windows\Fonts\msyh.ttc`**；Linux 需自行改脚本内字体路径或依赖 `DejaVu Sans`。

配色：`COLOR_STYLE` 可选 `rainbow`（默认）或 `scientific`。

### 对新笔记批量打主题

本目录**未**提供单独的「仅 transform」小脚本。对新数据应：

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

emb = SentenceTransformer("BAAI/bge-large-zh-v1.5")
topic_model = BERTopic.load("bertopic_results_optimized/saved_model", embedding_model=emb)
# 需与训练一致：先对 cleaned_full_text encode，再 transform(cleaned_seg_text, embeddings=...)
```

在线研判见 **`visualization/judge_server.py`**（笔记正文走 BERTopic，评论走 comment 情感模型；`npm run dev` 时由 Vite 自动启动）。

---

## 宏观主题（与 `MACRO_ANCHORS` 一致）

1. **AI内容创作** — 绘画/视频/音乐/小说等生成与作品展示。  
2. **AI应用与测评** — 工具、APP、大模型、智能体等测评与对比。  
3. **AI学习教程** — 教程、提示词、入门与进阶学习。  
4. **AI赋能工作生活** — 办公提效、副业、论文、代码等场景。  
5. **AI社会反思** — 版权、失业焦虑、伦理、深度伪造、隐私等。

统计与可视化请以 **`final_pro_topics.csv`**、**`pro_mapping_report.txt`** 为准；`logs/training_report.txt` 中旧版「MiniLM + K-Means」描述可能已过时。

---

## 与可视化 / 研判的衔接

**`visualization/`** 开发态从 `content/` 读取：

- `bertopic_results_optimized/final_pro_topics.csv`（主题分析主表）
- `rawdata/search_contents_*.json`（按 `note_id` 合并互动量、标签、`note_url` 等）
- 可选 `bertopic_results_optimized/topic_distribution_stats.csv`（模型质量图，见 `ModelQualityCharts.tsx`）

**`visualization/judge_server.py`** 加载 `content/bertopic_results_optimized/saved_model`，环境变量：

- `JUDGE_BERTOPIC_DIR` — 模型目录  
- `JUDGE_EMBEDDING_MODEL` — 默认 `BAAI/bge-large-zh-v1.5`  
- `JUDGE_TOPIC_BACKEND` — 设为 `lexical` 时可走词法后备（开发调试）

与评论模块的关联键为 **`note_id`**（见 `visualization/src/data/topicData.ts`、`commentLiveData.ts`）。

---

## Python 依赖

与仓库根目录 **`requirements_backup.txt`** / **`environment.yml`** 一致。本模块除通用科学栈外主要用到：`jieba`、`bertopic`、`sentence-transformers`、`umap-learn`、`hdbscan`；桑基图与词云为可选（`plotly`、`kaleido`、`wordcloud`，见备份清单注释）。

```bash
# 仓库根目录
conda env create -f environment.yml
conda activate studio
```

---

## GPU 与 PyTorch（含 RTX 50 系列）

在新架构 GPU 上，稳定版 PyTorch 可能缺少对应 CUDA 架构（如 sm_120）。可选：

1. `pip uninstall torch torchvision torchaudio -y` 后 `pip cache purge`
2. 按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装 **Nightly** 或已支持新架构的 CUDA 构建，例如：

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

生产环境请锁定经本地验证的版本号。

---

## 快速执行命令（在 `content/` 下）

```bash
python clean_desc.py
python clean_data.py
python bertopic_train.py
python bertopic_visualize.py
```

全流程前请确认 **`cleaned_desc.json`** 位于 `content/` 根目录（由 `clean_desc.py` 生成）；`clean_data.py` 不会自动从其他子目录寻找该文件。

开发可视化与研判（仓库根目录）：

```bash
cd ../visualization
npm run dev

# 另开终端
cd visualization && python judge_server.py
```

---

## 使用注意

1. **重训覆盖**：再次运行 `bertopic_train.py` 会覆盖 `bertopic_results_optimized/`；前端与 `judge_server` 需重启或重新加载。
2. **分词列**：聚类 `fit_transform` 使用 **`cleaned_seg_text`**（空格分词），嵌入使用 **`cleaned_full_text`**，二者勿混用。
3. **噪声行**：`is_noise=True` 或 `micro_topic_id=-1` 的笔记在统计宏观占比时应单独处理。
4. **镜像与缓存**：首次运行会下载 `bge-large-zh-v1.5`；可配置 `HF_HOME` 或使用 `hf_cache/`（若存在）。
