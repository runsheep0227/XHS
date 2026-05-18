# Studio — 小红书 AIGC 内容分析

> **最终研究代码** — 本仓库为课题结题后的完整可复现流水线与可视化交付物。

端到端研究流水线：笔记主题建模（BERTopic）、评论三分类情感（MacBERT 微调）、统计可视化与在线研判。整体框架见 [框架说明.md](框架说明.md)。

## 仓库结构

| 目录 | 说明 |
| --- | --- |
| `content/` | 笔记清洗、BERTopic 训练与可视化 |
| `comment/` | 评论清洗、LLM 伪标签、RoBERTa/MacBERT 训练与全量预测 |
| `visualization/` | React 仪表盘；开发时通过 Vite 提供 `content/`、`comment/` 数据 API，并可选拉起 `judge_server.py` |
| `environment.yml` | Conda 环境（推荐） |
| `requirements_backup.txt` | 可移植 pip 清单（不含 torch，避免 CUDA 变体冲突） |

各子模块详细步骤见 `content/content_README.md`、`comment/comment_README.md`、`visualization/README.md`。

## 环境安装

### Python（推荐 Conda）

在仓库根目录：

```bash
conda env create -f environment.yml
conda activate studio
```

仅使用 pip 时：先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装 `torch`，再执行 `pip install -r requirements_backup.txt`。

可选依赖（桑基图、词云）：`plotly`、`kaleido`、`wordcloud` 已写在 `requirements_backup.txt` 可选段，完整安装时一并装上即可。

### 前端

```bash
cd visualization
pnpm install
pnpm run dev
```

浏览器打开 **http://localhost:1306**。研判服务默认 **127.0.0.1:18999**（`judge_server.py`，可由 `vite.config.ts` 在 `npm run dev` 时自动启动）。

指定 Python 解释器（Windows 示例）：

```powershell
$env:PYTHON_PATH="D:\path\to\envs\studio\python.exe"
cd visualization
pnpm run dev
```

## 大文件与本地生成物

以下路径在 `.gitignore` 中，克隆后需自行训练或下载：

- `comment/checkpoint_temp/`、`comment/saved_model/` — 情感模型权重
- `content/hf_cache/` — 句向量模型缓存（如 `BAAI/bge-large-zh-v1.5`）

仓库已包含 BERTopic 导出结果 `content/bertopic_results_optimized/` 及多数中间数据，可直接跑可视化；重训会覆盖对应目录。

## 快速命令索引

```bash
# content/
python clean_desc.py && python clean_data.py && python bertopic_train.py

# comment/
python clean_com.py && python clean_data.py
# … 见 comment/comment_README.md

# 前端
cd visualization && pnpm run dev
```

## 许可与数据

研究用途数据集与模型权重请遵守各上游许可（HuggingFace、小红书抓取合规等）。勿将 `.env` 或 API 密钥提交到版本库。
