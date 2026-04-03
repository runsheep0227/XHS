# 小红书笔记内容分析模块

BERTopic详细地址(https://maartengr.github.io/BERTopic/index.html#citation)

相关论文:BERTopic、RoBERT-base

## 项目结构

```
content/
│
├── 📁 原始数据与缓存
│   ├── rawdata/                          # 📊 原始数据文件夹
│   │   ├── search_contents_2026-01-29.json
│   │   ├── search_contents_2026-01-30.json
│   │   └── ...                          # 按日期分类的爬取数据
│   ├── hf_cache/                         # 🤗 Hugging Face模型缓存
│   │   └── models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/
│   └── __pycache__/                      # 🗑️ Python编译缓存（可忽略）
│
├── 📁 数据处理流程
│   ├── 📝 步骤1: 字段提取
│   │   ├── clean_desc.py                 # 数据字段提取脚本
│   │   └── cleaned_desc.json             # 提取结果
│   ├── 🧹 步骤2: 数据清洗
│   │   ├── clean_data.py                 # 数据清洗脚本
│   │   └── bertopic_ready_data/          # 清洗后数据
│   │       ├── bertopic_cleaned_data.json
│   │       ├── bertopic_docs.txt
│   │       ├── bertopic_train_data.csv   # 训练主输入文件
│   │       ├── BERTopic数据清洗报告.txt
│   │       └── BERTopic完整清洗结果.xlsx
│   ├── 🧠 步骤3: 主题建模
│   │   ├── bertopic_train.py             # BERTopic训练脚本
│   │   └── bertopic_result/              # 建模结果
│   │       ├── bertopic_model/
│   │       │   ├── config.json           # 模型配置
│   │       │   ├── ctfidf_config.json
│   │       │   ├── ctfidf.safetensors
│   │       │   ├── topic_embeddings.safetensors
│   │       │   └── topics.json
│   │       ├── bertopic_topic_info.csv   # 主题宏观信息
│   │       └── document_topics.csv       # 文档主题分配
│   └── 🔮 步骤4: 主题预测
│       └── predict_topic.py              # 新数据预测脚本
│
├── 📁 配置与资源
│   ├── aigc_xhs_stopwords.txt            # 🚫 停用词表（小红书/AIGC语境）
│   └── aigc_keyword.txt                  # 🔑 关键词表
│
├── 📁 日志与监控
│   └── logs/                             # 📋 执行日志
│       ├── bertopic_train.log
│       ├── clean_quality_report.json
│       ├── data_cleaning_report.txt
│       └── training_report.txt
│
└── 📄 README.md                          # 📖 项目说明文档
```
## 数据处理流程

本项目采用4步流水线式数据处理流程：

### 步骤1: 数据字段提取 (clean_desc.py)
- **输入**: `rawdata/`文件夹中的原始JSON文件
- **功能**: 从原始数据中提取核心字段(note_id, title, desc)，进行数据校验和错误报告
- **输出**: `cleaned_desc.json`

### 步骤2: 数据清洗与预处理 (clean_data.py)
- **输入**: `cleaned_desc.json`
- **功能**: 
  - 文本清洗（去除标点、数字、表情符号等）
  - 中文分词处理
  - 基于停用词表过滤
  - 质量过滤（最小文本长度、最小词数要求）
- **输出**: `bertopic_ready_data/bertopic_train_data.csv`

### 步骤3: BERTopic主题建模 (bertopic_train.py)
- **输入**: `bertopic_ready_data/bertopic_train_data.csv`
- **功能**:
  - 使用Sentence Transformer生成文本嵌入
  - UMAP降维处理
  - K-means聚类生成主题
  - c-TF-IDF提取主题关键词
- **输出**: 
  - `bertopic_results/bertopic_model/` - 训练好的模型文件
  - `bertopic_results/topic_info.csv` - 主题信息
  - `bertopic_results/document_topic_assignment.csv` - 文档主题分配

### 步骤4: 主题预测 (predict_topic.py)
- **输入**: 新的小红书笔记数据
- **功能**: 使用训练好的模型对新数据进行主题预测
- **输出**: 新数据的主题分类结果

## 主题建模结果

基于小红书AIGC相关笔记数据，模型识别出5个主要主题：

1. **AI内容创作** (2,815篇) - 小说写作、换脸技术、深度伪造等
2. **AI工具测评** (2,896篇) - 各类AI工具的使用教程和测评
3. **AI学习教程** (3,036篇) - 产品经理、面试技巧、学习方法等
4. **AI生活融合** (3,440篇) - 机器学习、数据分析、效率工具等
5. **AI社会反思** (2,365篇) - 壁纸、写真、审美分享等

## 文件说明

- **clean_desc.py**: 数据字段提取脚本，负责从原始JSON中抽取核心字段
- **clean_data.py**: 数据清洗脚本，进行文本预处理和分词
- **bertopic_train.py**: BERTopic主题建模训练脚本
- **predict_topic.py**: 主题预测脚本，用于新数据分类
- **aigc_xhs_stopwords.txt**: 针对小红书AIGC内容的停用词表
- **aigc_keyword.txt**: 关键词表，用于文本处理参考
└── README.md                          # 项目说明

需要的安装包:

⚠️在RTX 50系列上运行Pytorch时会遇到CUDA兼容性问题。
核心问题在于PyTorch稳定版本的预编译二进制文件不支持sm_120计算能力，而RTX 50系列采用了较新的架构，需要更新的CUDA版本以及对应的PyTorch构建版本才能正常工作。
使用PyTorch Nightly构建新版本，但可能存在一定的不稳定性。
1.清理现有环境 pip uninstall troch trochvision torchaudio -y pip cache purge
2.安装支持CUDA 12.8的Nightly构建版本
pip3 install --pre torch torchvision torchaudio --index-
通过这一命令可以安装支持CUDA 12.8的PyTorch版本，该版本能够支持RTX 50系列的sm_120架构。