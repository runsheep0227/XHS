# 评论分析模块

基于RoBERTa的中文评论情感分析系统，专门针对小红书AIGC相关评论进行三分类情感分析（正向/中性/负向）。

## 🎯 项目概述

本项目构建了一个完整的中文评论情感分析流水线，包含数据清洗、大模型标注、模型微调、评估和批量预测等完整流程。通过结合大模型标注和RoBERTa微调，实现了高精度的三分类情感分析。

### 🏆 模型性能

- **准确率 (Accuracy)**: 78.60%
- **宏F1分数 (Macro-F1)**: 75.45%
- **支持情感类别**: 负向(-1)、中性(0)、正向(1)

### 📊 数据规模

- **原始评论数据**: 61,060 条
- **清洗后有效数据**: 55,940 条 (有效留存率: 91.61%)
- **大模型标注样本**: 5,000 条(训练集:验证集:测试集 = 8:1:1)

## 🏗️ 技术架构

### 核心模型
- **基础模型**: Chinese-RoBERTa-Base
- **分类器**: 线性分类层
- **最大序列长度**: 128 tokens
- **批处理大小**: 32
- **学习率**: 2e-5

### 大模型标注
- **使用模型**: Qwen3.5-9B (本地LM Studio部署)
- **标注维度**: 3层情感分类
- **Prompt设计**: 社会学专家视角，包含详细分类标准

## 🔄 数据处理流程

### 步骤1: 基础清洗 (`clean_com.py`)
**功能**: 原始数据预处理与字段提取
- 提取核心字段: note_id, nickname, content, ip_location
- 去除纯@用户内容
- 小红书表情符号处理（如[微笑R]→"微笑"）
- 内容去重
- 生成 `cleaned_com.json`

### 步骤2: 深度清洗 (`clean_data.py`)
**功能**: 文本质量过滤与标准化
- 特殊符号清理（保留中英文、数字、常用标点）
- 长度过滤（<2字符的短文本）
- 低信息熵内容过滤（基于meaningless_word.txt）
- 二次去重
- 生成 `final_cleaned_comments.json` 和 `bert_train_ready.csv`

### 步骤3: 数据抽样 (`random_sample.py`)
**功能**: 随机抽样用于大模型标注
- 从全量数据中随机抽取指定数量样本
- 保持数据分布的代表性
- 生成 `llm_sample_data.json`

### 步骤4: 大模型标注 (`llm_annotate.py`)
**功能**: 自动化情感标注
- 调用本地大模型API进行3分类标注
- 标注标准：
  - 1: 正向情感（强正向/弱正向）
  - 0: 中性情感（客观陈述）
  - -1: 负向情感（弱负向/强负向）
- 支持并发处理和重试机制
- 生成 `llm_labeled_result.csv`

### 步骤5: 数据集划分 (`split_dataset.py`)
**功能**: 训练集、验证集、测试集划分
- 仅保留有效标签（-1, 0, 1）
- 分层抽样：训练集80%、验证集10%、测试集10%
- 生成 `train.csv`, `val.csv`, `test.csv`

### 步骤6: 模型微调 (`train_roberta.py`)
**功能**: RoBERTa模型微调
- 加载Chinese-RoBERTa-Base预训练模型
- 添加分类头，支持3分类
- 使用AdamW优化器，线性学习率衰减
- 早停机制防止过拟合
- 保存最佳模型到 `saved_model/`

### 步骤7: 模型评估 (`evaluate_model.py`)
**功能**: 模型性能评估与可视化
- 在测试集上评估模型性能
- 生成混淆矩阵热力图
- 输出详细的分类报告
- 支持Precision、Recall、F1-score等指标

### 步骤8: 批量预测 (`predict_all.py`)
**功能**: 全量数据情感预测
- 使用微调后的模型对55,940条评论进行批量预测
- 生成包含情感标签的最终结果
- 输出 `final_sentiment_com.json`

## 📁 文件结构说明

```
comment/
│
├── 📁 rawdata/                    # 原始评论数据
│   ├── search_comments_*.json     # 按日期分类的原始爬取数据
│   └── ...
│
├── 📁 bert_data/                  # 核心数据处理区
│   ├── final_cleaned_comments.json # 清洗后的完整数据（55,940条）
│   ├── bert_train_ready.csv       # BERT训练格式数据
│   ├── llm_sample_data.json       # 大模型标注样本（5,000条）
│   ├── llm_labeled_result.csv     # 大模型标注结果
│   ├── train.csv                  # 训练集（约2,400条）
│   ├── val.csv                    # 验证集（约300条）
│   ├── test.csv                   # 测试集（约300条）
│   └── clean_data_report.txt      # 清洗报告
│
├── 📁 saved_model/                # 微调后的模型文件
│   ├── model.safetensors          # 模型权重
│   ├── config.json                # 模型配置
│   ├── tokenizer.json             # 分词器
│   └── label_mapping.json         # 标签映射
│
├── 📁 results/                    # 最终结果
│   ├── final_sentiment_com.json   # 全量预测结果
│   ├── confusion_matrix.png       # 混淆矩阵图
│   └── evaluation_report.txt      # 模型评估报告
│
├── 📁 checkpoint_temp/            # 训练检查点（临时）
│
├── 🐍 clean_com.py                # 基础数据清洗
├── 🐍 clean_data.py               # 深度数据清洗
├── 🐍 random_sample.py            # 数据抽样
├── 🐍 llm_annotate.py             # 大模型标注
├── 🐍 split_dataset.py            # 数据集划分
├── 🐍 train_roberta.py            # 模型微调
├── 🐍 evaluate_model.py           # 模型评估
├── 🐍 predict_all.py              # 批量预测
│
├── 📄 meaningless_word.txt        # 无意义词词典
└── 📄 README.md                   # 项目说明
```

## 🚀 快速开始

### 环境准备
```bash
# 安装依赖
pip install transformers datasets torch pandas scikit-learn matplotlib seaborn tqdm openai

# 启动LM Studio本地服务（用于大模型标注）
# 配置BASE_URL = "http://127.0.0.1:1234/v1"
```

### 完整流程执行
```bash
# 1. 基础清洗
python clean_com.py

# 2. 深度清洗
python clean_data.py

# 3. 数据抽样
python random_sample.py

# 4. 大模型标注（需要LM Studio运行）
python llm_annotate.py

# 5. 数据集划分
python split_dataset.py

# 6. 模型微调（需要GPU）
python train_roberta.py

# 7. 模型评估
python evaluate_model.py

# 8. 批量预测
python predict_all.py
```

## 📈 情感分析结果统计

基于55,940条小红书AIGC评论的情感分析结果：

- **正向情感**: 约40% - 包含兴奋、期待、赞同、满意等情绪
- **中性情感**: 约45% - 客观描述、技术讨论、中性观点
- **负向情感**: 约15% - 包含担忧、焦虑、不满、抵触等情绪

## 🔧 配置参数

### 大模型标注配置
- **模型**: qwen/qwen3.5-9b
- **API地址**: http://127.0.0.1:1234/v1
- **并发数**: 8
- **重试次数**: 3次

### 模型训练配置
- **基础模型**: hfl/chinese-roberta-wwm-ext
- **批次大小**: 32
- **学习率**: 2e-5
- **训练轮数**: 10
- **早停耐心**: 3

## 📚 使用建议

1. **数据质量控制**: 建议根据实际数据特点调整meaningless_word.txt
2. **标注质量**: 大模型标注后建议人工抽检10%-20%的样本
3. **模型调优**: 可根据数据特点调整学习率、批次大小等超参数
4. **结果分析**: 建议结合主题建模结果进行交叉分析

## 🔍 扩展应用

- **实时监控**: 可用于实时监控用户对AIGC产品的情感变化
- **产品优化**: 识别用户痛点，指导产品功能优化
- **内容推荐**: 基于情感倾向优化内容推荐策略
- **市场分析**: 分析不同用户群体对AIGC技术的接受度