# 评论分析模块

## 项目框架

STUDIO/
└── comment/                          # 评论分析模块工作目录
    │
    ├── 📁 rawdata/                   # 原始数据
    │   ├── search_comments_2026-02-25.json
    │   └── ... (其他原始爬取数据)
    │
    ├── 📁 bert_data/                 # 核心数据区:存放清洗后、供大模型及RoBERTa使用的数据
    │   ├── clean_data_report.txt     # [数据清洗] clean_data.py 生成的过滤报告
    │   ├── final_cleaned_comments.json # [数据清洗] 历经两次清洗后的最终全量有效数据(10w+)
    │   ├── llm_sample_data.json      # 随机抽样给大模型标注的原始数据 (如3000条)
    │   ├── llm_labeled_data.csv      # 大模型标注完成，且经过抽检修正的最终标注数据
    │   ├── train.csv                 # 拆分后的微调训练集 (占80%，约2400条)
    │   ├── val.csv                   # 拆分后的验证集 (占10%，约300条，供训练时监控)
    │   └── test.csv                  # 拆分后的测试集 (占10%，约300条，供最终评估)
    │
    ├── 📁 saved_model/               # 训练过程中自动生成，存放微调后最好的模型权重(权重文件较大)
    │   ├── config.json
    │   ├── pytorch_model.bin          # 或者 model.safetensors
    │   └── ... (tokenizer等文件)
    │
    ├── 📁 results/                    # 存放最终的输出结果与图表
    │   ├── evaluation_report.txt      # 测试集上的准确率(Accuracy)、宏F1分数(Macro-F1)等指标报告
    │   ├── confusion_matrix.png       # 混淆矩阵热力图 (可直接放在论文中)
    │   └── final_sentiment_com.json   # 包含了原始字段及 "emotion_label" 的10w+最终预测数据
    │
    ├── 📄 meaningless_word.txt       # 无意义词/停用词词典
    ├── 📄 requirements.txt           #  记录该项目的环境依赖(如 transformers, torch, pandas)
    │
    ├── 🐍 clean_com.py               # [基础清洗] 提字段、去纯@、去重、去[微笑R]表情
    ├── 🐍 clean_data.py              # [深度清洗] 结合 meaningless_word.txt 去除低信息熵和过短文本
    ├── 🐍 random_sample.py           # [抽样脚本] 从全量数据中抽取指定数量用于标注
    ├── 🐍 llm_annotate.py            # [大模型标注] 调用大模型API(Prompt设计)自动化打标签
    ├── 🐍 split_dataset.py           # [数据划分] 将标注好的数据按 8:1:1 拆分为 train/val/test
    ├── 🐍 train_roberta.py           # [模型微调] 在RTX 5070上微调中文RoBERTa模型
    ├── 🐍 evaluate_model.py          # [模型评估] 在test集上测试表现，生成混淆矩阵和报告
    └── 🐍 predict_all.py             # [全量推理] 用微调好的模型，为那10w+清洗好的评论贴上5维情感标签
