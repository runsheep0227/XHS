# 小红书笔记内容分析模块

BERTopic详细地址(https://maartengr.github.io/BERTopic/index.html#citation)

相关论文:BERTopic、RoBERT-base

## 项目结构

content/
├── __pycache__/                       # Python 编译缓存（可忽略）
├── .vscode/                           # VS Code 配置（可选）
├── bertopic_model/                    # 保存的 BERTopic 模型文件
│   └── my_bertopic_model/             # 序列化保存的完整模型（safetensors安全格式），供可视化与二次推理调用
│       ├── config.json                # BERTopic 模型的基础架构与超参数配置
│       ├── ctfidf_config.json         # 基于类的 TF-IDF (c-TF-IDF) 提取规则配置
│       ├── ctfidf.safetensors         # 序列化保存的特征词权重张量矩阵
│       ├── topic_embeddings.safetensors # 序列化保存的各个主题的中心语义向量
│       └── topics.json                # 主题编号与核心特征词的映射字典
├── bertopic_ready_data/               # 清洗后待训练的数据
│   ├── bertopic_cleaned_data.json     #
│   ├── bertopic_docs.txt              #
│   ├── bertopic_train_data.csv        # 包括 note_id, cleaned_full_text, cleaned_seg_text,BERTopic 模型训练的主输入文件
│   ├── BERTopic数据清洗报告.txt        #
│   └── BERTopic完整清洗结果.xlsx       #
├── bertopic_result/                   # 主题建模结果输出目录 (核心学术成果)
│   ├── bertopic_topic_info.csv        # 宏观主题级结果：包含主题ID、笔记数量、Top特征词组、代表性原文片段
│   └── document_topics.csv            # 微观文档级结果：单篇笔记的主题归属、分配置信度(Probability)及 note_id
├── bertopic_result/                   # 主题建模结果输出
├── docasset/                          # 初始数据，包含大量特殊字段
├── aigc_xhs_stopwords.txt             # 停用词表（针对小红书/AIGC语境）
├── bertopic_train.py                  # 3.bertopic主训练脚本,输出bertopic_model、bertopic_result文件
├── bertopic_visual.py                 # bertopic可视化脚本
├── clean_data.py                      # 2.清理描述字段,输入cleaned_desc.json文件,通过note_id去重,输出bertopic_ready_data文件
├── clean_desc.py                      # 1.整合初始数据,仅保留note_id,note_url,title,decs字段,输出 cleand_desc.json文件
├── cleaned_desc.json                  # 由初始数据清洗后得到的结果
├── noteurlmove.py                     # url移动，用于后续评论爬取。
└── README.md                          # 项目说明

需要的安装包:

⚠️在RTX 50系列上运行Pytorch时会遇到CUDA兼容性问题。
核心问题在于PyTorch稳定版本的预编译二进制文件不支持sm_120计算能力，而RTX 50系列采用了较新的架构，需要更新的CUDA版本以及对应的PyTorch构建版本才能正常工作。
使用PyTorch Nightly构建新版本，但可能存在一定的不稳定性。
1.清理现有环境 pip uninstall troch trochvision torchaudio -y pip cache purge
2.安装支持CUDA 12.8的Nightly构建版本
pip3 install --pre torch torchvision torchaudio --index-
通过这一命令可以安装支持CUDA 12.8的PyTorch版本，该版本能够支持RTX 50系列的sm_120架构。
