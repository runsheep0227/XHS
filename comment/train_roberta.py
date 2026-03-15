import os
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    """
    【学术指标计算器】
    在每一轮(Epoch)训练结束后，计算验证集上的 Accuracy 和 Macro-F1 分数。
    Macro-F1 是学术论文中最看重的指标，因为它能客观反映模型在各个情绪类别上的综合表现，不受样本数量不均衡的影响。
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    # 使用 macro 平均来计算 F1 分数
    f1 = f1_score(labels, predictions, average='macro')
    
    return {"accuracy": acc, "f1_macro": f1}

def main():
    print("🤖 正在为您启动 RoBERTa 模型微调程序...")
    
    # 1. 路径配置：稳妥获取绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, 'bert_data', 'train.csv')
    val_file = os.path.join(current_dir, 'bert_data', 'val.csv')
    output_model_dir = os.path.join(current_dir, 'saved_model')
    
    # 确保保存模型的文件夹存在
    os.makedirs(output_model_dir, exist_ok=True)

    # 2. 【加载数据】读取 CSV 并转换为 Hugging Face 支持的 Dataset 格式
    print("📁 正在加载训练集和验证集...")
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    
    # 清理可能存在的空值，并将 label 列重命名为 labels (这是 Hugging Face Trainer 的底层强制要求)
    df_train = df_train.dropna(subset=['content']).rename(columns={'label': 'labels'})
    df_val = df_val.dropna(subset=['content']).rename(columns={'label': 'labels'})
    
    # 确保标签是整数类型
    df_train['labels'] = df_train['labels'].astype(int)
    df_val['labels'] = df_val['labels'].astype(int)

    train_dataset = Dataset.from_pandas(df_train[['content', 'labels']])
    val_dataset = Dataset.from_pandas(df_val[['content', 'labels']])

    # 3. 【加载预训练模型与分词器】使用哈工大中文 RoBERTa-wwm-ext
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    print(f"📥 正在从 Hugging Face 下载/加载预训练大脑: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 设置 num_labels=3，因为我们有 0, 1, 2 共三个情感维度（正面/中立/负面）
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # 4. 【数据编码】将纯文本的评论转换为模型认识的数字张量 (Tensors)
    def tokenize_function(examples):
        # 针对 8G 显存优化：max_length=128，足够覆盖绝大多数小红书评论
        return tokenizer(examples["content"], truncation=True, max_length=256)

    print("✂️ 正在对文本进行分词与张量化编码...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # 动态填充器：自动把同一个 Batch 里的句子补齐到相同长度，极大地节省显存
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. 设置训练超参数为您 RTX 5070 8GB 量身定制的黄金配置
    print("⚙️ 正在配置训练超参数...")
    training_args = TrainingArguments(
        output_dir=os.path.join(current_dir, 'checkpoint_temp'), 
        num_train_epochs=4,                 
        per_device_train_batch_size=16,     
        per_device_eval_batch_size=32,      
        learning_rate=2e-5,                 
        weight_decay=0.01,                  
        eval_strategy="epoch",              
        save_strategy="epoch",              
        load_best_model_at_end=True,        
        metric_for_best_model="f1_macro",   
        greater_is_better=True,
        fp16=True,                          
        logging_steps=10,                   # [修改]: 删除了 logging_dir 避免警告
        save_total_limit=2,                 
        seed=42                             
    )

    # 6. 【初始化 Trainer 并开始训练】
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,         # [修改]: 将 tokenizer=tokenizer 改为 processing_class=tokenizer 适配新版
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("🔥 准备就绪！启动高强度训练！")
    trainer.train()

    # 7. 【保存最终的巅峰模型】
    print(f"🎉 训练圆满结束！正在将最强形态的模型保存至: {output_model_dir}")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    print("💡 微调已全部完成！下一步我们可以使用 evaluate_model.py 来对测试集进行评估啦！")

if __name__ == "__main__":
    # 清理一下显存缓存，确保以最干净的状态启动
    torch.cuda.empty_cache()
    main()