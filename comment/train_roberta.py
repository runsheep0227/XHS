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

# ==================== 核心配置与标签映射 ====================
# 新增：您定义的5层标签名称映射（核心修改）
LABEL_NAMES = {
    1: "正向情感（强正向/弱正向）",
    0: "中性情感（客观/中立）",
    -1: "负向情感（弱负向/强负向）",
    2: "解析失败（标签值非法）",
    -2: "调用失败（API/网络错误）"
}

# 模型底层不认识负数，必须将其映射为 0, 1, 2
LABEL2ID = {-1: 0, 0: 1, 1: 2}
ID2LABEL = {0: -1, 1: 0, 2: 1}

def compute_metrics(eval_pred):
    """计算 Accuracy 和 Macro-F1（完美适配原始标签 -1/0/1）"""
    logits, labels = eval_pred
    
    # 模型输出的是内部ID（0/1/2），先还原为原始标签（-1/0/1）
    pred_ids = np.argmax(logits, axis=-1)
    predictions = [ID2LABEL[pid] for pid in pred_ids]  
    true_labels = [ID2LABEL[lid] for lid in labels]    
    
    acc = accuracy_score(true_labels, predictions)
    # 采用 Macro-F1 保证小众情绪也能被公平评估
    f1 = f1_score(true_labels, predictions, average='macro')
    
    return {"accuracy": acc, "f1_macro": f1}

def main():
    print("🤖 正在为您启动 RoBERTa 模型微调程序（标签：-1负面 / 0中立 / 1正面）...")
    
    # ==================== 1. 稳妥的路径配置 ====================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, 'bert_data', 'train.csv')
    val_file = os.path.join(current_dir, 'bert_data', 'val.csv')
    output_model_dir = os.path.join(current_dir, 'saved_model')
    
    os.makedirs(output_model_dir, exist_ok=True)

    # ==================== 2. 高级数据清洗与加载 ====================
    print("📁 正在加载并清洗训练集和验证集...")
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    
    # 【核心过滤逻辑】：仅保留 label 为 -1, 0, 1 的有效数据
    # 自动丢弃 2(解析失败) 和 -2(调用失败) 的数据
    valid_labels =[-1, 0, 1]
    
    df_train = df_train[
        (df_train['content'].notna()) & 
        (df_train['content'].str.strip() != '') &
        (df_train['label'].isin(valid_labels))
    ].copy()
    
    df_val = df_val[
        (df_val['content'].notna()) & 
        (df_val['content'].str.strip() != '') &
        (df_val['label'].isin(valid_labels))
    ].copy()
    
    if len(df_train) == 0 or len(df_val) == 0:
        print("❌ 灾难性错误：训练集/验证集无有效数据！请检查 CSV 文件中是否包含 -1, 0, 1 标签。")
        return
    
    # 打印清洗后的分布情况，彰显学术严谨性
    print("\n📊 训练集有效数据分布：")
    train_ori_counts = df_train['label'].value_counts().sort_index()
    for label in valid_labels:
        count = train_ori_counts.get(label, 0)
        print(f"   ➤ {LABEL_NAMES[label]}: {count} 条")
        
    # 将业务标签 (-1,0,1) 转换为模型可读的 ID (0,1,2)
    df_train['labels'] = df_train['label'].map(LABEL2ID).astype(int)
    df_val['labels'] = df_val['label'].map(LABEL2ID).astype(int)

    # 封装为 HuggingFace 标准 Dataset
    train_dataset = Dataset.from_pandas(df_train[['content', 'labels']])
    val_dataset = Dataset.from_pandas(df_val[['content', 'labels']])

    # ==================== 3. 模型与分词器初始化 ====================
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    print(f"\n📥 正在加载预训练大脑: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 显式告知模型：我们做的是 3 分类，且绑定了映射字典
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL
    )

    # ==================== 4. 显存优化：动态分词与张量化 ====================
    def tokenize_function(examples):
        # 对于小红书评论，128字能覆盖99%的文本。
        return tokenizer(examples["content"], truncation=True, max_length=128)

    print("✂️ 正在对文本进行高效分词与张量编码...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["content"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["content"])
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ==================== 5. RTX 5070 黄金训练超参数 ====================
    print("⚙️ 正在为您注入 8GB 显卡专属的黄金训练配置...")
    training_args = TrainingArguments(
        output_dir=os.path.join(current_dir, 'checkpoint_temp'), 
        num_train_epochs=4,                 
        per_device_train_batch_size=16,      # 8G显卡极值，若依然爆显存，请改为8并设置 gradient_accumulation_steps=2
        per_device_eval_batch_size=32,       # 评估不占用梯度显存，可调大
        learning_rate=2e-5,                  # RoBERTa微调的真理学习率
        warmup_ratio=0.1,                    # 【工程师优化】：预热前10%步数，防止初期梯度爆炸，收敛更平滑
        weight_decay=0.01,                  
        eval_strategy="epoch",              
        save_strategy="epoch",              
        load_best_model_at_end=True,        
        metric_for_best_model="f1_macro",   
        greater_is_better=True,
        fp16=True,                           # 【核心】半精度计算，显存占用直接减半！
        logging_steps=10,                   
        save_total_limit=2,                 
        seed=42,
        report_to="none"                     # 关闭 wandb 等第三方日志看板
    )

    # ==================== 6. 初始化并引爆训练 ====================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,         # 【修正错误】：适配最新版 Transformers！旧版为 tokenizer=tokenizer
        data_collator=data_collator,
        compute_metrics=compute_metrics     
    )

    print("🔥 引擎全开！RTX 5070 开始高强度反向传播！")
    trainer.train()

    # ==================== 7. 保存巅峰模型与配置 ====================
    print(f"\n🎉 训练圆满完成！正在保存最强权重至: {output_model_dir}")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    # 额外保存您定义的业务配置（方便全量推理时自动翻译标签）
    import json
    config_path = os.path.join(output_model_dir, "label_mapping.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "LABEL2ID": LABEL2ID, 
            "ID2LABEL": ID2LABEL,
            "LABEL_NAMES": LABEL_NAMES
        }, f, ensure_ascii=False, indent=4)
        
    print(f"📝 标签字典已同步保存至: {config_path}")
    print("\n💡 工程师提示：微调结束！您的专属大模型已出炉，赶紧使用 evaluate_model.py 查收它的考试成绩单吧！")

if __name__ == "__main__":
    # 【工程师绝招】：训练前强制清空显存碎片，确保以 100% 的可用状态启动！
    torch.cuda.empty_cache()
    # 开启 CUDA 同步调试（若遇底层报错可快速定位）
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()