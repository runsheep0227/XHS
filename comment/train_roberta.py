import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score

# ==================== 1. 核心配置 ====================
LABEL2ID = {"-1": 0, "0": 1, "1": 2}
ID2LABEL = {0: "-1", 1: "0", 2: "1"}

# ==================== 2. 评估指标计算 (优化版) ====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 处理可能存在的 tuple (如果模型输出复杂结构)
    if isinstance(logits, tuple):
        logits = logits[0]
        
    predictions = np.argmax(logits, axis=-1)

    # 直接基于 ID (0, 1, 2) 计算，避免来回映射，提升效率
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    
    # 计算每个类的 F1，供内部参考，不再使用 print 干扰进度条
    f1_per_class = f1_score(labels, predictions, average=None, labels=[0, 1, 2])
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_neg": f1_per_class[0],
        "f1_neu": f1_per_class[1],
        "f1_pos": f1_per_class[2]
    }

# ==================== 3. 加权损失 Trainer ====================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 使用 get 而不是 pop，避免破坏 inputs 字典影响其他评估插件
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            # 确保 weights 与 logits 在同一设备 (GPU)
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    print("🚀 RTX 5070 Optimized Version Starting...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext" 
    
    # --- 加载数据 ---
    df_train = pd.read_csv(os.path.join(current_dir, 'bert_data', 'train.csv'))
    df_val = pd.read_csv(os.path.join(current_dir, 'bert_data', 'val.csv'))

    valid_labels = [-1, 0, 1]
    df_train = df_train[df_train['label'].isin(valid_labels)].copy()
    df_val = df_val[df_val['label'].isin(valid_labels)].copy()
    
    df_train['labels'] = df_train['label'].astype(str).map(LABEL2ID).astype(int)
    df_val['labels'] = df_val['label'].astype(str).map(LABEL2ID).astype(int)

    # --- 权重计算 ---
    train_label_counts = df_train['labels'].value_counts().sort_index()
    total = len(df_train)
    weights = [total / (3 * train_label_counts.get(i, 1)) for i in [0, 1, 2]]
    class_weights = torch.tensor(weights, dtype=torch.float32)
    # 归一化权重以防 Loss 初始值过大/过小导致梯度爆炸
    class_weights = class_weights / class_weights.mean()

    # --- 模型加载 ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL
    )

    # --- 数据处理 ---
    train_dataset = Dataset.from_pandas(df_train[['content', 'labels']])
    val_dataset = Dataset.from_pandas(df_val[['content', 'labels']])

    # 动态 padding (DataCollator) 结合固定截断
    MAX_LEN = 128
    tokenized_train = train_dataset.map(lambda x: tokenizer(x["content"], truncation=True, max_length=MAX_LEN), batched=True)
    tokenized_val = val_dataset.map(lambda x: tokenizer(x["content"], truncation=True, max_length=MAX_LEN), batched=True)

    # --- 训练参数 (RTX 5070 8GB 终极优化) ---
    # 根据你的操作系统设置 workers
    num_workers = 0 if os.name == 'nt' else 4 # nt 代表 Windows

    training_args = TrainingArguments(
        output_dir=os.path.join(current_dir, 'checkpoint_temp'),
        num_train_epochs=10,
        learning_rate=2e-5,
        
        # 显存管理策略
        per_device_train_batch_size=16, # 如果后续报 OOM，改为 8
        gradient_accumulation_steps=1,  # 如果 batch_size 改为 8，这里改为 2 即可保持原有效果
        per_device_eval_batch_size=32,
        
        # 硬件加速核心
        bf16=True, 
        optim="adamw_8bit", # 【新增优化】使用 8-bit AdamW 极大地节省优化器显存消耗
        
        # Dataloader设置
        dataloader_num_workers=num_workers, # 避免 Windows 死锁
        dataloader_pin_memory=True,
        
        # 策略优化
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2, # 减少磁盘占用
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
        
        # 【预留方案】如果不小心把 max_length 加大到了 512 导致显存不够，将下面这行设为 True
        gradient_checkpointing=False 
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("🔥 开始本地微调 (请使用任务管理器监控 8GB 显存使用量)...")
    trainer.train()
    
    # 保存结果
    save_path = os.path.join(current_dir, 'saved_model')
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ 任务完成，模型保存在: {save_path}")

if __name__ == "__main__":
    # 清理缓存，确保开局有干净的显存空间
    torch.cuda.empty_cache()
    main()