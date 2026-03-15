import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ================= 1. 字体与标签配置 =================
# 【为学术图表配置中文字体】防止热力图上的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统下使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 我们定义的 5 个情感维度，用于图表展示
TARGET_NAMES =["惊叹/期待", "赞同/喜悦", "客观/中立", "焦虑/担忧", "愤怒/抵触"]

# ================= 2. 简易的数据集类 =================
class TestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    print("📊 正在为您启动模型学术评估程序...")
    
    # ================= 3. 路径配置 =================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, 'bert_data', 'test.csv')
    model_dir = os.path.join(current_dir, 'saved_model')
    results_dir = os.path.join(current_dir, 'results')
    
    # 确保输出图表的 results 文件夹存在
    os.makedirs(results_dir, exist_ok=True)
    
    report_file = os.path.join(results_dir, 'evaluation_report.txt')
    cm_file = os.path.join(results_dir, 'confusion_matrix.png')

    if not os.path.exists(model_dir):
        print(f"❌ 找不到模型文件夹：{model_dir}\n请确保您已经成功运行了 train_roberta.py。")
        return

    # ================= 4. 加载测试数据 =================
    print("📁 正在读取测试集...")
    df_test = pd.read_csv(test_file)
    df_test = df_test.dropna(subset=['content', 'label'])
    
    texts = df_test['content'].tolist()
    true_labels = df_test['label'].tolist()
    
    # ================= 5. 加载微调好的模型与分词器 =================
    print("🧠 正在唤醒您的专属 RoBERTa 模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ 使用设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval() # 切换到评估模式，锁定权重

    # ================= 6. 批处理推理 (防止显存溢出) =================
    test_dataset = TestDataset(texts, true_labels, tokenizer)
    # 测试时不需要计算梯度，batch_size 可以设置大一点，如 32 或 64
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    pred_labels =[]
    
    print("🔍 正在对测试集进行盲测推理...")
    with torch.no_grad(): # 禁用梯度计算，极大地节省显存并加速
        for batch in tqdm(test_loader, desc="推理进度"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 取概率最大的类别作为预测结果
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            pred_labels.extend(preds)

    # ================= 7. 计算指标 =================
    print("\n📈 正在为您生成评估报告...")
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    
    # 详细的分类报告（精确率、召回率、F1）
    class_report = classification_report(
        true_labels, pred_labels, 
        target_names=TARGET_NAMES, 
        digits=4 # 保留4位小数，彰显学术严谨
    )

    # 组装文本报告
    final_report = (
        "====================================================\n"
        "           AIGC 情感分析模型 - 终极测试报告\n"
        "====================================================\n"
        f"🎯 总体准确率 (Accuracy):  {acc:.4f}\n"
        f"🏆 宏F1分数 (Macro-F1):    {macro_f1:.4f}\n\n"
        "📊 细粒度情绪分类表现:\n"
        f"{class_report}\n"
        "====================================================\n"
    )
    
    print(final_report)
    
    # 将报告保存到 results 文件夹
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(final_report)

    # ================= 8. 绘制并保存混淆矩阵热力图 =================
    print("🎨 正在为您绘制混淆矩阵热力图...")
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    # 使用 seaborn 绘制热力图，cmap选择学术界常用的蓝调 (Blues)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                annot_kws={"size": 12})
                
    plt.title('模型情感分类混淆矩阵 (Confusion Matrix)', fontsize=15, pad=15)
    plt.ylabel('真实情绪标签 (True Label)', fontsize=12)
    plt.xlabel('模型预测情绪 (Predicted Label)', fontsize=12)
    plt.xticks(rotation=45) # x轴标签稍微旋转，防止文字挤在一起
    plt.tight_layout()
    
    # 保存高分辨率图片，dpi=300 直接满足期刊论文印刷要求
    plt.savefig(cm_file, dpi=300)
    plt.close()
    
    print(f"🎉 评估大功告成！\n📄 报告已保存至: {report_file}\n🖼️ 混淆矩阵图已保存至: {cm_file}")
    print("💡 您可以直接打开 results 文件夹，欣赏您的科研成果啦！")

if __name__ == "__main__":
    main()