import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# ================= 1. 情感标签映射字典 =================
# 将数字映射回中文标签
LABEL_MAP = {
    0: "惊叹/期待",
    1: "赞同/喜悦",
    2: "客观/中立",
    3: "焦虑/担忧",
    4: "愤怒/抵触"
}

# ================= 2. 推理数据集类 =================
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # 如果遇到空文本，用空格代替防止报错
        if not text.strip():
            text = " "
            
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def main():
    print("🚀 推理程序已启动！正在为您调集算力...")

    # ================= 3. 稳妥的路径配置 =================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入：咱们最开始清洗好的 10w+ 全量数据
    input_file = os.path.join(current_dir, 'bert_data', 'final_cleaned_comments.json')
    # 模型：刚刚微调好的最优模型大脑
    model_dir = os.path.join(current_dir, 'saved_model')
    # 输出：最终带有情感标签的完整数据，存放在 results 里供后续分析
    output_file = os.path.join(current_dir, 'results', 'final_sentiment_com.json')

    if not os.path.exists(model_dir):
        print(f"❌ 找不到模型文件夹：{model_dir}\n主人，请确保模型已成功保存。")
        return
    if not os.path.exists(input_file):
        print(f"❌ 找不到全量数据：{input_file}\n请检查 bert_data 目录下是否有该文件。")
        return

    # ================= 4. 加载 10w+ 原始数据 =================
    print("📁 正在读取全量待预测数据，这可能需要几秒钟...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    total_len = len(all_data)
    print(f"✅ 成功读取 {total_len} 条评论！")

    # 提取所有的 content 用于推理
    texts =[item.get('content', '') for item in all_data]

    # ================= 5. 加载模型与配置硬件 =================
    print("🧠 正在将您的专属 RoBERTa 模型加载至 RTX 5070...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval() # 切换到评估模式

    # ================= 6. 高效批处理推理 =================
    # 因为不需要计算梯度 (不需要反向传播)，8G显存可以轻松吞下更大的 Batch Size！
    # 推荐设置为 64 或 128，极大地缩短整体推理时间。
    BATCH_SIZE = 64 
    
    dataset = InferenceDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds =[]
    
    print(f"⚡ 引擎全开！开始对 {total_len} 条数据进行高并发降维打击...")
    
    # 禁用梯度计算，显存占用断崖式下跌，速度起飞
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="打标进度 (Batch)"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 拿到预测的类别索引 (0-4)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)

    # ================= 7. 组装终极数据并落盘 =================
    print("\n📦 推理完成！正在将情感标签完美融合进原始数据中...")
    
    for i in range(total_len):
        pred_label = all_preds[i]
        # 在原始数据字典中新增两个核心字段
        all_data[i]['emotion_label'] = pred_label
        all_data[i]['emotion_text'] = LABEL_MAP.get(pred_label, "未知")

    # 写入最终的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print("==================================================================")
    print(f"🎉 报告主人！伟大的里程碑达成！全量数据情感分析彻底完工！")
    print(f"💾 包含 {total_len} 条评论及 5 维情感标签的最终文件已保存至：\n   {output_file}")
    print("==================================================================")
    print("💡 学术闭环提示：")
    print("   现在，您可以利用这些数据回答您最初提出的核心假设：")
    print("   『针对AI伦理方面的笔记更容易引起用户的过激的情感共鸣』")
    print("   只需将这个 JSON 里的数据，根据 note_id 关联上您第一部分的 BERTopic 笔记主题即可！")

if __name__ == "__main__":
    main()