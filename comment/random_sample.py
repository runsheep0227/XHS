import os
import json
import random

def main():
    print("正在执行数据随机抽样与字段精简...")
    
    # 动态获取当前脚本所在目录的绝对路径 (即 .../comment/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 动态拼接数据路径
    input_file = os.path.join(current_dir, 'bert_data', 'final_cleaned_comments.json')
    output_file = os.path.join(current_dir, 'bert_data', 'llm_sample_data.json')
    
    # 设置抽样数量，可以随时在这里修改
    SAMPLE_SIZE = 5000
    
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件：{input_file}\n请确保前面清洗数据的脚本已成功生成该文件。")
        return

    # 2. 读取全量清洗好的数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_len = len(data)
    print(f"✅ 成功读取全量数据，共计 {total_len} 条评论。")
    
    if total_len < SAMPLE_SIZE:
        print(f"⚠️ 全量数据不足 {SAMPLE_SIZE} 条，将直接提取全部数据。")
        SAMPLE_SIZE = total_len

    # 3. 设定随机种子，保证每次抽样结果一致（可复现）
    random.seed(42)
    
    # 4. 执行随机抽样
    sampled_data = random.sample(data, SAMPLE_SIZE)
    
    # 5. 过滤冗余字段，大模型仅需要 note_id 和 content
    filtered_sample_data =[]
    for item in sampled_data:
        filtered_sample_data.append({
            "note_id": item.get("note_id", ""),
            "content": item.get("content", "")
        })
    
    # 6. 保存精简后的抽样结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_sample_data, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 抽样与精简完成！成功抽取 {SAMPLE_SIZE} 条纯净数据，已保存至 {output_file}。")

if __name__ == "__main__":
    main()