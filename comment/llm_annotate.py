import os
import json
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import time

# ================= 预备配置区（适配LM Studio本地API） =================
BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
MODEL_NAME = "qwen/qwen3.5-9b"

# ================= 输出路径配置 =================
OUTPUT_DIR = r"E:\document\PG\studio\comment\bert_data"
OUTPUT_FILENAME = "llm_labeled_result"

# 初始化本地OpenAI客户端
try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("✅ OpenAI 客户端初始化成功")
except Exception as e:
    print(f"❌ 客户端初始化失败：{e}")
    exit()

SYSTEM_PROMPT = """你是一个专门研究社交媒体情感倾向的情感分析专家。你的任务是分析小红书关于AIGC（人工智能生成内容）的评论的态度。

### 标签定义：
1. 【标签 1：正向】：表达赞美、惊喜、期待、安利、求教学。
   - 例子："AI画的也太精美了！"、"求一个提示词"、"提高效率神器"。
2. 【标签 0：中性】：纯客观描述、询问价格/工具、不带情绪的陈述。
   - 例子："这是用Midjourney做的吗？"、"AIGC目前还在发展阶段"。
3. 【标签 -1：负向】：表达焦虑、抵触、批判、反感、吐槽质量差。
   - 例子："AI味儿太重了，看着反胃"、"画师要失业了，真恶心"、"手画得稀碎"。

### 注意事项：
1. 识别反讽：如“真能吹”、“绝绝子（用于负面时）”、“画得‘真好’（带引号）”通常为负向。
2. 识别需求：如“蹲一个提示词”、“求带”通常为正向。

### 标注逻辑（思维链）：
对于每条评论，请按以下步骤思考：
1. 提取关键词和核心语气（注意反讽和Emoji）。
2. 判断评论者对AIGC的态度。
3. 给出最终标签。

### 示例（Few-Shot）：
用户："这AI画的太绝了，求带！" -> 思考：关键词"太绝了"表示惊叹，"求带"表示认可。 -> 结果：{"label": 1}
用户："又是AI，现在满大街都是这种塑料感的图。" -> 思考：关键词"塑料感"表示廉价、厌恶。 -> 结果：{"label": -1}
用户："请问这个是用什么软件生成的？" -> 思考：纯工具性询问，无明显情感。 -> 结果：{"label": 0}

### 输出格式（必须是纯JSON）：
{"reason": "分析理由", "label": 数字}
数字只能是 1, 0, -1。"""

def get_llm_label(content):
    retry_count = 3
    for i in range(retry_count):
        try:
            response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请标注这条评论：{content}"}
            ],
            temperature=0.01,  # 建议调低到 0.01 提高稳定性
            max_tokens=150,    # 稍微调大一点，给模型留出思考空间
            timeout=200
            )
            
            reply = response.choices[0].message.content.strip()
            match = re.search(r'(\{.*?"label"\s*:\s*(-?\d).*?\})', reply, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                result = json.loads(json_str)
                
                # 这里的 label 获取增加一个默认值检查
                label = result.get('label')
                # 兼容模型可能输出字符串 "1" 的情况
                if label is not None:
                    label = int(label)
                
                if label in [1, 0, -1]:
                    # 如果 Prompt 里要求了 reason，可以一并返回方便后续人工校验
                    reason = result.get('reason', '无理由')
                    return label, f"[{reason}] | 原回复: {reply}"
                else:
                    return 2, f"标签值非法：{label}"
            else:
                # 容错处理：如果没搜到 JSON，尝试直接搜数字（最后的倔强）
                # 寻找回复中是否孤零零出现过 1, 0, -1
                last_ditch = re.findall(r'(?<=^|[^-\d])(-1|0|1)(?=[^-\d]|$)', reply)
                if last_ditch:
                    return int(last_ditch[0]), f"正则降级匹配: {reply}"
                
                return 2, f"未找到JSON格式：{reply}"
        
        except Exception as e:
            error_msg = str(e)
            if i < retry_count - 1:
                delay = 2 ** i
                tqdm.write(f"⚠️ 失败重试({i+1}/3)：{error_msg}")
                time.sleep(delay)
                continue
            return -2, f"最终失败：{error_msg}"

def process_single_item(item):
    note_id = item.get('note_id', '')
    content = item.get('content', '')
    
    if not content.strip():
        return None
        
    label, raw_reply = get_llm_label(content)
    return {
        'note_id': note_id,
        'content': content,
        'label': label,
        'raw_response': raw_reply
    }

LABEL_NAMES = {
    1: "正向情感（强正向/弱正向）",
    0: "中性情感（客观/中立）",
    -1: "负向情感（弱负向/强负向）",
    2: "解析失败（标签值非法）",
    -2: "调用失败（API/网络错误）"
}

def get_label_statistics(df):
    label_counts = df['label'].value_counts().sort_index()
    total_success = len(df[(df['label'] != -2) & (df['label'] != 2)])
    positive_count = label_counts.get(1, 0)
    neutral_count = label_counts.get(0, 0)
    negative_count = label_counts.get(-1, 0)
    
    stats_data = [
        {'标签编码': 1, '标签名称': '正向情感', '数量': positive_count, '占比(%)': round((positive_count/total_success)*100, 2) if total_success > 0 else 0},
        {'标签编码': 0, '标签名称': '中性情感', '数量': neutral_count, '占比(%)': round((neutral_count/total_success)*100, 2) if total_success > 0 else 0},
        {'标签编码': -1, '标签名称': '负向情感', '数量': negative_count, '占比(%)': round((negative_count/total_success)*100, 2) if total_success > 0 else 0},
        {'标签编码': 2, '标签名称': '解析失败', '数量': label_counts.get(2, 0), '占比(%)': round((label_counts.get(2, 0)/len(df))*100, 2) if len(df) > 0 else 0},
        {'标签编码': -2, '标签名称': '调用失败', '数量': label_counts.get(-2, 0), '占比(%)': round((label_counts.get(-2, 0)/len(df))*100, 2) if len(df) > 0 else 0},
        {'标签编码': '总计', '标签名称': '有效标注', '数量': total_success, '占比(%)': round((total_success/len(df))*100, 2) if len(df) > 0 else 0},
        {'标签编码': '总计', '标签名称': '标注失败', '数量': label_counts.get(-2, 0) + label_counts.get(2, 0), '占比(%)': round(((label_counts.get(-2, 0) + label_counts.get(2, 0))/len(df))*100, 2) if len(df) > 0 else 0},
        {'标签编码': '总计', '标签名称': '数据总数', '数量': len(df), '占比(%)': 100.0}
    ]
    
    return pd.DataFrame(stats_data)

def print_label_statistics(df):
    stats_df = get_label_statistics(df)
    
    print("\n" + "="*60)
    print("📈 3层情感标签统计结果")
    print("="*60)
    
    for _, row in stats_df.iterrows():
        if row['标签编码'] in [1, 0, -1]:
            print(f"🔹 {row['标签名称']}: {row['数量']} 条 ({row['占比(%)']}%)")
    
    print("\n❌ 错误统计：")
    for _, row in stats_df.iterrows():
        if row['标签编码'] in [2, -2]:
            print(f"🔹 {row['标签名称']}: {row['数量']} 条 ({row['占比(%)']}%)")
    
    print("\n📊 总体统计：")
    for _, row in stats_df.iterrows():
        if row['标签编码'] == '总计':
            print(f"🔹 {row['标签名称']}: {row['数量']} 条 ({row['占比(%)']}%)")
    
    print("="*60)

def main():
    print("📌 开始调用本地模型进行3层情感标注（正向/中性/负向）...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'bert_data', 'llm_sample_data.json')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_data_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.csv")
    output_stats_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}_stats.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件：{input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
        
    print(f"✅ 成功读取 {len(sample_data)} 条待标注数据。")
    print(f"📁 输出路径已配置为：{OUTPUT_DIR}")
    print("⚠️ 模型推理中，请耐心等待...\n")

    results = []
    MAX_WORKERS = 1  # 先单线程，稳定优先
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_item, item): item for item in sample_data}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sample_data), desc="3层情感标注"):
            result = future.result()
            if result is not None:
                results.append(result)
                
                short_content = result['content'][:30].replace('\n', ' ')
                label_desc = LABEL_NAMES.get(result['label'], "未知标签")
                tqdm.write(f"📝 评论：[{short_content}...] ➡️ {label_desc}({result['label']})")
        
    df = pd.DataFrame(results)
    call_fail_count = len(df[df['label'] == -2])
    parse_fail_count = len(df[df['label'] == 2])
    total_fail_count = call_fail_count + parse_fail_count
    
    if total_fail_count > 0:
        print(f"\n⚠️ 标注失败统计：调用失败 {call_fail_count} 条 | 解析失败 {parse_fail_count} 条")
        
    df.to_csv(output_data_file, index=False, encoding='utf-8-sig')
    print(f"\n🎉 3层情感标注完成！结果已保存至：{output_data_file}")
    
    stats_df = get_label_statistics(df)
    stats_df.to_csv(output_stats_file, index=False, encoding='utf-8-sig')
    print(f"📊 统计结果已保存至：{output_stats_file}")
    
    print(f"\n📊 标注汇总：")
    print(f"   有效标注：{len(results)-total_fail_count} 条")
    print(f"   正向情感：{len(df[df['label'] == 1])} 条")
    print(f"   中性情感：{len(df[df['label'] == 0])} 条")
    print(f"   负向情感：{len(df[df['label'] == -1])} 条")
    print(f"   标注失败：{total_fail_count} 条")
    
    print_label_statistics(df)

if __name__ == "__main__":
    main()