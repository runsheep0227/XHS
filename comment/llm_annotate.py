import os
import json
import pandas as pd
import re
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import time

# ================= 预备配置区（适配LM Studio本地API） =================
# 正确的本地API地址
BASE_URL = "http://127.0.0.1:1234/v1"  # 替换为你的LM Studio本地地址和端口
# API Key随便填（LM Studio本地调用不需要真实Key）
API_KEY = "lm-studio"
# 替换为你从LM Studio复制的Model Identifier（示例，以你实际的为准）
MODEL_NAME = "qwen/qwen3.5-9b"

# ================= 输出路径配置 =================
# 统一的输出文件夹路径
OUTPUT_DIR = r"E:\document\PG\studio\comment\bert_data"
# 输出文件名
OUTPUT_FILENAME = "llm_labeled_result"

# 初始化本地OpenAI客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 优化后的Prompt（3层情感分类） =================
SYSTEM_PROMPT = """你是一个专业的社会学与社会媒体数据分析专家，正在进行一项关于公众对“AIGC（人工智能生成内容）”态度的学术研究。
你的任务是：根据我提供的小红书评论，判断该评论所属的【3层情感维度】。
请严格从以下 3个类别中选择 1 个最贴切的标签（输出对应的数字 1, 0, -1）：

标签 1：【正向情感】包含所有积极情绪，如：
- 强正向：惊叹/期待/兴奋/激动（如"太震撼了！这AI画风绝了"、"未来已来，迫不及待想用"）
- 弱正向：赞同/喜悦/好奇/满意（如"用起来真方便"、"这个功能很实用"）

标签 0：【中性情感】无情感倾向的陈述性内容，仅描述事实或中性观点（如"AIGC在文案创作有潜力"、"技术仍在迭代中"）

标签 -1：【负向情感】包含所有消极情绪，如：
- 弱负向：焦虑/担忧/不满/失望（如"担心AI取代设计师"、"内容真实性存疑"）
- 强负向：愤怒/抵触/厌恶/反感（如"坚决抵制AI生成内容"、"这玩意儿太可怕了"）

【重要要求】：
1. 当评论同时包含正负情绪时，选择主导情绪
2. 请仅输出 JSON 格式的结果，不要有任何其他解释或多余字符
3. 格式必须为：{"label": 数字}（数字只能是1/0/-1）"""

def get_llm_label(content):
    """调用LM Studio本地模型获取3层情感标签，带重试机制"""
    retry_count = 3
    for i in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请标注这条评论的3层情感标签：{content}"}
                ],
                temperature=0.1,  # 低温度，输出更稳定（3分类适合低temperature）
                max_tokens=100,   # 输出限制，避免过长回复
                timeout=60
            )
            
            reply = response.choices[0].message.content.strip()
            # 提取JSON部分
            match = re.search(r'\{.*"label".*?\}', reply, re.DOTALL)
            if match:
                json_str = match.group(0)
                result = json.loads(json_str)
                label = int(result.get('label', 2))
                # 验证标签是否在合法范围内
                if label in [1, 0, -1]:
                    return label, reply
                else:
                    return 2, f"标签值非法：{label}，原始回复：{reply}"
            else:
                return 2, f"未找到JSON格式结果：{reply}"
        
        except Exception as e:
            error_msg = str(e)
            if i < retry_count - 1:
                delay = 2 ** i
                tqdm.write(f"⚠️ 调用失败，{delay}秒后重试：{error_msg}")
                time.sleep(delay)
                continue
            return -2, f"最终失败：{error_msg}"  # 用-2区分标注失败和标签错误

def process_single_item(item):
    """处理单条数据，返回标注结果"""
    note_id = item.get('note_id', '')
    content = item.get('content', '')
    
    if not content.strip():
        return None
        
    label, raw_reply = get_llm_label(content)
    return {
        'note_id': note_id,
        'content': content,
        'label': label,  # 1=正向 0=中性 -1=负向 -2=调用失败 2=解析失败
        'raw_response': raw_reply
    }

# 新增：3层标签名称映射（核心修改）
LABEL_NAMES = {
    1: "正向情感（强正向/弱正向）",
    0: "中性情感（客观/中立）",
    -1: "负向情感（弱负向/强负向）",
    2: "解析失败（标签值非法）",
    -2: "调用失败（API/网络错误）"
}

def get_label_statistics(df):
    """获取3层标签统计结果（返回DataFrame）"""
    # 统计各标签数量
    label_counts = df['label'].value_counts().sort_index()
    
    # 计算总标注成功数（排除-2和2）
    total_success = len(df[(df['label'] != -2) & (df['label'] != 2)])
    # 计算3类核心标签数量
    positive_count = label_counts.get(1, 0)
    neutral_count = label_counts.get(0, 0)
    negative_count = label_counts.get(-1, 0)
    
    # 构建统计数据
    stats_data = [
        # 核心3层标签
        {'标签编码': 1, '标签名称': '正向情感', '数量': positive_count, 
         '占比(%)': round((positive_count/total_success)*100, 2) if total_success > 0 else 0},
        {'标签编码': 0, '标签名称': '中性情感', '数量': neutral_count, 
         '占比(%)': round((neutral_count/total_success)*100, 2) if total_success > 0 else 0},
        {'标签编码': -1, '标签名称': '负向情感', '数量': negative_count, 
         '占比(%)': round((negative_count/total_success)*100, 2) if total_success > 0 else 0},
        # 错误统计
        {'标签编码': 2, '标签名称': '解析失败', '数量': label_counts.get(2, 0), 
         '占比(%)': round((label_counts.get(2, 0)/len(df))*100, 2) if len(df) > 0 else 0},
        {'标签编码': -2, '标签名称': '调用失败', '数量': label_counts.get(-2, 0), 
         '占比(%)': round((label_counts.get(-2, 0)/len(df))*100, 2) if len(df) > 0 else 0},
        # 总计行
        {'标签编码': '总计', '标签名称': '有效标注', '数量': total_success, 
         '占比(%)': round((total_success/len(df))*100, 2) if len(df) > 0 else 0},
        {'标签编码': '总计', '标签名称': '标注失败', '数量': label_counts.get(-2, 0) + label_counts.get(2, 0), 
         '占比(%)': round(((label_counts.get(-2, 0) + label_counts.get(2, 0))/len(df))*100, 2) if len(df) > 0 else 0},
        {'标签编码': '总计', '标签名称': '数据总数', '数量': len(df), '占比(%)': 100.0}
    ]
    
    return pd.DataFrame(stats_data)

def print_label_statistics(df):
    """打印3层标签统计结果"""
    stats_df = get_label_statistics(df)
    
    print("\n" + "="*60)
    print("📈 3层情感标签统计结果")
    print("="*60)
    
    # 打印核心3层标签
    for _, row in stats_df.iterrows():
        if row['标签编码'] in [1, 0, -1]:
            print(f"🔹 {row['标签名称']}: {row['数量']} 条 ({row['占比(%)']}%)")
    
    # 打印错误统计
    print("\n❌ 错误统计：")
    for _, row in stats_df.iterrows():
        if row['标签编码'] in [2, -2]:
            print(f"🔹 {row['标签名称']}: {row['数量']} 条 ({row['占比(%)']}%)")
    
    # 打印总计
    print("\n📊 总体统计：")
    for _, row in stats_df.iterrows():
        if row['标签编码'] == '总计':
            print(f"🔹 {row['标签名称']}: {row['数量']} 条 ({row['占比(%)']}%)")
    
    print("="*60)

def main():
    print("📌 开始调用本地模型进行3层情感标注（正向/中性/负向）...")
    
    # 配置输入输出路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'bert_data', 'llm_sample_data.json')
    
    # 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 标注结果文件路径（CSV）
    output_data_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.csv")
    # 统计结果文件路径（CSV）
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
    MAX_WORKERS = 2  # 根据本地硬件调整，避免显存不足
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_item, item): item for item in sample_data}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sample_data), desc="3层情感标注"):
            result = future.result()
            if result is not None:
                results.append(result)
                
                # 打印简洁的标注结果
                short_content = result['content'][:30].replace('\n', ' ')
                label_desc = LABEL_NAMES.get(result['label'], "未知标签")
                tqdm.write(f"📝 评论：[{short_content}...] ➡️ {label_desc}({result['label']})")
        
    df = pd.DataFrame(results)
    # 计算各类失败数量
    call_fail_count = len(df[df['label'] == -2])  # 调用失败
    parse_fail_count = len(df[df['label'] == 2])  # 解析失败
    total_fail_count = call_fail_count + parse_fail_count
    
    if total_fail_count > 0:
        print(f"\n⚠️ 标注失败统计：调用失败 {call_fail_count} 条 | 解析失败 {parse_fail_count} 条")
        print("💡 可查看raw_response列排查失败原因（如模型输出格式错误、网络超时等）")
        
    # 保存标注结果（UTF-8带BOM，兼容Excel）
    df.to_csv(output_data_file, index=False, encoding='utf-8-sig') 
    print(f"\n🎉 3层情感标注完成！结果已保存至：{output_data_file}")
    
    # 生成并保存统计结果
    stats_df = get_label_statistics(df)
    stats_df.to_csv(output_stats_file, index=False, encoding='utf-8-sig')
    print(f"📊 统计结果已保存至：{output_stats_file}")
    
    # 打印核心统计
    print(f"\n📊 标注汇总：")
    print(f"   有效标注：{len(results)-total_fail_count} 条")
    print(f"   正向情感：{len(df[df['label'] == 1])} 条")
    print(f"   中性情感：{len(df[df['label'] == 0])} 条")
    print(f"   负向情感：{len(df[df['label'] == -1])} 条")
    print(f"   标注失败：{total_fail_count} 条")
    
    # 打印详细统计结果
    print_label_statistics(df)

if __name__ == "__main__":
    # 可选：测试单条数据的3层标注效果
    """
    test_cases = [
        ("AIGC太牛了，科幻成真！", 1),  # 正向
        ("AIGC技术仍在迭代中", 0),       # 中性
        ("担心AI取代设计师工作", -1)    # 负向
    ]
    for content, expected_label in test_cases:
        label, reply = get_llm_label(content)
        print(f"\n测试评论：{content}")
        print(f"预期标签：{expected_label} | 实际标签：{label}")
        print(f"模型回复：{reply[:100]}")
    exit()  # 测试完成后注释掉这行，运行完整标注
    """
    main()