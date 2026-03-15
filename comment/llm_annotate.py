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
BASE_URL = "http://192.168.1.3:1234/v1"  # 替换为你的LM Studio本地地址和端口
# API Key随便填（LM Studio本地调用不需要真实Key）
API_KEY = "lm-studio"
# 替换为你从LM Studio复制的Model Identifier（示例，以你实际的为准）
MODEL_NAME = "openai/gpt-oss-20b"

# ================= 输出路径配置（新增） =================
# 统一的输出文件夹路径
OUTPUT_DIR = r"E:\document\PG\studio\comment\bert_data"
# 输出文件名（不带后缀）
OUTPUT_FILENAME = "llm_labeled_result"

# 初始化本地OpenAI客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 优化后的Prompt（适配9B模型，指令更清晰） =================
SYSTEM_PROMPT = """你是一个专业的社会学与社会媒体数据分析专家，正在进行一项关于公众对“AIGC（人工智能生成内容）”态度的学术研究。
你的任务是：根据我提供的小红书评论，判断该评论所属的【细粒度情感维度】。
请严格从以下 5 个类别中选择 1 个最贴切的标签（输出对应的数字 0-4）：

标签 0：【惊叹/期待/兴奋/激动】（强正向）表现出强烈技术崇拜与未来憧憬，情绪高涨如'太震撼了！这AI画风绝了'、'未来已来，迫不及待想用'、'人类创造力被彻底解放'。典型特征：感叹词密集（太/绝/神/炸裂）、未来展望（即将/未来/革命性）、主动表达兴奋（迫不及待/想体验）
标签 1：【赞同/喜悦/好奇/满意】（弱/中度正向）温和积极的认同感，情绪平和但明显正面，如'用起来真方便'、'这个功能很实用'、'帮我省下不少时间'。典型特征：肯定性词汇（好/实用/不错）、功能受益描述（省时/提效/便捷）、无强烈情绪词（无'太'、'绝'等）
标签 2：【客观/中立】（中立）无情感倾向的陈述性内容，仅描述事实或中性观点，如'AIGC在文案创作有潜力'、'技术仍在迭代中'、'需要验证真实性'。典型特征：中性词汇（有潜力/在迭代/需验证）、无情感动词（不使用'好/坏/担心'）、聚焦技术本身而非情绪
标签 3：【焦虑/担忧/不满/失望】（弱/中度负向）对技术潜在风险的温和忧虑，情绪谨慎但负面，如'担心AI取代设计师'、'内容真实性存疑'、'需要加强监管'。典型特征：担忧性词汇（担心/存疑/风险）、风险指向（取代/真实性/监管）、条件句（如果/可能/需要）
标签 4：【愤怒/抵触/厌恶/反感】（强负向）强烈排斥与对抗情绪，表达激烈不满，如'坚决抵制AI生成内容'、'这玩意儿太可怕了'、'必须禁止AIGC'。典型特征：绝对化表述（必须/坚决/绝对）、威胁性词汇（可怕/危害/污染）、情绪化动词（抵制/禁止/拒绝）

【重要要求】：请仅输出 JSON 格式的结果，不要有任何其他解释或多余字符。当评论同时包含正负情绪时，选择主导情绪（如'功能不错但担心隐私'→选标签3）
格式必须为：{"label": 数字}"""

def get_llm_label(content):
    """调用LM Studio本地模型获取标签，带重试机制"""
    retry_count = 3
    for i in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请标注这条评论的情感标签：{content}"}
                ],
                temperature=0.8,  # 低温度，输出更稳定
                max_tokens=200,   # 输出限制，避免过长回复
                timeout=60
            )
            
            reply = response.choices[0].message.content.strip()
            match = re.search(r'\{.*"label".*?\}', reply, re.DOTALL)
            if match:
                json_str = match.group(0)
                result = json.loads(json_str)
                return int(result.get('label', 2)), reply
            else:
                return 2, f"解析失败：{reply}"
        
        except Exception as e:
            error_msg = str(e)
            if i < retry_count - 1:
                delay = 2 ** i
                tqdm.write(f"⚠️ 调用失败，{delay}秒后重试：{error_msg}")
                time.sleep(delay)
                continue
            return -1, f"最终失败：{error_msg}"

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

# 新增：定义标签名称映射，方便统计展示
LABEL_NAMES = {
    0: "惊叹/期待/兴奋/激动（强正向）",
    1: "赞同/喜悦/好奇/满意（弱/中度正向）",
    2: "客观/中立",
    3: "焦虑/担忧/不满/失望（弱/中度负向）",
    4: "愤怒/抵触/厌恶/反感（强负向）",
    -1: "标注失败"
}

def get_label_statistics(df):
    """获取标签统计结果（返回DataFrame）"""
    # 统计各标签数量
    label_counts = df['label'].value_counts().sort_index()
    
    # 计算总标注成功数（排除-1）
    total_success = len(df[df['label'] != -1])
    
    # 构建统计数据
    stats_data = []
    for label_code in [-1, 0, 1, 2, 3, 4]:
        count = label_counts.get(label_code, 0)
        label_name = LABEL_NAMES.get(label_code, f"未知标签({label_code})")
        
        # 计算占比
        if label_code == -1:
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        else:
            percentage = (count / total_success) * 100 if total_success > 0 else 0
        
        stats_data.append({
            '标签编码': label_code,
            '标签名称': label_name,
            '数量': count,
            '占比(%)': round(percentage, 2)
        })
    
    # 添加总计行
    stats_data.append({
        '标签编码': '总计',
        '标签名称': '有效标注',
        '数量': total_success,
        '占比(%)': round((total_success/len(df))*100, 2) if len(df) > 0 else 0
    })
    stats_data.append({
        '标签编码': '总计',
        '标签名称': '标注失败',
        '数量': label_counts.get(-1, 0),
        '占比(%)': round((label_counts.get(-1, 0)/len(df))*100, 2) if len(df) > 0 else 0
    })
    stats_data.append({
        '标签编码': '总计',
        '标签名称': '数据总数',
        '数量': len(df),
        '占比(%)': 100.0
    })
    
    return pd.DataFrame(stats_data)

def print_label_statistics(df):
    """打印标签统计结果"""
    stats_df = get_label_statistics(df)
    
    print("\n" + "="*60)
    print("📈 标签数量统计结果")
    print("="*60)
    
    # 打印详细统计
    for _, row in stats_df.iterrows():
        if row['标签编码'] != '总计':
            print(f"🔹 {row['标签名称']}: {row['数量']} 条 ({row['占比(%)']}%)")
        else:
            print(f"📊 {row['标签名称']}: {row['数量']} 条 ({row['占比(%)']}%)")
    
    print("="*60)

def main():
    print("📌 开始调用本地模型进行情感标注（8GB显存）...")
    
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
    MAX_WORKERS = 2  # 模型并发设为2，效率更高
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_item, item): item for item in sample_data}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sample_data), desc="模型标注"):
            result = future.result()
            if result is not None:
                results.append(result)
                
                short_content = result['content'][:20].replace('\n', ' ')
                clean_reply = result['raw_response'][:50].replace('\n', ' ')
                tqdm.write(f"📝 评论：[{short_content}...] ➡️ 标签：{result['label']}")
        
    df = pd.DataFrame(results)
    fail_count = len(df[df['label'] == -1])
    
    if fail_count > 0:
        print(f"\n⚠️ 有 {fail_count} 条数据标注失败，可查看raw_response列排查原因。")
        
    # 保存标注结果
    df.to_csv(output_data_file, index=False, encoding='utf-8-sig') 
    print(f"\n🎉 标注完成！结果已保存至：{output_data_file}")
    
    # 生成并保存统计结果
    stats_df = get_label_statistics(df)
    stats_df.to_csv(output_stats_file, index=False, encoding='utf-8-sig')
    print(f"📊 统计结果已保存至：{output_stats_file}")
    
    print(f"📊 标注成功：{len(results)-fail_count} 条 | 失败：{fail_count} 条")
    
    # 打印统计结果
    print_label_statistics(df)

if __name__ == "__main__":
    # 在main函数开头添加测试代码
    """
    test_content = "AIGC太牛了，科幻成真！"  # 应该标0
    test_label, test_reply = get_llm_label(test_content)
    print(f"测试结果：评论[{test_content}] → 标签[{test_label}]，原始回复[{test_reply}]")
    exit()  # 测试完退出，确认正确后再注释掉
    """
    main()