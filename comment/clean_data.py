import json
import os
import re
import csv

# ===================== 核心路径配置：基于脚本自身的相对路径，移动文件无影响=====================
# 获取当前脚本所在目录（绝对路径），作为所有路径的基准
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 输入文件路径：clean_com.py输出的初步清洗结果
INPUT_FILE = os.path.join(SCRIPT_DIR, "cleaned_com.json")
# 无意义词表路径
MEANINGLESS_WORD_FILE = os.path.join(SCRIPT_DIR, "meaningless_word.txt")
# 输出文件夹路径
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "bert_data")
# 输出文件路径
# 1. 完整结构化数据，用于后续主题-情感关联分析
FINAL_JSON_FILE = os.path.join(OUTPUT_DIR, "final_cleaned_comments.json")
# 2. BERT训练就绪数据，符合您设定的标准训练格式
TRAIN_CSV_FILE = os.path.join(OUTPUT_DIR, "bert_train_ready.csv")
# 3. 清洗报告
REPORT_FILE = os.path.join(OUTPUT_DIR, "clean_data_report.txt")

# ===================== 清洗规则配置 =====================
# 【特殊符号过滤正则】：保留中文、英文、数字、中文标点（。！？，；：""''()（）、）、英文标点(.!?,;:"'())
# 仅去除无意义乱码、不可见字符、装饰性符号，保留情感相关标点
INVALID_CHAR_PATTERN = re.compile(
    r"[^\u4e00-\u9fa5a-zA-Z0-9\u3002\uff1f\uff01\uff0c\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\uff08\uff09\u3001.!?,;:\"'()\s]"
)
# 多个连续空白字符合并为单个空格
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# ===================== 工具函数 =====================
def load_meaningless_words(file_path: str) -> set:
    """
    加载无意义词表，做标准化预处理
    :param file_path: meaningless_word.txt路径
    :return: 去重后的无意义词集合
    """
    if not os.path.exists(file_path):
        print(f"警告：无意义词表 {file_path} 不存在，将跳过低信息熵过滤步骤")
        return set()
    
    meaningless_words = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # 去除首尾空格、换行，跳过空行、注释行（#开头）
                word = line.strip()
                if word and not word.startswith("#"):
                    meaningless_words.add(word)
    except Exception as e:
        print(f"警告：读取无意义词表失败，错误信息：{str(e)}，将跳过低信息熵过滤步骤")
        return set()
    
    print(f"成功加载无意义词表，共 {len(meaningless_words)} 个无意义词汇")
    return meaningless_words

def is_low_information_text(text: str, meaningless_words: set) -> bool:
    """
    低信息熵判断逻辑
    判断文本是否为低信息熵内容：将文本中所有无意义词替换后，剩余有效内容长度<2，判定为低信息
    避免了"包含无意义词就删除"的误判，仅过滤完全无有效信息的评论
    :param text: 待判断的评论内容
    :param meaningless_words: 无意义词集合
    :return: True=低信息需过滤，False=有效内容保留
    """
    if not meaningless_words:
        return False
    
    # 替换文本中所有无意义词
    filtered_text = text
    for word in meaningless_words:
        filtered_text = filtered_text.replace(word, "")
    
    # 替换后剩余有效内容长度<5，判定为低信息
    return len(filtered_text.strip()) < 5

# ===================== 主清洗流程 =====================
if __name__ == "__main__":
    print("="*60)
    print("开始执行小红书评论二次精细化清洗（MacBERT训练适配版）")
    print(f"【路径校验】脚本所在目录：{SCRIPT_DIR}")
    print(f"【路径校验】输入文件路径：{INPUT_FILE}")
    print(f"【路径校验】输出文件夹路径：{OUTPUT_DIR}")
    print("="*60)

    # -------------------- 1. 前置校验与初始化 --------------------
    # 校验输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件 {INPUT_FILE} 不存在！请先运行clean_com.py完成初步清洗")
        exit(1)
    
    # 创建输出文件夹，不存在则自动创建，已存在不报错
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载无意义词表
    meaningless_word_set = load_meaningless_words(MEANINGLESS_WORD_FILE)

    # 初始化统计指标（学术论文可直接引用）
    stats = {
        "total_input_count": 0,                # 输入总评论数
        "invalid_char_filtered": 0,             # 特殊符号清理后为空过滤条数
        "short_text_filtered": 0,               # 短文本（<2字符）过滤条数
        "low_info_filtered": 0,                  # 低信息熵文本过滤条数
        "duplicate_filtered": 0,                 # 二次去重过滤条数
        "final_valid_count": 0                   # 最终有效评论数
    }

    # 初始化结果容器
    final_cleaned_data = []
    content_unique_set = set()

    # -------------------- 2. 读取初步清洗数据 --------------------
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：输入文件 {INPUT_FILE} 格式错误，不是合法的JSON文件")
        exit(1)
    except Exception as e:
        print(f"错误：读取输入文件失败，错误信息：{str(e)}")
        exit(1)
    
    stats["total_input_count"] = len(raw_data)
    print(f"成功读取初步清洗数据，共 {stats['total_input_count']} 条评论")
    print("-"*60)

    # -------------------- 3. 逐条执行二次清洗 --------------------
    for index, comment in enumerate(raw_data):
        # 提取核心字段，容错处理
        note_id = comment.get("note_id", "")
        nickname = comment.get("nickname", "")
        content = comment.get("content", "")
        ip_location = comment.get("ip_location", "")
        original_content = content

        # -------------------- 清洗规则1：无意义特殊符号清理 --------------------
        # 去除无效特殊字符
        cleaned_content = INVALID_CHAR_PATTERN.sub("", content)
        # 合并多个连续空白字符
        cleaned_content = MULTI_SPACE_PATTERN.sub(" ", cleaned_content)
        # 去除首尾空格
        cleaned_content = cleaned_content.strip()

        # 特殊符号清理后为空，直接过滤
        if not cleaned_content:
            stats["invalid_char_filtered"] += 1
            continue

        # -------------------- 清洗规则2：短文本过滤（长度<5字符） --------------------
        if len(cleaned_content) < 5:  # 这里设置为5字符以下，避免过于严格导致情感表达被误过滤
            stats["short_text_filtered"] += 1
            continue

        # -------------------- 清洗规则3：基于无意义词表的低信息熵过滤 --------------------
        if is_low_information_text(cleaned_content, meaningless_word_set):
            stats["low_info_filtered"] += 1
            continue

        # -------------------- 清洗规则4：二次内容去重，兜底保证数据唯一性 --------------------
        if cleaned_content in content_unique_set:
            stats["duplicate_filtered"] += 1
            continue
        content_unique_set.add(cleaned_content)

        # -------------------- 有效数据留存 --------------------
        final_cleaned_data.append({
            "note_id": note_id,
            "nickname": nickname,
            "content": cleaned_content,
            "ip_location": ip_location,
            "original_content": original_content  # 保留原始内容，用于后续校验与学术溯源
        })

    # 更新最终有效数据量
    stats["final_valid_count"] = len(final_cleaned_data)

    # -------------------- 4. 输出清洗结果 --------------------
    # 4.1 输出完整结构化JSON数据
    try:
        with open(FINAL_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(final_cleaned_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"错误：输出JSON文件失败，错误信息：{str(e)}")
        exit(1)

    # 4.2 输出MacBERT训练就绪CSV文件（标准格式：note_id、txt，后续可直接新增label列）
    try:
        with open(TRAIN_CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(["note_id", "txt"])
            # 写入数据
            for item in final_cleaned_data:
                writer.writerow([item["note_id"], item["content"]])
    except Exception as e:
        print(f"错误：输出训练CSV文件失败，错误信息：{str(e)}")
        exit(1)

    # 4.3 生成学术清洗报告
    report_content = f"""
============================================================
小红书评论数据二次清洗报告
============================================================
一、基础信息
- 输入数据来源：cleaned_com.json
- 无意义词表：meaningless_word.txt
- 清洗时间：{os.path.getmtime(FINAL_JSON_FILE)}
- 脚本版本：MacBERT训练

二、清洗统计数据
1. 输入总评论数：{stats['total_input_count']} 条
2. 特殊符号清理后为空过滤：{stats['invalid_char_filtered']} 条
3. 长度<2字符短文本过滤：{stats['short_text_filtered']} 条
4. 低信息熵无意义文本过滤：{stats['low_info_filtered']} 条
5. 二次重复内容过滤：{stats['duplicate_filtered']} 条
------------------------------------------------------------
6. 最终有效评论数：{stats['final_valid_count']} 条
7. 数据有效留存率：{round(stats['final_valid_count']/stats['total_input_count']*100, 2)}%

三、输出文件说明
1. final_cleaned_comments.json：完整结构化数据，包含全字段与原始内容，用于主题-情感关联分析
2. macbert_train_ready.csv：MacBERT模型训练就绪数据，标准格式为note_id、txt，可直接新增label列用于微调
3. clean_data_report.txt：本清洗报告
============================================================
    """
    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report_content)
    except Exception as e:
        print(f"警告：生成清洗报告失败，错误信息：{str(e)}")

    # -------------------- 5. 控制台输出结果 --------------------
    print("="*60)
    print("二次精细化清洗全部完成！")
    print(f"输入总评论数：{stats['total_input_count']} 条")
    print(f"最终有效评论数：{stats['final_valid_count']} 条")
    print(f"数据有效留存率：{round(stats['final_valid_count']/stats['total_input_count']*100, 2)}%")
    print("-"*60)
    print(f"完整结构化数据已保存至：{FINAL_JSON_FILE}")
    print(f"训练数据已保存至：{TRAIN_CSV_FILE}")
    print(f"清洗报告已保存至：{REPORT_FILE}")
    print("="*60)