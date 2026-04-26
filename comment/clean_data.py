import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional, Sequence, Tuple

# ===================== 核心路径配置：基于脚本自身的相对路径，移动文件无影响=====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "cleaned_com.json")
MEANINGLESS_WORD_FILE = os.path.join(SCRIPT_DIR, "meaningless_word.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "bert_data")
FINAL_JSON_FILE = os.path.join(OUTPUT_DIR, "final_cleaned_comments.json")
TRAIN_CSV_FILE = os.path.join(OUTPUT_DIR, "bert_train_ready.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "clean_data_report.txt")

# ===================== 清洗规则配置 =====================
# 输入：cleaned_com.json 中 content 为一次清洗后正文；若存在 original_content（clean_com 写入的 @/表情处理前原文），
# 则输出的 original_content 沿用该字段以便溯源，否则与 content 相同。
# 输出字段说明：二次清洗结果仅保留 note_id、nickname、content、original_content；
# 不保留评论 IP 地址、属地等标识类字段（上游若存在，本脚本不落盘）。
# 清洗后有效文本最短长度（字符数），低于则丢弃
MIN_CLEANED_LENGTH = 5
# 剔除无意义词后剩余文本最短长度，低于则视为低信息熵
LOW_INFO_REMAINING_MIN = 5

INVALID_CHAR_PATTERN = re.compile(
    r"[^\u4e00-\u9fa5a-zA-Z0-9\u3002\uff1f\uff01\uff0c\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u3001.!?,;:\"'()\s]"
)
MULTI_SPACE_PATTERN = re.compile(r"\s+")


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
                word = line.strip()
                if word and not word.startswith("#"):
                    meaningless_words.add(word)
    except Exception as e:
        print(f"警告：读取无意义词表失败，错误信息：{str(e)}，将跳过低信息熵过滤步骤")
        return set()

    print(f"成功加载无意义词表，共 {len(meaningless_words)} 个无意义词汇")
    return meaningless_words


def meaningless_words_strip_order(meaningless_words: set) -> list:
    """按词长度降序排列，优先替换长串，减少多轮重叠带来的冗余扫描。"""
    return sorted(meaningless_words, key=len, reverse=True)


def is_low_information_text(
    text: str,
    meaningless_words: Sequence[str],
    remaining_min: int = LOW_INFO_REMAINING_MIN,
) -> bool:
    """
    低信息熵判断：将无意义词从文本中去掉后，若剩余有效内容长度 < remaining_min，则判定为低信息。
    meaningless_words 建议为按长度降序的序列（见 meaningless_words_strip_order）。
    """
    if not meaningless_words:
        return False

    filtered_text = text
    for word in meaningless_words:
        filtered_text = filtered_text.replace(word, "")

    return len(filtered_text.strip()) < remaining_min


def char_clean_content(content: str) -> str:
    """规则1：白名单字符保留，空白规范化。"""
    cleaned = INVALID_CHAR_PATTERN.sub("", content)
    cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def process_one_comment(
    comment: dict[str, Any],
    meaningless_words_ordered: Sequence[str],
    content_unique_set: set,
    *,
    min_cleaned_length: int = MIN_CLEANED_LENGTH,
    low_info_remaining_min: int = LOW_INFO_REMAINING_MIN,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    对单条评论执行二次清洗（不含落盘）。
    返回 (record, None) 表示保留；(None, stat_key) 表示丢弃，stat_key 用于累计统计。
    保留记录不含评论 IP / 属地等字段，仅业务所需文本与关联 id。
    """
    note_id = comment.get("note_id", "")
    nickname = comment.get("nickname", "")
    content = comment.get("content", "")
    if not isinstance(content, str):
        content = str(content) if content is not None else ""

    raw_upstream = comment.get("original_content")
    if isinstance(raw_upstream, str) and raw_upstream.strip():
        original_content = raw_upstream
    else:
        original_content = content

    cleaned_content = char_clean_content(content)
    if not cleaned_content:
        return None, "invalid_char_filtered"

    if len(cleaned_content) < min_cleaned_length:
        return None, "short_text_filtered"

    if is_low_information_text(
        cleaned_content, meaningless_words_ordered, remaining_min=low_info_remaining_min
    ):
        return None, "low_info_filtered"

    if cleaned_content in content_unique_set:
        return None, "duplicate_filtered"

    content_unique_set.add(cleaned_content)
    return {
        "note_id": note_id,
        "nickname": nickname,
        "content": cleaned_content,
        "original_content": original_content,
    }, None


def retention_percent(final_count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(final_count / total * 100, 2)


def format_report_timestamp(path: str) -> str:
    if not os.path.exists(path):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    print("=" * 60)
    print("开始执行小红书评论二次精细化清洗（MacBERT训练适配版）")
    print(f"【路径校验】脚本所在目录：{SCRIPT_DIR}")
    print(f"【路径校验】输入文件路径：{INPUT_FILE}")
    print(f"【路径校验】输出文件夹路径：{OUTPUT_DIR}")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件 {INPUT_FILE} 不存在！请先运行clean_com.py完成初步清洗")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    meaningless_word_set = load_meaningless_words(MEANINGLESS_WORD_FILE)
    meaningless_words_ordered = meaningless_words_strip_order(meaningless_word_set)

    stats = {
        "total_input_count": 0,
        "invalid_char_filtered": 0,
        "short_text_filtered": 0,
        "low_info_filtered": 0,
        "duplicate_filtered": 0,
        "final_valid_count": 0,
    }

    final_cleaned_data = []
    content_unique_set = set()

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：输入文件 {INPUT_FILE} 格式错误，不是合法的JSON文件")
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取输入文件失败，错误信息：{str(e)}")
        sys.exit(1)

    stats["total_input_count"] = len(raw_data)
    print(f"成功读取初步清洗数据，共 {stats['total_input_count']} 条评论")
    print("-" * 60)

    for comment in raw_data:
        record, drop_key = process_one_comment(
            comment,
            meaningless_words_ordered,
            content_unique_set,
            min_cleaned_length=MIN_CLEANED_LENGTH,
            low_info_remaining_min=LOW_INFO_REMAINING_MIN,
        )
        if record is not None:
            final_cleaned_data.append(record)
        elif drop_key == "invalid_char_filtered":
            stats["invalid_char_filtered"] += 1
        elif drop_key == "short_text_filtered":
            stats["short_text_filtered"] += 1
        elif drop_key == "low_info_filtered":
            stats["low_info_filtered"] += 1
        elif drop_key == "duplicate_filtered":
            stats["duplicate_filtered"] += 1

    stats["final_valid_count"] = len(final_cleaned_data)
    retention = retention_percent(stats["final_valid_count"], stats["total_input_count"])

    try:
        with open(FINAL_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(final_cleaned_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"错误：输出JSON文件失败，错误信息：{str(e)}")
        sys.exit(1)

    try:
        with open(TRAIN_CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["note_id", "txt"])
            for item in final_cleaned_data:
                writer.writerow([item["note_id"], item["content"]])
    except Exception as e:
        print(f"错误：输出训练CSV文件失败，错误信息：{str(e)}")
        sys.exit(1)

    train_csv_basename = os.path.basename(TRAIN_CSV_FILE)
    report_time = format_report_timestamp(FINAL_JSON_FILE)
    report_content = f"""
============================================================
小红书评论数据二次清洗报告
============================================================
一、基础信息
- 输入数据来源：cleaned_com.json
- 无意义词表：meaningless_word.txt
- 清洗时间：{report_time}
- 脚本版本：MacBERT训练
- 短文本阈值（字符）：{MIN_CLEANED_LENGTH}
- 低信息剩余长度阈值（字符）：{LOW_INFO_REMAINING_MIN}

二、清洗统计数据
1. 输入总评论数：{stats['total_input_count']} 条
2. 特殊符号清理后为空过滤：{stats['invalid_char_filtered']} 条
3. 长度<{MIN_CLEANED_LENGTH}字符短文本过滤：{stats['short_text_filtered']} 条
4. 低信息熵无意义文本过滤：{stats['low_info_filtered']} 条
5. 二次重复内容过滤：{stats['duplicate_filtered']} 条
------------------------------------------------------------
6. 最终有效评论数：{stats['final_valid_count']} 条
7. 数据有效留存率：{retention}%

三、输出文件说明
1. final_cleaned_comments.json：结构化数据（note_id、nickname、content、original_content），不含评论 IP；用于主题-情感关联分析
2. {train_csv_basename}：MacBERT模型训练就绪数据，标准格式为note_id、txt，可直接新增label列用于微调
3. clean_data_report.txt：本清洗报告
============================================================
    """
    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report_content)
    except Exception as e:
        print(f"警告：生成清洗报告失败，错误信息：{str(e)}")

    print("=" * 60)
    print("二次精细化清洗全部完成！")
    print(f"输入总评论数：{stats['total_input_count']} 条")
    print(f"最终有效评论数：{stats['final_valid_count']} 条")
    print(f"数据有效留存率：{retention}%")
    print("-" * 60)
    print(f"完整结构化数据已保存至：{FINAL_JSON_FILE}")
    print(f"训练数据已保存至：{TRAIN_CSV_FILE}")
    print(f"清洗报告已保存至：{REPORT_FILE}")
    print("=" * 60)
