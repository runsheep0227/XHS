import json
import os
import re

# ===================== 路径（相对脚本目录）=====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "rawdata")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "cleaned_com.json")

# ===================== 配置 =====================
# 输出保留字段（顺序即 JSON 中键顺序）
REQUIRED_FIELDS = ["note_id", "content", "user_id", "nickname"]

AT_PATTERN = re.compile(r"@\S+\s*")
EMOJI_PATTERN = re.compile(r"\[([^\[\]]+)\]")
FILE_NAME_PREFIX = "search_comments_"

MIN_TEXT_LENGTH = 5
PURE_EMOJI_PATTERN = re.compile(
    r"^[😀-🙏\U0001F300-\U0001F9FF哈呵嘻嘿呀哎呃啊嗯\s。，、；：？！…~\-_]*$"
)
PURE_NUMBER_PATTERN = re.compile(r"^\d+$")
PURE_REPEATED_PATTERN = re.compile(r"^(.)\1{2,}$")


def process_emoji_text(match):
    emoji_text = match.group(1).strip()
    return emoji_text.rstrip("R").strip()


def pick_fields(comment):
    """从单条评论中取出保留字段，None 转为空字符串。"""
    out = {}
    for key in REQUIRED_FIELDS:
        val = comment.get(key)
        out[key] = "" if val is None else val
    return out


def is_meaningful_content(content):
    if len(content) < MIN_TEXT_LENGTH:
        return False, "too_short"
    if PURE_NUMBER_PATTERN.match(content):
        return False, "pure_number"
    if PURE_REPEATED_PATTERN.match(content):
        return False, "pure_repeated"
    if PURE_EMOJI_PATTERN.match(content):
        return False, "pure_emoji"
    return True, "valid"


if __name__ == "__main__":
    print("=" * 70)
    print("小红书评论清洗：表情文字提取、@ 去除、质量过滤、按笔记+正文去重")
    print(
        f"保留字段：{', '.join(REQUIRED_FIELDS)}，另写入 original_content（@/表情处理前的原文）"
    )
    print("-" * 70)
    print(f"脚本目录：{SCRIPT_DIR}")
    print(f"原始数据目录：{RAW_DATA_DIR}")
    print(f"输出文件：{OUTPUT_FILE}")
    print(f"最小正文长度：{MIN_TEXT_LENGTH}")
    print("-" * 70)

    if not os.path.isdir(RAW_DATA_DIR):
        print("错误：未找到 rawdata 目录（应与 clean_com.py 同级）。")
        raise SystemExit(1)

    raw_files = [
        n
        for n in os.listdir(RAW_DATA_DIR)
        if n.startswith(FILE_NAME_PREFIX) and n.endswith(".json")
    ]
    if not raw_files:
        print(f"错误：rawdata 中无 `{FILE_NAME_PREFIX}*.json` 文件。")
        raise SystemExit(1)

    cleaned_result = []
    # 同一正文在不同 note_id 下各保留一条；仅当 (笔记, 清洗后正文) 已出现时丢弃
    note_content_seen = set()
    total_raw_count = 0
    pure_at_filtered = 0
    too_short_filtered = 0
    pure_number_filtered = 0
    pure_repeated_filtered = 0
    pure_emoji_filtered = 0
    duplicate_filtered = 0
    emoji_processed_count = 0

    for file_name in raw_files:
        path = os.path.join(RAW_DATA_DIR, file_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError:
            print(f"  警告：{file_name} JSON 无效，已跳过")
            continue
        except OSError as e:
            print(f"  警告：读取 {file_name} 失败：{e}，已跳过")
            continue

        comment_list = raw_data if isinstance(raw_data, list) else raw_data.get("data", [])
        total_raw_count += len(comment_list)

        for comment in comment_list:
            cleaned_comment = pick_fields(comment)
            original_content = cleaned_comment["content"]
            if not isinstance(original_content, str):
                original_content = str(original_content) if original_content is not None else ""
            processed_content = AT_PATTERN.sub("", original_content)

            if EMOJI_PATTERN.search(processed_content):
                emoji_processed_count += 1
            processed_content = EMOJI_PATTERN.sub(process_emoji_text, processed_content).strip()
            cleaned_comment["original_content"] = original_content
            cleaned_comment["content"] = processed_content

            meaningful, reason = is_meaningful_content(processed_content)
            if not meaningful:
                if not processed_content and AT_PATTERN.sub("", original_content).strip():
                    pure_at_filtered += 1
                else:
                    if reason == "too_short":
                        too_short_filtered += 1
                    elif reason == "pure_number":
                        pure_number_filtered += 1
                    elif reason == "pure_repeated":
                        pure_repeated_filtered += 1
                    elif reason == "pure_emoji":
                        pure_emoji_filtered += 1
                continue

            note_id = cleaned_comment["note_id"]
            if not isinstance(note_id, str):
                note_id = str(note_id) if note_id is not None else ""
            dedupe_key = (note_id, processed_content)
            if dedupe_key in note_content_seen:
                duplicate_filtered += 1
                continue
            note_content_seen.add(dedupe_key)
            cleaned_result.append(cleaned_comment)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned_result, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"错误：写入失败：{e}")
        raise SystemExit(1)

    print("\n【统计】")
    print("-" * 70)
    print(f"原始评论数：           {total_raw_count:>8}")
    print(f"├ 纯@ 等无效：         {pure_at_filtered:>8}")
    print(f"├ 过短（<{MIN_TEXT_LENGTH}）：         {too_short_filtered:>8}")
    print(f"├ 纯数字：             {pure_number_filtered:>8}")
    print(f"├ 纯重复字：           {pure_repeated_filtered:>8}")
    print(f"├ 纯表情/符号：        {pure_emoji_filtered:>8}")
    print(f"└ 同笔记同正文重复：   {duplicate_filtered:>8}")
    filtered = (
        pure_at_filtered
        + too_short_filtered
        + pure_number_filtered
        + pure_repeated_filtered
        + pure_emoji_filtered
        + duplicate_filtered
    )
    print("-" * 70)
    print(f"过滤合计：             {filtered:>8}")
    print(f"保留条数：             {len(cleaned_result):>8}")
    print(f"含表情的处理条数：     {emoji_processed_count:>8}")
    pct = (filtered / total_raw_count * 100) if total_raw_count else 0.0
    print(f"过滤率：               {pct:>7.1f}%")
    print("-" * 70)
    print(f"已写入：{OUTPUT_FILE}")
    print("=" * 70)
