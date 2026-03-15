import json
import os
import re

# ===================== 基于脚本自身的相对路径配置 =====================
# 获取当前脚本所在的目录（绝对路径），作为所有相对路径的基准
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始数据文件夹：脚本同级的rawdata文件夹
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "rawdata")
# 清洗后输出文件：与脚本同级
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "cleaned_com.json")

# ===================== 功能配置项 =====================
# 需要保留的核心字段
REQUIRED_FIELDS = ["note_id", "nickname", "content", "ip_location"]
# 匹配@用户的正则表达式（适配小红书@格式，支持多个@、中英文用户名）
AT_PATTERN = re.compile(r"@\S+\s*")
# 匹配小红书[表情名]格式的正则表达式
EMOJI_PATTERN = re.compile(r"\[([^\[\]]+)\]")
# 原始数据文件名匹配规则
FILE_NAME_PATTERN = "search_comments_"

# ===================== 表情文字处理回调函数=====================
def process_emoji_text(match):
    """
    处理匹配到的表情内容：
    1. 提取括号内的文字
    2. 去除末尾的大写R（仅去除末尾，不影响中间的R）
    3. 去除处理后的首尾空格
    """
    emoji_text = match.group(1).strip()
    # 仅去除末尾的大写R，使用rstrip('R')，如果末尾没有R则不做处理
    cleaned_emoji = emoji_text.rstrip('R').strip()
    return cleaned_emoji

# ===================== 初始化变量 =====================
cleaned_result = []
content_unique_set = set()
# 统计指标
total_raw_count = 0           # 原始总数据量
pure_at_filtered = 0          # 纯@数据过滤条数
duplicate_filtered = 0        # 重复内容过滤条数
emoji_processed_count = 0     # 成功处理表情的评论条数
pure_emoji_filtered = 0       # 纯表情无意义内容过滤条数

# ===================== 核心清洗逻辑 =====================
if __name__ == "__main__":
    print("="*60)
    print("开始执行小红书评论初期数据清洗（含表情核心文字提取+去R优化）...")
    print("-"*60)
    print(f"【路径校验】当前脚本所在目录：{SCRIPT_DIR}")
    print(f"【路径校验】原始数据文件夹路径：{RAW_DATA_DIR}")
    print(f"【路径校验】输出文件路径：{OUTPUT_FILE}")

    # 1. 检查原始数据文件夹是否存在
    if not os.path.exists(RAW_DATA_DIR):
        print(f"错误：原始数据文件夹不存在！请确认 rawdata 文件夹与 clean_com.py 放在同一级目录")
        exit(1)

    # 2. 遍历文件夹，获取所有符合命名规则的json文件
    raw_file_list = [
        file for file in os.listdir(RAW_DATA_DIR)
        if file.startswith(FILE_NAME_PATTERN) and file.endswith(".json")
    ]

    if not raw_file_list:
        print(f"错误：在 rawdata 文件夹中未找到符合 search_comments_xxx.json 命名规则的文件！")
        exit(1)

    # 3. 逐个处理每个原始数据文件
    for file_name in raw_file_list:
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        
        try:
            # 读取json文件，适配utf-8编码，避免中文乱码
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError:
            print(f"  警告：文件 {file_name} 格式错误，无法解析，已跳过")
            continue
        except Exception as e:
            print(f"  警告：读取文件 {file_name} 失败，错误信息：{str(e)}，已跳过")
            continue

        # 兼容两种常见的爬虫数据结构：根节点为数组 / 根节点为字典且数据在data字段中
        comment_list = raw_data if isinstance(raw_data, list) else raw_data.get("data", [])
        file_raw_count = len(comment_list)
        total_raw_count += file_raw_count

        # 4. 逐条清洗评论数据
        for comment in comment_list:
            # -------------------- 清洗规则1：仅保留指定字段 --------------------
            cleaned_comment = {
                field: comment.get(field, "") for field in REQUIRED_FIELDS
            }
            original_content = cleaned_comment["content"]
            processed_content = original_content

            # -------------------- 清洗规则2：去除@用户字段 --------------------
            processed_content = AT_PATTERN.sub("", processed_content)

            # -------------------- 清洗规则3：提取表情核心文字+去末尾R --------------------
            # 先判断是否包含表情，用于统计
            if EMOJI_PATTERN.search(processed_content):
                emoji_processed_count += 1
            # 使用回调函数处理每个匹配到的表情，去除末尾R
            processed_content = EMOJI_PATTERN.sub(process_emoji_text, processed_content)

            # -------------------- 统一去除首尾空格，清理无效空白 --------------------
            processed_content = processed_content.strip()
            cleaned_comment["content"] = processed_content

            # -------------------- 过滤纯@/纯表情无意义数据（清洗后内容为空） --------------------
            if not processed_content:
                # 区分是纯@还是纯表情过滤，用于统计
                if len(original_content.strip()) > 0 and not AT_PATTERN.sub("", original_content).strip():
                    pure_at_filtered += 1
                else:
                    pure_emoji_filtered += 1
                continue

            # -------------------- 清洗规则4：content去重，完全相同的内容仅保留一条 --------------------
            if processed_content in content_unique_set:
                duplicate_filtered += 1
                continue
            content_unique_set.add(processed_content)
            cleaned_result.append(cleaned_comment)

    # 5. 将清洗结果写入输出文件
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned_result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"错误：写入输出文件失败，错误信息：{str(e)}")
        exit(1)

    # 6. 输出清洗统计结果
    print("数据清洗全部完成！")
    print(f"原始总评论数：{total_raw_count} 条")
    print(f"过滤纯@无意义评论：{pure_at_filtered} 条")
    print(f"过滤纯表情无意义评论：{pure_emoji_filtered} 条")
    print(f"处理表情的有效评论：{emoji_processed_count} 条")
    print(f"过滤重复内容评论：{duplicate_filtered} 条")
    print(f"最终保留有效评论：{len(cleaned_result)} 条")
    print(f"清洗后数据已保存至：{OUTPUT_FILE}")
    print("="*60)