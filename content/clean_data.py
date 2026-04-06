import json
import os
import re
import jieba
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

# ==================== 配置 ====================
WORK_DIR = Path(__file__).resolve().parent
INPUT_FILE = WORK_DIR / "cleaned_desc.json"
OUTPUT_DIR = WORK_DIR / "bertopic_ready_data"
LOG_DIR = WORK_DIR / "logs"
STOPWORDS_FILE = WORK_DIR / "aigc_xhs_stopwords.txt"
KEYWORDS_FILE = WORK_DIR / "aigc_keyword.txt"

# 清洗参数
KEEP_ENGLISH = True # 是否保留英文字符
MIN_FULL_TEXT_LEN = 20 # 最小文本长度（字符数）
MIN_SEG_WORD_COUNT = 10 # 最小分词数量（去除停用词后）

# ==================== 加载函数 ====================

def load_stopwords(filepath: Path) -> set:
    """加载停用词表"""
    if not filepath.exists():
        print(f"⚠️ 停用词文件不存在: {filepath}")
        return set()
    
    stopwords = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word and not word.startswith("#"):
                stopwords.add(word)
    
    print(f"📚 加载停用词: {len(stopwords)} 个")
    return stopwords


def load_keywords(filepath: Path) -> List[str]:
    """加载AI关键词并注入jieba"""
    if not filepath.exists():
        print(f"⚠️ 关键词文件不存在: {filepath}")
        return []
    
    keywords = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip()
            if kw and not kw.startswith("#"):
                keywords.append(kw)
                jieba.add_word(kw, freq=100000)
    
    print(f"🤖 加载AI关键词: {len(keywords)} 个")
    return keywords


# ==================== 清洗函数 ====================

def clean_illegal_chars(text: str) -> str:
    """清理非法字符"""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'[\u200b-\u200f\ufeff\ufff9-\ufffb\u2028\u2029\u00ad\u061c\u180e]+', '', text)
    text = re.sub(r'[\uE000-\uF8FF]', '', text)
    return text


def clean_text(title: str, desc: str, stopwords: set) -> Tuple[str, str, Dict]:
    """清洗单条文本"""
    info = {
        "has_title": bool(title and title.strip()),
        "has_desc": bool(desc and desc.strip()),
        "title_len": len(title) if title else 0,
        "desc_len": len(desc) if desc else 0
    }
    
    raw_text = f"{title}。 {desc}" if title else desc
    raw_text = str(raw_text) if raw_text else ""
    
    # 移除表情标签
    text = re.sub(r'\[[\u4e00-\u9fa5a-zA-Z0-9]+\]', '', raw_text)
    
    # 处理话题标签
    text = re.sub(r'#([^\[\#\s]+)\[话题\]#?', r' \1 ', text)
    text = re.sub(r'#(\S+)', r' \1 ', text)
    
    # 移除@和链接
    text = re.sub(r'@\S+|https?://\S+|www\.\S+', '', text)
    
    # 只保留中文/英文+空格
    if KEEP_ENGLISH:
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', ' ', text)
    else:
        text = re.sub(r'[^\u4e00-\u9fa5\s]', ' ', text)
    
    # 移除数字
    text = re.sub(r'[0-9０１２３４５６７８９]+', '', text)
    
    # 清理+标准化
    text = clean_illegal_chars(text)
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned_full_text = text
    
    info["full_text_len"] = len(cleaned_full_text)
    
    # 分词
    word_list = jieba.lcut(cleaned_full_text)
    
    # 过滤停用词和单字
    seg_words = []
    for word in word_list:
        word = word.strip().lower()
        if not word:
            continue
        if word in stopwords:
            continue
        if len(word) == 1 and re.match(r'[\u4e00-\u9fa5]', word):
            continue
        seg_words.append(word)
    
    cleaned_seg_text = " ".join(seg_words)
    info["seg_word_count"] = len(seg_words)
    
    info["is_valid"] = (
        len(cleaned_full_text) >= MIN_FULL_TEXT_LEN and
        len(seg_words) >= MIN_SEG_WORD_COUNT
    )
    
    return cleaned_full_text, cleaned_seg_text, info


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 数据清洗程序启动")
    print("-" * 60)
    
    # 确保日志目录存在（用于保存质量报告）
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载资源
    stopwords = load_stopwords(STOPWORDS_FILE)
    keywords = load_keywords(KEYWORDS_FILE)
    
    # 2. 读取数据
    if not INPUT_FILE.exists():
        print(f"❌ 输入文件不存在: {INPUT_FILE}")
        sys.exit(1)
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    if not isinstance(raw_data, list):
        print(f"❌ 数据格式错误: 期望list")
        sys.exit(1)
    
    print(f"📂 读取数据: {len(raw_data)} 条")
    
    # 3. 清洗处理
    processed = []
    valid_count = 0
    invalid_count = 0
    quality_stats = {
        "empty_title": 0,
        "empty_desc": 0,
        "short_text": 0,
        "few_words": 0,
        "valid": 0
    }
    
    for note in raw_data:
        note_id = note.get("note_id", "")
        title = note.get("title", "")
        desc = note.get("desc", "")
        
        full_text, seg_text, info = clean_text(title, desc, stopwords)
        
        if not info["has_title"]:
            quality_stats["empty_title"] += 1
        if not info["has_desc"]:
            quality_stats["empty_desc"] += 1
        if not full_text or len(full_text) < MIN_FULL_TEXT_LEN:
            quality_stats["short_text"] += 1
        if info["seg_word_count"] < MIN_SEG_WORD_COUNT:
            quality_stats["few_words"] += 1
        
        record = {
            "note_id": note_id,
            "original_title": clean_illegal_chars(title),
            "original_desc": clean_illegal_chars(desc),
            "cleaned_full_text": full_text,
            "cleaned_seg_text": seg_text,
            "quality": info
        }
        
        if info["is_valid"]:
            valid_count += 1
            quality_stats["valid"] += 1
        else:
            invalid_count += 1
        
        processed.append(record)
    
    # 4. 统计
    print(f"\n📊 清洗统计:")
    print(f"   总输入: {len(raw_data)}")
    print(f"   有效样本: {valid_count} ({valid_count/len(raw_data)*100:.1f}%)")
    print(f"   无效样本: {invalid_count}")
    
    # 5. 过滤有效数据
    df = pd.DataFrame(processed)
    df_valid = df[df["quality"].apply(lambda x: x.get("is_valid", False))].reset_index(drop=True)
    
    print(f"\n📊 过滤后有效样本: {len(df_valid)}")
    
    # 6. 输出文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 主训练数据
    train_cols = ["note_id", "cleaned_full_text", "cleaned_seg_text"]
    df_train = df_valid[train_cols]
    train_file = OUTPUT_DIR / "bertopic_train_data.csv"
    df_train.to_csv(train_file, index=False, encoding="utf-8-sig")
    print(f"✅ 主训练数据: {train_file}")
    
    # 完整JSON
    json_file = OUTPUT_DIR / "bertopic_cleaned_data.json"
    df_valid.to_json(json_file, orient="records", lines=True, force_ascii=False, indent=2)
    print(f"✅ 完整JSON: {json_file}")
    
    # 7. 质量报告（保存到logs目录）
    quality_file = LOG_DIR / "clean_quality_report.json"
    quality_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "keep_english": KEEP_ENGLISH,
            "min_full_text_len": MIN_FULL_TEXT_LEN,
            "min_seg_word_count": MIN_SEG_WORD_COUNT,
            "stopwords_count": len(stopwords),
            "keywords_count": len(keywords)
        },
        "stats": {
            "total_input": len(raw_data),
            "valid_samples": valid_count,
            "invalid_samples": invalid_count,
            "valid_ratio": valid_count / len(raw_data) if raw_data else 0
        },
        "quality_distribution": quality_stats
    }
    with open(quality_file, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)
    print(f"✅ 质量报告: {quality_file}")
    
    # 人类可读报告（保存到logs目录）
    report_file = LOG_DIR / "data_cleaning_report.txt"
    report = f"""
小红书AI笔记数据清洗报告
{'='*50}

【一、基本信息】
清洗时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
输入文件: {INPUT_FILE}
数据目录: {OUTPUT_DIR}

【二、配置参数】
保留英文: {KEEP_ENGLISH}
最小文本长度: {MIN_FULL_TEXT_LEN} 字符
最小分词数量: {MIN_SEG_WORD_COUNT} 个
停用词数量: {len(stopwords)}
关键词数量: {len(keywords)}

【三、数据漏斗】
原始数据: {len(raw_data)} 条
有效样本: {valid_count} 条 ({valid_count/len(raw_data)*100:.1f}%)
无效样本: {invalid_count} 条

【四、输出文件】
主训练数据: {train_file.name}
完整JSON: {json_file.name}
质量报告: {quality_file.name}
{'='*50}
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ 清洗报告: {report_file}")
    
    print("🎉 清洗完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()