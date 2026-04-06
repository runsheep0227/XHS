import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter
import logging

# ==================== 配置 ====================
WORK_DIR = Path(__file__).resolve().parent
INPUT_DIR = WORK_DIR / "rawdata"
OUTPUT_FILE = WORK_DIR / "cleaned_desc.json"
LOG_DIR = WORK_DIR / "logs"

# 必填字段
REQUIRED_FIELDS = ["note_id", "title", "desc"]

# 字段类型定义
FIELD_TYPES = {
    "note_id": str,
    "title": str,
    "desc": str
}

# ==================== 日志配置 ====================
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "clean_desc.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


# ==================== 核心函数 ====================

def validate_note(note: Dict[str, Any], note_index: int) -> tuple[bool, List[str]]:
    """
    校验单条笔记数据
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # 1. 检查必填字段
    for field in REQUIRED_FIELDS:
        if field not in note:
            errors.append(f"缺少必填字段: {field}")
            continue
        
        value = note[field]
        
        # 2. 类型校验
        if not isinstance(value, FIELD_TYPES[field]):
            errors.append(f"字段 {field} 类型错误: 期望 {FIELD_TYPES[field]}, 实际 {type(value)}")
            continue
        
        # 3. note_id校验
        if field == "note_id":
            if not value or str(value).strip() == "":
                errors.append(f"note_id为空或无效: {value}")
            elif len(str(value)) < 10:
                errors.append(f"note_id长度异常: {value}")
        
        # 4. title校验
        elif field == "title":
            if not value or str(value).strip() == "":
                errors.append(f"title为空")
        
        # 5. desc校验
        elif field == "desc":
            if value is None:
                errors.append(f"desc为None")
    
    return len(errors) == 0, errors


def extract_fields(note: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """提取并标准化字段"""
    return {
        "note_id": str(note.get("note_id", "")).strip(),
        "title": str(note.get("title", "")).strip() if note.get("title") else "",
        "desc": str(note.get("desc", "")).strip() if note.get("desc") else ""
    }


def process_file(file_path: Path) -> tuple[List[Dict], Dict]:
    """处理单个JSON文件"""
    valid_notes = []
    error_report = {
        "file": file_path.name,
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": []
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        error_report["errors"].append(f"JSON解析失败: {str(e)}")
        logger.error(f"❌ 文件 {file_path.name} JSON解析失败: {e}")
        return [], error_report
    except Exception as e:
        error_report["errors"].append(f"文件读取失败: {str(e)}")
        logger.error(f"❌ 文件 {file_path.name} 读取失败: {e}")
        return [], error_report
    
    if not isinstance(data, list):
        error_report["errors"].append(f"数据类型错误: 期望list, 实际{type(data)}")
        logger.error(f"❌ 文件 {file_path.name} 数据类型错误")
        return [], error_report
    
    error_report["total"] = len(data)
    
    for idx, note in enumerate(data):
        is_valid, errors = validate_note(note, idx)
        
        if is_valid:
            extracted = extract_fields(note)
            if extracted["title"] or extracted["desc"]:
                valid_notes.append(extracted)
                error_report["valid"] += 1
            else:
                error_report["invalid"] += 1
                error_report["errors"].append(f"笔记{idx}: title和desc均为空")
        else:
            error_report["invalid"] += 1
            for err in errors:
                error_report["errors"].append(f"笔记{idx}: {err}")
    
    return valid_notes, error_report


def check_duplicate_ids(notes: List[Dict]) -> Dict:
    """检查note_id重复情况"""
    note_ids = [n["note_id"] for n in notes]
    counter = Counter(note_ids)
    duplicates = {k: v for k, v in counter.items() if v > 1}
    
    return {
        "unique_count": len(set(note_ids)),
        "total_count": len(note_ids),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates
    }


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 数据字段抽取程序启动")
    print("-" * 60)
    
    if not INPUT_DIR.exists():
        logger.error(f"❌ 输入目录不存在: {INPUT_DIR}")
        sys.exit(1)
    
    json_files = list(INPUT_DIR.glob("*.json"))
    if not json_files:
        logger.error(f"❌ 输入目录中没有JSON文件: {INPUT_DIR}")
        sys.exit(1)
    
    print(f"📂 发现 {len(json_files)} 个JSON文件")
    
    all_notes = []
    all_error_reports = []
    
    for file_path in json_files:
        valid_notes, error_report = process_file(file_path)
        all_notes.extend(valid_notes)
        all_error_reports.append(error_report)
    
    total_input = sum(r["total"] for r in all_error_reports)
    total_valid = len(all_notes)
    total_invalid = sum(r["invalid"] for r in all_error_reports)
    
    print(f"\n📊 第一阶段统计:")
    print(f"   原始数据总量: {total_input}")
    print(f"   有效笔记: {total_valid}")
    print(f"   无效笔记: {total_invalid}")
    
    print(f"\n🔍 根据note_id去重，发现重复笔记，将保留第一条")
    dup_info = check_duplicate_ids(all_notes)

    print(f"   重复笔记数量: {dup_info['duplicate_count']}")
        
    # 简洁的去重方式
    seen = {}
    for note in all_notes:
        seen.setdefault(note["note_id"], note)
    all_notes = list(seen.values())
    
    print(f"   去重后笔记数量: {len(all_notes)}")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_notes, f, ensure_ascii=False, indent=2)
    
    # 保存错误报告到logs目录
    error_report_file = LOG_DIR / "extract_error_report.json"
    error_summary = {
        "extract_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": len(json_files),
        "total_input": total_input,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "after_dedup": len(all_notes),
        "duplicate_info": dup_info,
        "file_reports": all_error_reports
    }
    with open(error_report_file, "w", encoding="utf-8") as f:
        json.dump(error_summary, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ 字段抽取完成!")
    print(f"📄 数据输出: {OUTPUT_FILE}")
    print(f"📋 错误报告: {error_report_file}")
    print(f"📊 最终有效笔记: {len(all_notes)} 条")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
