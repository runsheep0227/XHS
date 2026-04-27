import os
import shutil
from datetime import datetime

# ================= 配置区域 =================
# 源文件夹路径
SOURCE_DIR = r"E:\document\PG\rawdata\MediaCrawler\data\xhs\json"

# 目标文件夹路径
TARGET_COMMENT_DIR = r"E:\document\PG\studio\comment\rawdata"
TARGET_CONTENT_DIR = r"E:\document\PG\studio\content\rawdata"

# 文件类型关键词 (根据文件名判断)
KEYWORD_COMMENT = "comments"
KEYWORD_CONTENT = "contents"

# ===========================================

def classify_and_move_files():
    # 1. 检查源目录是否存在
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 错误：源目录不存在 -> {SOURCE_DIR}")
        return

    # 2. 确保目标目录存在 (不存在则创建)
    os.makedirs(TARGET_COMMENT_DIR, exist_ok=True)
    os.makedirs(TARGET_CONTENT_DIR, exist_ok=True)
    print("✅ 目标目录检查完毕。")
    print("-" * 60)

    # 3. 获取源目录下所有文件
    files = os.listdir(SOURCE_DIR)
    json_files = [f for f in files if f.endswith('.json')]
    
    if not json_files:
        print("⚠️  源目录下没有找到 .json 文件。")
        return

    print(f"🔍 发现 {len(json_files)} 个 JSON 文件，开始处理...\n")

    moved_count = 0
    replaced_count = 0
    error_count = 0

    for filename in json_files:
        source_path = os.path.join(SOURCE_DIR, filename)
        target_path = ""
        file_type = ""

        # 4. 判断文件类型并确定目标路径
        if KEYWORD_COMMENT in filename:
            target_path = os.path.join(TARGET_COMMENT_DIR, filename)
            file_type = "评论 (comments)"
        elif KEYWORD_CONTENT in filename:
            target_path = os.path.join(TARGET_CONTENT_DIR, filename)
            file_type = "内容 (contents)"
        else:
            # 如果都不包含，跳过或按需处理，这里选择跳过
            print(f"⏭️  跳过未知类型文件：{filename}")
            continue

        try:
            # 5. 检查是否已存在（实现替换逻辑）
            if os.path.exists(target_path):
                # 如果存在，先删除旧文件
                os.remove(target_path)# Path 对象的删除方法
                print(f"🔄 [覆盖] {file_type}: {filename}")
                replaced_count += 1
            else:
                print(f"📄 [复制] {file_type}: {filename}")
            
            # 6. 执行复制 (shutil.copy2 支持 Path 对象)
            shutil.copy2(source_path, target_path)
            moved_count += 1

        except Exception as e:
            print(f"💥 [错误] 处理文件 {filename} 时失败：{e}")
            error_count += 1

    # 7. 输出总结
    print("-" * 60)
    print(f"🎉 处理完成！")
    print(f"   ✅ 成功复制/更新：{moved_count} 个文件")
    print(f"   🔄 覆盖旧文件：{replaced_count} 个")
    print(f"   ❌ 失败文件数：{error_count} 个")
    print(f"   📂 评论存放于：{TARGET_COMMENT_DIR}")
    print(f"   📂 内容存放于：{TARGET_CONTENT_DIR}")

if __name__ == "__main__":
    # 记录开始时间
    start_time = datetime.now()
    print(f"⏰ 任务开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    classify_and_move_files()
    
    # 记录结束时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print("=" * 60)
    print(f"⏱️  任务耗时：{duration:.2f} 秒")