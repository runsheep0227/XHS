import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义有效情感标签（仅保留-1/0/1）
VALID_LABELS = [-1, 0, 1]

def main():
    print("正在为您执行数据划分 (8:1:1 分层抽样)...")
    
    # 路径获取：动态获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义输入和输出路径
    input_file = os.path.join(current_dir, 'bert_data', 'llm_labeled_result.csv')
    train_file = os.path.join(current_dir, 'bert_data', 'train.csv')
    val_file = os.path.join(current_dir, 'bert_data', 'val.csv')
    test_file = os.path.join(current_dir, 'bert_data', 'test.csv')
    
    # 1. 检查标注数据是否存在
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件：{input_file}\n，请确保您已经成功运行了 llm_annotate.py。")
        return

    # 2. 读取大模型标注好的数据
    df = pd.read_csv(input_file)
    total_raw = len(df)
    print(f"✅ 成功读取标注数据，共计 {total_raw} 条。")
    
    # ========== 核心修正1：仅保留有效情感标签(-1/0/1)，剔除所有无效标签 ==========
    # 第一步：筛选有效标签（-1/0/1）
    df_valid_label = df[df['label'].isin(VALID_LABELS)].copy()
    invalid_label_count = total_raw - len(df_valid_label)
    
    # 第二步：剔除文本为空/空白的无效数据
    df_clean = df_valid_label[
        (df_valid_label['content'].notna()) & 
        (df_valid_label['content'].str.strip() != '')
    ].copy()
    empty_content_count = len(df_valid_label) - len(df_clean)
    total_invalid = invalid_label_count + empty_content_count
    
    # 打印清理报告（精准说明清理类型）
    if total_invalid > 0:
        print(f"🧹 数据清理报告：")
        print(f"   - 剔除无效标签（2/-2等）：{invalid_label_count} 条")
        print(f"   - 剔除空文本数据：{empty_content_count} 条")
        print(f"   - 总计清理：{total_invalid} 条 | 有效数据剩余：{len(df_clean)} 条")
    
    # 检查清理后是否有数据
    if len(df_clean) == 0:
        print("❌ 清理后无有效数据！请检查标注结果中的label列是否包含-1/0/1。")
        return

    # ========== 核心修正2：检查有效标签的样本数，处理分层抽样小样本问题 ==========
    # 统计有效标签的样本数
    label_counts = df_clean['label'].value_counts().sort_index()
    print("\n📊 有效情感标签分布（仅-1/0/1）：")
    for label in VALID_LABELS:
        count = label_counts.get(label, 0)
        print(f"   标签 {label}: {count} 条")
    
    # 找出样本数 < 2 的有效标签（无法分层抽样）
    small_sample_labels = [label for label in VALID_LABELS if label_counts.get(label, 0) < 2]
    stratify_flag = True  # 是否启用分层抽样
    stratify_y = df_clean['label']
    
    if small_sample_labels:
        print(f"\n⚠️ 发现小样本情感标签（样本数<2）：{small_sample_labels}")
        print("   解决方案：关闭分层抽样，改用随机拆分（保证数据不丢失）")
        stratify_flag = False
        stratify_y = None

    # 4. 提取特征和标签
    X = df_clean[['note_id', 'content']]
    y = df_clean['label']

    # 5. 第一次划分：训练集(80%) 和 临时集(20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=stratify_y if stratify_flag else None  # 有小样本则关闭分层
    )
    
    # 6. 第二次划分：验证集(10%) 和 测试集(10%)
    # 临时集分层抽样（仅当临时集标签数都≥2时启用）
    temp_label_counts = y_temp.value_counts()
    temp_small_sample = [label for label in VALID_LABELS if temp_label_counts.get(label, 0) < 2]
    temp_stratify_y = y_temp if (not temp_small_sample) and stratify_flag else None
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_stratify_y
    )

    # 7. 将特征和标签重新拼回 DataFrame
    df_train = pd.concat([X_train, y_train.rename('label')], axis=1)
    df_val = pd.concat([X_val, y_val.rename('label')], axis=1)
    df_test = pd.concat([X_test, y_test.rename('label')], axis=1)

    # 8. 保存为 CSV（UTF-8带BOM，兼容Excel）
    df_train.to_csv(train_file, index=False, encoding='utf-8-sig')
    df_val.to_csv(val_file, index=False, encoding='utf-8-sig')
    df_test.to_csv(test_file, index=False, encoding='utf-8-sig')

    # 9. 打印切分报告（精准到有效情感标签）
    print("\n🎉 数据划分完成！切分详情如下：")
    total_clean = len(df_clean)
    print(f"📦 训练集 (train.csv) : {len(df_train)} 条 ({len(df_train)/total_clean*100:.1f}%)")
    print(f"📦 验证集 (val.csv)   : {len(df_val)} 条 ({len(df_val)/total_clean*100:.1f}%)")
    print(f"📦 测试集 (test.csv)  : {len(df_test)} 条 ({len(df_test)/total_clean*100:.1f}%)")
    
    # 打印各数据集有效标签分布
    print("\n📊 训练集有效情感标签分布：")
    train_label_counts = df_train['label'].value_counts().sort_index()
    for label in VALID_LABELS:
        count = train_label_counts.get(label, 0)
        print(f"  标签 {label}: {count} 条")
    
    print("\n📊 验证集有效情感标签分布：")
    val_label_counts = df_val['label'].value_counts().sort_index()
    for label in VALID_LABELS:
        count = val_label_counts.get(label, 0)
        print(f"  标签 {label}: {count} 条")
    
    print("\n📊 测试集有效情感标签分布：")
    test_label_counts = df_test['label'].value_counts().sort_index()
    for label in VALID_LABELS:
        count = test_label_counts.get(label, 0)
        print(f"  标签 {label}: {count} 条")
        
    print("\n💡 数据集已准备就绪！仅包含-1/0/1情感标签，可启动train_roberta.py微调！")

if __name__ == "__main__":
    main()