import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("正在为您执行数据划分 (8:1:1 分层抽样)...")
    
    # 路径获取：动态获取当前脚本所在目录的绝对路径 (即 .../comment/)
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
    
    # 3. 数据清洗：剔除 API 失败的数据 (label == -1) 或文本为空的数据
    df_clean = df[(df['label'] != -1) & (df['content'].notna())].copy()
    total_clean = len(df_clean)
    
    if total_clean < total_raw:
        print(f"🧹 已自动为您清理了 {total_raw - total_clean} 条无效/失败数据。有效数据剩余：{total_clean} 条。")

    # 4. 提取特征和标签
    X = df_clean[['note_id', 'content']]
    y = df_clean['label']

    # 5. 第一次划分：分离出 训练集(80%) 和 临时集(20%)
    # stratify=y 是核心！保证各个情感标签的比例在切分后完全一致
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. 第二次划分：将临时集(20%) 对半分，得到 验证集(10%) 和 测试集(10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 7. 将特征和标签重新拼回 DataFrame 准备保存
    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # 8. 保存为 CSV，供后续 RoBERTa 模型读取
    df_train.to_csv(train_file, index=False, encoding='utf-8-sig')
    df_val.to_csv(val_file, index=False, encoding='utf-8-sig')
    df_test.to_csv(test_file, index=False, encoding='utf-8-sig')

    # 9. 打印切分报告
    print("\n🎉 数据划分圆满完成！切分详情如下：")
    print(f"📦 训练集 (train.csv) : {len(df_train)} 条 ({len(df_train)/total_clean*100:.1f}%)")
    print(f"📦 验证集 (val.csv)   : {len(df_val)} 条 ({len(df_val)/total_clean*100:.1f}%)")
    print(f"📦 测试集 (test.csv)  : {len(df_test)} 条 ({len(df_test)/total_clean*100:.1f}%)")
    
    print("\n📊 各情感标签在训练集中的分布情况：")
    label_counts = df_train['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  标签 {label}: {count} 条")
        
    print("\n💡 下一步：您的数据集已准备就绪！启动train_roberta.py运行RTX 5070 显卡进行模型微调啦！")

if __name__ == "__main__":
    main()