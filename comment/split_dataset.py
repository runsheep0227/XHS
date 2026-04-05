import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# 定义有效情感标签（仅保留-1/0/1）
VALID_LABELS = [-1, 0, 1]


def oversample_train_negatives(
    df_train: pd.DataFrame, multiplier: float, seed: int = 42
) -> pd.DataFrame:
    """
    仅在训练集中对 label==-1 过采样，使负向条数约为原来的 multiplier 倍
    （通过有放回抽样追加行）。验证/测试集分布不变，便于诚实评估。
    """
    if multiplier <= 1.0:
        return df_train
    neg_mask = df_train["label"] == -1
    neg = df_train.loc[neg_mask]
    if len(neg) == 0:
        return df_train
    extra_n = int(round(len(neg) * (multiplier - 1.0)))
    if extra_n <= 0:
        return df_train
    extra = neg.sample(n=extra_n, replace=True, random_state=seed)
    out = pd.concat([df_train, extra], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="按 8:1:1 分层划分 train/val/test；可选仅对训练集负样本过采样。"
    )
    p.add_argument(
        "--train_neg_multiplier",
        type=float,
        default=1.0,
        help=(
            "训练集中负向(label=-1)过采样倍数，默认 1.0 不放大。"
            "例如 1.5 表示在原有负样本基础上约再多 50%% 条(有放回)。"
            "val/test 仍为分层划分，比例与全库一致。"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="划分与过采样随机种子",
    )
    return p.parse_args()


def main():
    args = parse_args()
    mult = float(args.train_neg_multiplier)
    if mult < 1.0:
        print("❌ --train_neg_multiplier 须 >= 1.0")
        return

    print("正在为您执行数据划分 (8:1:1 分层抽样)...")
    if mult > 1.0:
        print(f"   （训练集负样本过采样倍数: {mult}，验证/测试集不变）")
    
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

    # 2. 读取大模型标注好的数据（与写入时一致使用 utf-8-sig）
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    total_raw = len(df)
    print(f"✅ 成功读取标注数据，共计 {total_raw} 条。列：{list(df.columns)}")

    if "label" not in df.columns:
        print("❌ CSV 中缺少 label 列。")
        return

    # 统一为数值，避免 Excel/手工保存后 label 变成字符串导致 isin 匹配不到、整表被清空
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    
    # ========== 仅保留有效情感标签(-1/0/1)，剔除所有无效标签 ==========
    # 第一步：筛选有效标签（-1/0/1）
    df_valid_label = df[df["label"].isin(VALID_LABELS)].copy()
    invalid_label_count = total_raw - len(df_valid_label)
    
    # 第二步：剔除文本为空/空白的无效数据
    df_clean = df_valid_label[
        (df_valid_label['content'].notna()) & 
        (df_valid_label['content'].str.strip() != '')
    ].copy()
    empty_content_count = len(df_valid_label) - len(df_clean)
    total_invalid = invalid_label_count + empty_content_count

    # 标签列在后续分层与保存中用 int，避免写出 1.0
    df_clean["label"] = df_clean["label"].astype(int)
    
    # 打印清理报告（始终输出，便于核对「详细」流失原因）
    print("\n🧹 数据清理报告：")
    print(f"   - 原始行数：{total_raw}")
    print(f"   - 剔除无效/空标签（非 -1/0/1、无法解析为空）：{invalid_label_count} 条")
    print(f"   - 剔除空文本 content：{empty_content_count} 条")
    print(f"   - 总计清理：{total_invalid} 条 | 有效数据剩余：{len(df_clean)} 条")
    if total_invalid == 0:
        print("   （无需清理，全部有效）")
    
    # 检查清理后是否有数据
    if len(df_clean) == 0:
        print("❌ 清理后无有效数据！请检查标注结果中的 label 是否为数字 -1/0/1（Excel 保存后常为文本，本脚本已尝试自动转数字）。")
        return

    # ========== 检查有效标签的样本数，处理分层抽样小样本问题 ==========
    # 统计有效标签的样本数
    label_counts = df_clean['label'].value_counts().sort_index()
    print("\n📊 有效情感标签分布（仅-1/0/1）：")
    for label in VALID_LABELS:
        count = label_counts.get(label, 0)
        print(f"   标签 {label}: {count} 条")
    
    # 找出样本数 < 2 的有效标签（无法分层抽样）
    small_sample_labels = [label for label in VALID_LABELS if label_counts.get(label, 0) < 2]
    stratify_flag = True  # 是否启用分层抽样
    
    if small_sample_labels:
        print(f"\n⚠️ 发现小样本情感标签（样本数<2）：{small_sample_labels}")
        print("   解决方案：关闭分层抽样，改用随机拆分（保证数据不丢失）")
        stratify_flag = False

    # 4–7. 直接按「整行」划分，保留 raw_response 等所有列（此前只取 note_id+content 会丢掉详细字段）
    stratify_arg = df_clean["label"] if stratify_flag else None
    df_train, df_temp = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify_arg,
    )

    temp_label_counts = df_temp["label"].value_counts()
    temp_small_sample = [label for label in VALID_LABELS if temp_label_counts.get(label, 0) < 2]
    temp_stratify = df_temp["label"] if (not temp_small_sample) and stratify_flag else None

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=args.seed,
        stratify=temp_stratify,
    )

    train_before = len(df_train)
    neg_before = (df_train["label"] == -1).sum()
    df_train = oversample_train_negatives(df_train, mult, seed=args.seed)
    if mult > 1.0:
        neg_after = (df_train["label"] == -1).sum()
        print(
            f"\n📌 训练集负向过采样: {train_before} → {len(df_train)} 条 "
            f"（负向 {neg_before} → {neg_after} 条）"
        )

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