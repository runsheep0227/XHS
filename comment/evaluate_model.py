import json
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ================= 1. 字体与标签配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

LABEL2ID = {-1: 0, 0: 1, 1: 2}
ID2LABEL = {0: -1, 1: 0, 2: 1}
TARGET_NAMES = ["负向情感(-1)", "中性情感(0)", "正向情感(1)"]
NUM_CLASSES = 3


# ================= 2. 数据集类 =================
class TestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ================= 3. 可视化函数 =================

def plot_confusion_matrix(cm, save_path):
    """图1：原始数量混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                annot_kws={"size": 16}, ax=ax)
    ax.set_title('混淆矩阵（样本数量）', fontsize=15, pad=15)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ 混淆矩阵 → {save_path}")


def plot_normalized_confusion_matrix(cm, save_path):
    """图2：归一化混淆矩阵（百分比）"""
    # 按行归一化：每行 = 该真实类别的预测分布
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                annot_kws={"size": 14}, ax=ax,
                vmin=0, vmax=1)
    ax.set_title('归一化混淆矩阵（行方向百分比）', fontsize=15, pad=15)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ 归一化混淆矩阵 → {save_path}")


def plot_metrics_bar(true_labels, pred_labels, save_path):
    """图3：各类别 Precision / Recall / F1 柱状图"""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=[0, 1, 2], zero_division=0
    )

    x = np.arange(NUM_CLASSES)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_p = ax.bar(x - width, precision, width, label='Precision', color='#4C72B0', edgecolor='white')
    bars_r = ax.bar(x, recall, width, label='Recall', color='#55A868', edgecolor='white')
    bars_f = ax.bar(x + width, f1, width, label='F1-Score', color='#C44E52', edgecolor='white')

    # 在柱子顶部标注数值
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_NAMES, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('各类别 Precision / Recall / F1-Score 对比', fontsize=15, pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ 指标柱状图 → {save_path}")


def plot_confidence_distribution(probs, true_labels, pred_labels, save_path):
    """图4：预测置信度分布直方图（正确 vs 错误）"""
    max_probs = np.max(probs, axis=1)
    is_correct = np.array(true_labels) == np.array(pred_labels)

    correct_conf = max_probs[is_correct]
    wrong_conf = max_probs[~is_correct]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 41)  # 40 个区间

    ax.hist(correct_conf, bins=bins, alpha=0.7, label=f'预测正确 ({len(correct_conf)}条)',
            color='#55A868', edgecolor='white')
    ax.hist(wrong_conf, bins=bins, alpha=0.7, label=f'预测错误 ({len(wrong_conf)}条)',
            color='#C44E52', edgecolor='white')

    ax.axvline(np.mean(max_probs), color='black', linestyle='--', linewidth=1,
               label=f'平均置信度: {np.mean(max_probs):.3f}')

    ax.set_xlabel('最大 Softmax 概率（置信度）', fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title('预测置信度分布（正确 vs 错误）', fontsize=15, pad=15)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ 置信度分布 → {save_path}")


def plot_per_class_confidence(probs, true_labels, save_path):
    """图5：每个类别内部的置信度箱线图"""
    max_probs = np.max(probs, axis=1)
    data = []
    for cls_id in range(NUM_CLASSES):
        mask = np.array(true_labels) == cls_id
        data.append(max_probs[mask])

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, labels=TARGET_NAMES, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))

    colors = ['#4C72B0', '#55A868', '#C44E52']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 在每个箱线旁标注均值
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.text(i + 1, mean_val, f'μ={mean_val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('最大 Softmax 概率', fontsize=12)
    ax.set_title('各类别预测置信度分布（箱线图）', fontsize=15, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ 各类别置信度箱线图 → {save_path}")


def plot_roc_curves(true_labels, probs, save_path):
    """图6：多类别 ROC 曲线（One-vs-Rest）"""
    y_true_bin = label_binarize(true_labels, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#4C72B0', '#55A868', '#C44E52']

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f'{TARGET_NAMES[i]} (AUC={roc_auc:.4f})')

    # 微平均
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color='gray', linewidth=2, linestyle='--',
            label=f'微平均 (AUC={roc_auc_micro:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC 曲线（One-vs-Rest）', fontsize=15, pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ ROC 曲线 → {save_path}")


def plot_pr_curves(true_labels, probs, save_path):
    """图7：Precision-Recall 曲线"""
    y_true_bin = label_binarize(true_labels, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#4C72B0', '#55A868', '#C44E52']

    for i in range(NUM_CLASSES):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], probs[:, i])
        ax.plot(rec, prec, color=colors[i], linewidth=2,
                label=f'{TARGET_NAMES[i]} (AP={ap:.4f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall 曲线', fontsize=15, pad=15)
    ax.legend(fontsize=11, loc='lower left')
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.05])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ PR 曲线 → {save_path}")


def plot_metrics_radar(true_labels, pred_labels, save_path):
    """图8：各类别指标雷达图"""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=[0, 1, 2], zero_division=0
    )

    categories = ['Precision', 'Recall', 'F1-Score']
    # 每个类别的三个指标
    values_neg = [precision[0], recall[0], f1[0]]
    values_neu = [precision[1], recall[1], f1[1]]
    values_pos = [precision[2], recall[2], f1[2]]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    colors = ['#4C72B0', '#55A868', '#C44E52']

    for values, name, color in [
        (values_neg, TARGET_NAMES[0], colors[0]),
        (values_neu, TARGET_NAMES[1], colors[1]),
        (values_pos, TARGET_NAMES[2], colors[2]),
    ]:
        vals = values + values[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('各类别指标雷达图', fontsize=15, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 雷达图 → {save_path}")


def _collect_errors(texts, true_labels, pred_labels, probs):
    """逐条误判信息，供 CSV 与 TXT 复用。"""
    pred_class_probs = [float(probs[i][pred_labels[i]]) for i in range(len(pred_labels))]
    errors = []
    for i in range(len(true_labels)):
        if true_labels[i] != pred_labels[i]:
            errors.append({
                "index": i,
                "text": str(texts[i]),
                "true_label": ID2LABEL[true_labels[i]],
                "pred_label": ID2LABEL[pred_labels[i]],
                "confidence": pred_class_probs[i],
                "prob_neg": float(probs[i][0]),
                "prob_neu": float(probs[i][1]),
                "prob_pos": float(probs[i][2]),
            })
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    return errors


def save_misclassified_csv(df_test, errors, save_path):
    """全部误判样本写入 CSV（完整正文，便于 Excel 筛选）。"""
    rows = []
    for err in errors:
        i = err["index"]
        row = {
            "sample_index": i,
            "true_label": err["true_label"],
            "pred_label": err["pred_label"],
            "confidence": round(err["confidence"], 6),
            "prob_neg": round(err["prob_neg"], 6),
            "prob_neu": round(err["prob_neu"], 6),
            "prob_pos": round(err["prob_pos"], 6),
            "content": err["text"],
        }
        if "note_id" in df_test.columns:
            row["note_id"] = df_test.iloc[i].get("note_id", "")
        rows.append(row)
    out = pd.DataFrame(rows)
    # note_id 放前面更顺眼
    cols = [c for c in ["note_id", "sample_index", "true_label", "pred_label", "confidence",
                        "prob_neg", "prob_neu", "prob_pos", "content"] if c in out.columns]
    out = out[cols]
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"   ✅ 误判样本 CSV（完整内容）→ {save_path}  ({len(out)} 条)")


def save_error_analysis(texts, true_labels, pred_labels, probs, save_path, df_test=None):
    """
    文本版错误分析：默认写入全部误判的完整正文（按预测置信度从高到低）。
    若误判过多（>400 条），仅写入前 400 条正文，并提示查看 CSV。
    """
    errors = _collect_errors(texts, true_labels, pred_labels, probs)
    max_lines_in_txt = 400
    truncated = len(errors) > max_lines_in_txt
    errors_for_txt = errors[:max_lines_in_txt] if truncated else errors

    lines = []
    lines.append("=" * 70)
    lines.append("  错误分析报告（预测错误样本，完整正文；按「预测置信度」从高到低）")
    lines.append("=" * 70)
    lines.append(f"  总测试样本: {len(true_labels)} 条")
    lines.append(f"  预测错误:   {len(errors)} 条")
    lines.append(f"  错误率:     {len(errors) / max(len(true_labels), 1) * 100:.2f}%")
    if df_test is not None and "note_id" in df_test.columns:
        lines.append("  （每条下的 note_id 与 test.csv 行对应 sample_index 列）")
    lines.append("=" * 70)
    if truncated:
        lines.append(f"  【说明】误判超过 {max_lines_in_txt} 条，本文仅列出前 {max_lines_in_txt} 条完整正文；")
        lines.append("          全部误判请打开同目录 misclassified_test.csv")
        lines.append("=" * 70)

    for rank, err in enumerate(errors_for_txt, 1):
        lines.append(f"\n{'─' * 70}")
        lines.append(f"【误判 #{rank}】sample_index={err['index']}")
        if df_test is not None and "note_id" in df_test.columns:
            nid = df_test.iloc[err["index"]].get("note_id", "")
            lines.append(f"note_id: {nid}")
        lines.append(f"真实标签: {err['true_label']}  →  模型预测: {err['pred_label']}")
        lines.append(f"预测置信度(对预测类): {err['confidence']:.4f}")
        lines.append(
            f"P(负)={err['prob_neg']:.4f}  P(中)={err['prob_neu']:.4f}  P(正)={err['prob_pos']:.4f}"
        )
        lines.append("─ 评论正文（完整）")
        lines.append(err["text"])

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"   ✅ 错误分析报告（TXT）→ {save_path}")
    return errors


# ================= 4. 主程序 =================
def main():
    print("📊 3分类情感模型 - 全面评估程序启动")
    print("=" * 60)

    # --- 路径 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, 'bert_data', 'test.csv')
    model_dir = os.path.join(current_dir, 'saved_model')
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(model_dir):
        print(f"❌ 找不到模型: {model_dir}")
        return

    # --- 加载数据 ---
    print("\n📁 加载测试集...")
    df_test = pd.read_csv(test_file, encoding="utf-8-sig")
    df_test = df_test.dropna(subset=["content", "label"])
    df_test["label"] = pd.to_numeric(df_test["label"], errors="coerce")
    df_test = df_test[df_test["label"].isin([-1, 0, 1])].copy()
    df_test["label"] = df_test["label"].astype(int)
    df_test = df_test.reset_index(drop=True)
    texts = df_test["content"].astype(str).tolist()
    true_labels = [LABEL2ID[int(x)] for x in df_test["label"].tolist()]
    print(f"   有效测试数据: {len(texts)} 条")

    # --- 加载模型 ---
    print("\n🧠 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # --- 推理（收集概率） ---
    test_dataset = TestDataset(texts, true_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_probs = []
    pred_labels = []

    print("\n🔍 推理中...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="   进度"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            pred_labels.extend(preds)

    all_probs = np.array(all_probs)
    pred_labels = np.array(pred_labels)

    # --- 基础指标 ---
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    class_report = classification_report(
        true_labels, pred_labels, target_names=TARGET_NAMES, digits=4
    )

    print(f"\n{'=' * 60}")
    print(f"  🎯 Accuracy:  {acc:.4f}")
    print(f"  🏆 Macro-F1:  {macro_f1:.4f}")
    print(f"\n{class_report}")
    print(f"{'=' * 60}")

    # --- 保存文字报告 ---
    report_file = os.path.join(results_dir, 'evaluation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  AIGC 情感分析模型(3分类) - 终极测试报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Macro-F1:  {macro_f1:.4f}\n\n")
        f.write(f"{class_report}\n")
    print(f"\n📄 文字报告 → {report_file}")

    # ================= 5. 生成全部可视化 =================
    print("\n🎨 生成可视化图表...")
    cm = confusion_matrix(true_labels, pred_labels)

    plot_confusion_matrix(cm, os.path.join(results_dir, '01_confusion_matrix.png'))
    plot_normalized_confusion_matrix(cm, os.path.join(results_dir, '02_confusion_matrix_normalized.png'))
    plot_metrics_bar(true_labels, pred_labels, os.path.join(results_dir, '03_metrics_bar_chart.png'))
    plot_confidence_distribution(all_probs, true_labels, pred_labels,
                                 os.path.join(results_dir, '04_confidence_distribution.png'))
    plot_per_class_confidence(all_probs, true_labels,
                              os.path.join(results_dir, '05_per_class_confidence.png'))
    plot_roc_curves(true_labels, all_probs, os.path.join(results_dir, '06_roc_curves.png'))
    plot_pr_curves(true_labels, all_probs, os.path.join(results_dir, '07_pr_curves.png'))
    plot_metrics_radar(true_labels, pred_labels, os.path.join(results_dir, '08_metrics_radar.png'))

    viz_json = os.path.join(results_dir, "eval_viz_payload.json")
    with open(viz_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "true_labels": [int(x) for x in true_labels],
                "pred_labels": [int(x) for x in pred_labels.tolist()],
                "probs": all_probs.tolist(),
            },
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    print(f"   ✅ 前端可视化载荷 → {viz_json}")

    mis_csv = os.path.join(results_dir, "misclassified_test.csv")
    errors = save_error_analysis(
        texts,
        true_labels,
        pred_labels,
        all_probs,
        os.path.join(results_dir, "09_error_analysis.txt"),
        df_test=df_test,
    )
    save_misclassified_csv(df_test, errors, mis_csv)

    # --- 最终汇总 ---
    print(f"\n{'=' * 60}")
    print("🎉 评估全部完成！生成文件清单：")
    print(f"{'=' * 60}")
    output_files = [
        ('01_confusion_matrix.png',          '原始数量混淆矩阵'),
        ('02_confusion_matrix_normalized.png','归一化混淆矩阵（百分比）'),
        ('03_metrics_bar_chart.png',         '各类别 P/R/F1 柱状图'),
        ('04_confidence_distribution.png',    '置信度分布直方图'),
        ('05_per_class_confidence.png',       '各类别置信度箱线图'),
        ('06_roc_curves.png',                 'ROC 曲线（One-vs-Rest）'),
        ('07_pr_curves.png',                  'Precision-Recall 曲线'),
        ('08_metrics_radar.png',              '各类别指标雷达图'),
        ('eval_viz_payload.json',             '前端复现图表用（标签+softmax，与 PNG 同源）'),
        ('09_error_analysis.txt',             '误判明细 TXT（完整正文，最多400条）'),
        ('misclassified_test.csv',            '误判样本 CSV（全部误判+完整 content）'),
        ('evaluation_report.txt',             '文字评估报告'),
    ]
    for fname, desc in output_files:
        print(f"   📊 {fname}  ← {desc}")

    print(f"\n💡 所有文件已保存至: {results_dir}/")


if __name__ == "__main__":
    main()
