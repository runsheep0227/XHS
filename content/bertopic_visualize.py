import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

# 直接指定你的微软雅黑字体文件路径
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"  # 改成你实际的路径

# 注册字体
fm.fontManager.addfont(FONT_PATH)
font_prop = fm.FontProperties(fname=FONT_PATH)
font_name = font_prop.get_name()

# 全局生效
matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

print(f"强制绑定字体: {font_name} -> {FONT_PATH}")
# ====== 之后再 import 其他库 ======

import pandas as pd
import numpy as np
import os
import sys
import torch
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings("ignore")

# ==================== 环境配置 ====================
os.environ['NUMBA_CACHE_DIR'] = str(Path(os.environ.get('TEMP', '/tmp')) / 'numba_cache')
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 可视化库 ====================
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，防止弹窗卡死
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ==================== 算法库 ====================
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity

# ==================== 路径配置 ====================
WORK_DIR = Path(r"E:\document\PG\studio\content")
DATA_DIR = WORK_DIR / "bertopic_ready_data"
MODEL_DIR = WORK_DIR / "bertopic_results_optimized" / "saved_model"
RESULTS_CSV = WORK_DIR / "bertopic_results_optimized" / "final_pro_topics.csv"
VIS_OUTPUT_DIR = WORK_DIR / "bertopic_visualizations"

INPUT_FILE = DATA_DIR / "bertopic_train_data.csv"

# ==================== 宏观主题定义（与训练脚本保持一致） ====================
MACRO_ANCHORS = {
    "AI内容创作": {
        "color": "#E74C3C",
        "emoji": "🎨"
    },
    "AI应用与测评": {
        "color": "#3498DB",
        "emoji": "📱"
    },
    "AI学习教程": {
        "color": "#2ECC71",
        "emoji": "📚"
    },
    "AI赋能工作生活": {
        "color": "#F39C12",
        "emoji": "💼"
    },
    "AI社会反思": {
        "color": "#9B59B6",
        "emoji": "🤔"
    }
}

# ==================== 中文字体配置 ====================
def setup_chinese_font():
    """自动寻找系统中可用的中文字体"""
    import matplotlib.font_manager as fm

    # 常见中文字体名称
    candidates = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi',
        'FangSong', 'STHeiti', 'STSong', 'STKaiti',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Noto Sans CJK SC', 'Noto Serif CJK SC',
        'Source Han Sans SC', 'Source Han Serif SC',
        'AR PL UMing CN', 'AR PL UKai CN',
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  ✅ 使用中文字体: {font}")
            return font

    # 兜底：尝试手动加载
    font_paths = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            font_prop = fm.FontProperties(fname=fp)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  ✅ 手动加载字体: {fp}")
            return font_prop.get_name()

    print("  ⚠️ 未找到中文字体，图表中文可能显示为方块")
    return None


# ==================== 日志 ====================
VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==================== 数据加载 ====================
def load_all():
    """加载模型、结果CSV和原始数据"""
    logger.info("📂 加载训练结果数据...")
    df_result = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")

    logger.info("📂 加载原始训练数据...")
    df_raw = pd.read_csv(INPUT_FILE).dropna(subset=["cleaned_full_text", "cleaned_seg_text"])

    logger.info("📂 加载已保存的 BERTopic 模型...")
    topic_model = BERTopic.load(str(MODEL_DIR))

    return topic_model, df_result, df_raw


# ==================== 可视化模块 ====================

# -------------------------------------------------------
# 图1: 宏观主题分布饼图 + 数据覆盖率环图（组合图）
# -------------------------------------------------------
def plot_macro_distribution(df_result):
    """宏观主题分布与数据覆盖率"""
    logger.info("📊 绘制图1: 宏观主题分布...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 左图：宏观主题分布饼图 ---
    ax1 = axes[0]
    macro_counts = df_result['macro_topic_name'].value_counts()

    # 分离噪声与有效数据
    noise_count = macro_counts.get("噪声数据(Outliers)", 0)
    valid_counts = macro_counts.drop("噪声数据(Outliers)", errors='ignore')

    colors = []
    labels = []
    sizes = []
    for name, count in valid_counts.items():
        colors.append(MACRO_ANCHORS.get(name, {}).get("color", "#95A5A6"))
        labels.append(name)
        sizes.append(count)

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    for t in texts:
        t.set_fontsize(11)

    ax1.set_title("宏观主题分布", fontsize=16, fontweight='bold', pad=20)

    # --- 右图：覆盖率环图 ---
    ax2 = axes[1]
    total = len(df_result)
    valid_total = total - noise_count
    coverage = valid_total / total * 100

    # 外环：有效数据
    outer_colors = [MACRO_ANCHORS.get(n, {}).get("color", "#95A5A6") for n in valid_counts.index]
    ax2.pie(
        valid_counts.values, colors=outer_colors,
        radius=1.0, wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1)
    )
    # 内环：覆盖率
    inner_colors = ['#2ECC71', '#ECF0F1']
    ax2.pie(
        [valid_total, noise_count],
        colors=inner_colors,
        radius=0.65,
        wedgeprops=dict(width=0.25, edgecolor='white', linewidth=1)
    )

    ax2.text(0, 0, f"{coverage:.1f}%\n覆盖率", ha='center', va='center',
             fontsize=20, fontweight='bold', color='#2C3E50')
    ax2.set_title("数据覆盖率", fontsize=16, fontweight='bold', pad=20)

    fig.suptitle("宏观主题全景", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / "01_macro_distribution.png", dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info("  ✅ 图1 已保存")


# -------------------------------------------------------
# 图2: 微观主题规模排名柱状图
# -------------------------------------------------------
def plot_micro_topic_sizes(df_result, topic_model):
    """每个微观主题的文档数量排名"""
    logger.info("📊 绘制图2: 微观主题规模排名...")

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info['Topic'] != -1].copy()

    # 建立 topic_id -> macro_name 映射
    id_to_macro = df_result.groupby('micro_topic_id')['macro_topic_name'].first().to_dict()

    topic_info['macro'] = topic_info['Topic'].map(id_to_macro)
    topic_info['color'] = topic_info['macro'].map(
        lambda x: MACRO_ANCHORS.get(x, {}).get("color", "#95A5A6")
    )
    topic_info = topic_info.sort_values('Count', ascending=True)

    fig, ax = plt.subplots(figsize=(14, max(8, len(topic_info) * 0.35)))

    bars = ax.barh(
        range(len(topic_info)),
        topic_info['Count'].values,
        color=topic_info['color'].values,
        edgecolor='white',
        linewidth=0.5,
        height=0.7
    )

    # 标注关键词
    for i, (idx, row) in enumerate(topic_info.iterrows()):
        tid = row['Topic']
        topic_words = topic_model.get_topic(tid)
        kw = ", ".join([w[0] for w in topic_words[:3]])
        ax.text(row['Count'] + 2, i, f"T{tid}: {kw}", va='center', fontsize=8, color='#555')

    ax.set_yticks(range(len(topic_info)))
    ax.set_yticklabels([f"T{t}" for t in topic_info['Topic']], fontsize=9)
    ax.set_xlabel("文档数量", fontsize=12)
    ax.set_title("微观主题规模排名", fontsize=16, fontweight='bold')

    # 图例
    legend_patches = [
        mpatches.Patch(color=v['color'], label=k)
        for k, v in MACRO_ANCHORS.items()
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10, framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / "02_micro_topic_sizes.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    logger.info("  ✅ 图2 已保存")


# -------------------------------------------------------
# 图3: UMAP 二维投影散点图
# -------------------------------------------------------
def plot_umap_projection(topic_model, df_result):
    """将文档嵌入降维到2D，按主题着色"""
    logger.info("📊 绘制图3: UMAP 二维投影（这需要重新计算嵌入，可能较慢）...")

    try:
        # 获取模型内部已有的降维结果
        # BERTopic 内部的 UMAP 是降到 n_components 维（训练时设为10）
        # 我们需要再单独做一次降到2维的可视化投影

        # 提取文档嵌入（如果模型保存了的话）
        # 这里用一个快捷方式：直接从模型的 embeddings_ 属性获取
        if hasattr(topic_model, 'embedding_model') and topic_model.embedding_model is not None:
            # 重新加载文本做嵌入太慢，我们用另一种方式
            # 直接用 UMAP 对已有的高维表示做2D投影
            logger.info("  正在计算2D UMAP投影...")

            # 从模型中提取 c-TF-IDF 矩阵作为替代
            ctfidf = topic_model.c_tf_idf_
            if ctfidf is not None:
                from scipy.sparse import csr_matrix
                if isinstance(ctfidf, csr_matrix):
                    ctfidf_dense = ctfidf.toarray()
                else:
                    ctfidf_dense = np.array(ctfidf)

                # 如果维度太高，先选前N个主题
                topic_info = topic_model.get_topic_info()
                valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].values

                # 用UMAP降到2D
                umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.3,
                               metric='cosine', random_state=42)

                # 对文档级别做投影：使用 topic_model 的 reduce_embeddings 或手动计算
                # 更简单的方式：直接用 topic_model 的内置方法
                # 如果有预计算的嵌入
                if hasattr(topic_model, '_outliers') or True:
                    # 方案B：用代表性文档做可视化
                    logger.info("  使用代表性文档进行主题可视化...")
                    _plot_topic_centers(topic_model, df_result)
                    return
    except Exception as e:
        logger.warning(f"  ⚠️ UMAP投影计算失败: {e}，使用替代方案")
        _plot_topic_centers(topic_model, df_result)
        return


def _plot_topic_centers(topic_model, df_result):
    """替代方案：用主题关键词向量做2D投影"""
    try:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device=device)

        topic_info = topic_model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1].copy()

        # 为每个主题构建描述
        topic_texts = []
        topic_ids = []
        for _, row in valid_topics.iterrows():
            tid = row['Topic']
            words = " ".join([w[0] for w in topic_model.get_topic(tid)[:15]])
            topic_texts.append(words)
            topic_ids.append(tid)

        # 编码
        embeddings = emb_model.encode(topic_texts, show_progress_bar=False)

        # UMAP降到2D
        umap_2d = UMAP(n_components=2, n_neighbors=min(10, len(embeddings)-1),
                       min_dist=0.5, metric='cosine', random_state=42)
        coords = umap_2d.fit_transform(embeddings)

        # 映射颜色
        id_to_macro = df_result.groupby('micro_topic_id')['macro_topic_name'].first().to_dict()

        fig, ax = plt.subplots(figsize=(14, 10))

        for i, tid in enumerate(topic_ids):
            macro = id_to_macro.get(tid, "未知")
            color = MACRO_ANCHORS.get(macro, {}).get("color", "#95A5A6")
            count = valid_topics[valid_topics['Topic'] == tid]['Count'].values[0]
            size = max(50, count * 3)  # 点大小与文档数成正比

            ax.scatter(coords[i, 0], coords[i, 1], c=color, s=size,
                       alpha=0.7, edgecolors='white', linewidth=1.5, zorder=3)

            # 标注主题ID和关键词
            kw = ", ".join([w[0] for w in topic_model.get_topic(tid)[:2]])
            ax.annotate(f"T{tid}\n{kw}", (coords[i, 0], coords[i, 1]),
                        fontsize=7, ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor=color, alpha=0.8))

        ax.set_title("主题语义空间分布（UMAP 2D投影）", fontsize=16, fontweight='bold')
        ax.set_xlabel("UMAP-1", fontsize=11)
        ax.set_ylabel("UMAP-2", fontsize=11)

        # 图例
        legend_patches = [
            mpatches.Patch(color=v['color'], label=k)
            for k, v in MACRO_ANCHORS.items()
        ]
        ax.legend(handles=legend_patches, loc='upper left', fontsize=10, framealpha=0.9)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        fig.savefig(VIS_OUTPUT_DIR / "03_umap_projection.png", dpi=200, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        logger.info("  ✅ 图3 已保存")

    except Exception as e:
        logger.warning(f"  ⚠️ 主题中心可视化也失败: {e}")


# -------------------------------------------------------
# 图4: 映射置信度分布（直方图 + 箱线图）
# -------------------------------------------------------
def plot_confidence_distribution(df_result):
    """评估映射质量：置信度分布"""
    logger.info("📊 绘制图4: 映射置信度分布...")

    valid_df = df_result[df_result['macro_topic_name'] != "噪声数据(Outliers)"].copy()

    if valid_df.empty:
        logger.warning("  ⚠️ 无有效数据，跳过图4")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- 左上：全局置信度直方图 ---
    ax1 = axes[0, 0]
    confidences = valid_df['mapping_confidence'].values
    ax1.hist(confidences, bins=30, color='#3498DB', edgecolor='white', alpha=0.8)
    ax1.axvline(np.median(confidences), color='#E74C3C', linestyle='--', linewidth=2,
                label=f'中位数: {np.median(confidences):.3f}')
    ax1.axvline(0.35, color='#F39C12', linestyle=':', linewidth=2,
                label='低置信阈值: 0.35')
    ax1.set_xlabel("映射置信度", fontsize=11)
    ax1.set_ylabel("主题数量", fontsize=11)
    ax1.set_title("全局置信度分布", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)

    # --- 右上：按宏观类别的置信度箱线图 ---
    ax2 = axes[0, 1]
    macro_groups = valid_df.groupby('macro_topic_name')['mapping_confidence'].apply(list).to_dict()
    box_data = []
    box_labels = []
    box_colors = []
    for macro in MACRO_ANCHORS:
        if macro in macro_groups and len(macro_groups[macro]) > 0:
            box_data.append(macro_groups[macro])
            box_labels.append(macro)
            box_colors.append(MACRO_ANCHORS[macro]["color"])

    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.axhline(0.35, color='#F39C12', linestyle=':', linewidth=1.5)
    ax2.set_ylabel("映射置信度", fontsize=11)
    ax2.set_title("各类别置信度对比", fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)

    # --- 左下：置信度 vs 文档数量散点 ---
    ax3 = axes[1, 0]
    topic_stats = valid_df.groupby(['micro_topic_id', 'macro_topic_name']).agg(
        count=('note_id', 'count'),
        confidence=('mapping_confidence', 'first')
    ).reset_index()

    for macro in MACRO_ANCHORS:
        subset = topic_stats[topic_stats['macro_topic_name'] == macro]
        if not subset.empty:
            ax3.scatter(subset['count'], subset['confidence'],
                        c=MACRO_ANCHORS[macro]["color"], label=macro,
                        s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

    ax3.axhline(0.35, color='#F39C12', linestyle=':', linewidth=1.5, label='低置信阈值')
    ax3.set_xlabel("主题文档数量", fontsize=11)
    ax3.set_ylabel("映射置信度", fontsize=11)
    ax3.set_title("主题规模 vs 置信度", fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower right')

    # --- 右下：低置信度主题详情 ---
    ax4 = axes[1, 1]
    low_conf = topic_stats[topic_stats['confidence'] < 0.35].sort_values('confidence')
    if not low_conf.empty:
        y_pos = range(len(low_conf))
        colors = [MACRO_ANCHORS.get(row['macro_topic_name'], {}).get("color", "#95A5A6")
                  for _, row in low_conf.iterrows()]
        bars = ax4.barh(y_pos, low_conf['confidence'], color=colors, edgecolor='white')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(
            [f"T{row['micro_topic_id']} → {row['macro_topic_name'][:6]}"
             for _, row in low_conf.iterrows()],
            fontsize=9
        )
        ax4.axvline(0.35, color='#F39C12', linestyle=':', linewidth=1.5)
        ax4.set_xlabel("映射置信度", fontsize=11)
        ax4.set_title(f"低置信度主题 ({len(low_conf)} 个)", fontsize=13, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "所有主题置信度\n均高于阈值 ✓",
                 ha='center', va='center', fontsize=16, color='#2ECC71',
                 transform=ax4.transAxes)
        ax4.set_title("低置信度主题检查", fontsize=13, fontweight='bold')

    fig.suptitle("映射置信度评估", fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / "04_confidence_distribution.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    logger.info("  ✅ 图4 已保存")


# -------------------------------------------------------
# 图5: 各宏观类别的关键词词云对比
# -------------------------------------------------------
def plot_keyword_comparison(topic_model, df_result):
    """每个宏观类别提取Top关键词，做横向对比"""
    logger.info("📊 绘制图5: 宏观类别关键词对比...")

    macro_keywords = {}
    for macro in MACRO_ANCHORS:
        # 找到属于该宏观类别的所有微观主题
        belonging = df_result[df_result['macro_topic_name'] == macro]['micro_topic_id'].unique()
        belonging = [t for t in belonging if t != -1]

        # 汇总所有关键词及权重
        word_weights = {}
        for tid in belonging:
            topic_words = topic_model.get_topic(tid)
            if topic_words:
                for word, weight in topic_words[:15]:
                    word_weights[word] = word_weights.get(word, 0) + weight

        # 取Top15
        sorted_words = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)[:15]
        macro_keywords[macro] = sorted_words

    fig, axes = plt.subplots(1, 5, figsize=(25, 8))

    for i, (macro, kw_list) in enumerate(macro_keywords.items()):
        ax = axes[i]
        color = MACRO_ANCHORS[macro]["color"]

        if kw_list:
            words = [w[0] for w in kw_list][::-1]
            weights = [w[1] for w in kw_list][::-1]

            # 归一化权重用于颜色深浅
            max_w = max(weights) if weights else 1
            bar_colors = [plt.cm.colors.to_rgba(color, alpha=0.4 + 0.6 * w / max_w)
                          for w in weights]

            ax.barh(range(len(words)), weights, color=bar_colors, edgecolor='white', height=0.7)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=9)
            ax.set_xlabel("累积权重", fontsize=9)
        else:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14, color='#999')

        ax.set_title(macro, fontsize=12, fontweight='bold', color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle("五大宏观类别 Top 关键词对比", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / "05_keyword_comparison.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    logger.info("  ✅ 图5 已保存")


# -------------------------------------------------------
# 图6: 主题间距离热力图
# -------------------------------------------------------
def plot_topic_distance_heatmap(topic_model, df_result):
    """计算各微观主题之间的语义距离"""
    logger.info("📊 绘制图6: 主题间语义距离热力图...")

    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()

    if len(valid_topics) < 2:
        logger.warning("  ⚠️ 有效主题不足2个，跳过图6")
        return

    # 为每个主题构建关键词文本
    topic_texts = []
    topic_labels = []
    for tid in valid_topics:
        words = " ".join([w[0] for w in topic_model.get_topic(tid)[:15]])
        topic_texts.append(words)
        topic_labels.append(f"T{tid}")

    # 编码
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device=device)
        embeddings = emb_model.encode(topic_texts, show_progress_bar=False)

        # 余弦相似度矩阵
        sim_matrix = cosine_similarity(embeddings)

            # 按宏观类别排序
        id_to_macro = df_result.groupby('micro_topic_id')['macro_topic_name'].first().to_dict()

        # 获取每个主题的文档数（用于二级排序）
        topic_count_map = topic_info.set_index('Topic')['Count'].to_dict()

        # 构建排序键：(宏观类别序号, -文档数)
        macro_order = list(MACRO_ANCHORS.keys())
        sort_keys = []
        for tid in valid_topics:
            macro_name = id_to_macro.get(tid, "")
            macro_idx = macro_order.index(macro_name) if macro_name in macro_order else 999
            doc_count = topic_count_map.get(tid, 0)
            sort_keys.append((macro_idx, -doc_count))

        # 排序并获取索引
        sort_idx = sorted(range(len(valid_topics)), key=lambda i: sort_keys[i])


        # 绘制热力图
        n = len(valid_topics)
        fig_size = max(10, n * 0.45)
        # 排序并获取索引
        sort_idx = sorted(range(len(valid_topics)), key=lambda i: sort_keys[i])

        # ====== 补上这三行 ======
        sim_matrix_sorted = sim_matrix[np.ix_(sort_idx, sort_idx)]
        labels_sorted = [topic_labels[i] for i in sort_idx]
        n = len(valid_topics)
# =======================

        fig_size = max(10, n * 0.45)
        # 排序并获取索引
        sort_idx = sorted(range(len(valid_topics)), key=lambda i: sort_keys[i])
        sim_matrix_sorted = sim_matrix[np.ix_(sort_idx, sort_idx)]
        labels_sorted = [topic_labels[i] for i in sort_idx]
        n = len(valid_topics)

        fig_size = max(10, n * 0.45)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        cmap = LinearSegmentedColormap.from_list('custom',
                                                  ['#F0F0F0', '#3498DB', '#2C3E50'])
        im = ax.imshow(sim_matrix_sorted, cmap=cmap, aspect='auto', vmin=0.2, vmax=1.0)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels_sorted, fontsize=7, rotation=90)
        ax.set_yticklabels(labels_sorted, fontsize=7)

        # 添加宏观类别颜色条标记
        id_to_sorted_idx = {sort_idx[i]: i for i in range(n)}
        for i, tid in enumerate([valid_topics[j] for j in sort_idx]):
            macro = id_to_macro.get(tid, "")
            color = MACRO_ANCHORS.get(macro, {}).get("color", "#95A5A6")
            ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 0.4, 1,
                                        facecolor=color, clip_on=False, transform=ax.transData))
            ax.add_patch(plt.Rectangle((i - 0.5, n - 0.1), 1, 0.4,
                                        facecolor=color, clip_on=False, transform=ax.transData))

        plt.colorbar(im, ax=ax, shrink=0.7, label="余弦相似度")
        ax.set_title("主题间语义距离热力图", fontsize=16, fontweight='bold', pad=15)

        plt.tight_layout()
        fig.savefig(VIS_OUTPUT_DIR / "06_topic_distance_heatmap.png", dpi=200, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        logger.info("  ✅ 图6 已保存")

    except Exception as e:
        logger.warning(f"  ⚠️ 热力图绘制失败: {e}")


# -------------------------------------------------------
# 图7: 各类别文档长度分布（评估文本质量）
# -------------------------------------------------------
def plot_text_length_distribution(df_result, df_raw):
    """各类别文本长度分布，检测数据质量问题"""
    logger.info("📊 绘制图7: 文本长度分布...")

    # 合并原始数据的文本长度
    df_raw['text_len'] = df_raw['cleaned_full_text'].str.len()
    merged = df_result.merge(df_raw[['note_id', 'text_len']], on='note_id', how='left')

    valid_df = merged[merged['macro_topic_name'] != "噪声数据(Outliers)"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 左图：各类别文本长度箱线图 ---
    ax1 = axes[0]
    box_data = []
    box_labels = []
    box_colors = []
    for macro in MACRO_ANCHORS:
        subset = valid_df[valid_df['macro_topic_name'] == macro]['text_len'].dropna()
        if len(subset) > 0:
            box_data.append(subset)
            box_labels.append(macro)
            box_colors.append(MACRO_ANCHORS[macro]["color"])

    if box_data:
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax1.set_ylabel("文本长度（字符数）", fontsize=11)
    ax1.set_title("各类别文本长度分布", fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)

    # --- 右图：有效数据 vs 噪声的文本长度对比 ---
    ax2 = axes[1]
    noise_lens = merged[merged['macro_topic_name'] == "噪声数据(Outliers)"]['text_len'].dropna()
    valid_lens = valid_df['text_len'].dropna()

    if len(noise_lens) > 0 and len(valid_lens) > 0:
        ax2.hist(valid_lens, bins=50, color='#3498DB', alpha=0.6, label=f'有效数据 (均值:{valid_lens.mean():.0f})',
                 edgecolor='white', density=True)
        ax2.hist(noise_lens, bins=50, color='#E74C3C', alpha=0.6, label=f'噪声数据 (均值:{noise_lens.mean():.0f})',
                 edgecolor='white', density=True)
        ax2.set_xlabel("文本长度（字符数）", fontsize=11)
        ax2.set_ylabel("密度", fontsize=11)
        ax2.set_title("有效数据 vs 噪声：文本长度对比", fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, "数据不足", ha='center', va='center', fontsize=14)

    fig.suptitle("数据质量评估", fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / "07_text_length.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    logger.info("  ✅ 图7 已保存")


# -------------------------------------------------------
# 图8: 训练过程总览仪表盘（一页总览）
# -------------------------------------------------------
def plot_dashboard(df_result, topic_model):
    """一页式总览仪表盘"""
    logger.info("📊 绘制图8: 训练效果总览仪表盘...")

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    total = len(df_result)
    noise_count = (df_result['macro_topic_name'] == "噪声数据(Outliers)").sum()
    valid_count = total - noise_count
    coverage = valid_count / total * 100

    topic_info = topic_model.get_topic_info()
    n_micro = len(topic_info[topic_info['Topic'] != -1])
    n_macro = len(set(df_result['macro_topic_name']) - {"噪声数据(Outliers)"})

    valid_df = df_result[df_result['macro_topic_name'] != "噪声数据(Outliers)"]
    avg_conf = valid_df['mapping_confidence'].mean() if not valid_df.empty else 0
    low_conf_count = (valid_df['mapping_confidence'] < 0.35).sum() if not valid_df.empty else 0

    # --- KPI卡片行 ---
    kpi_data = [
        ("总数据量", f"{total:,}", "#3498DB"),
        ("覆盖率", f"{coverage:.1f}%", "#2ECC71" if coverage > 80 else "#E74C3C"),
        ("微观主题", f"{n_micro}", "#9B59B6"),
        ("宏观类别", f"{n_macro}/5", "#F39C12"),
        ("平均置信度", f"{avg_conf:.3f}", "#1ABC9C"),
        ("低置信主题", f"{low_conf_count}", "#E74C3C" if low_conf_count > 0 else "#2ECC71"),
    ]

    for i, (title, value, color) in enumerate(kpi_data):
        row = 0
        col = i
        if col >= 4:
            row = 1
            col = i - 4
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.65, value, ha='center', va='center',
                fontsize=28, fontweight='bold', color=color,
                transform=ax.transAxes)
        ax.text(0.5, 0.25, title, ha='center', va='center',
                fontsize=12, color='#666', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#E0E0E0')
            spine.set_linewidth(1.5)

    # --- 宏观分布环图 ---
    ax_pie = fig.add_subplot(gs[1, 2:])
    macro_counts = valid_df['macro_topic_name'].value_counts()
    colors = [MACRO_ANCHORS.get(n, {}).get("color", "#95A5A6") for n in macro_counts.index]
    wedges, _, _ = ax_pie.pie(
        macro_counts.values, labels=macro_counts.index, colors=colors,
        autopct='%1.0f%%', startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5),
        textprops={'fontsize': 9}
    )
    ax_pie.set_title("宏观分布", fontsize=13, fontweight='bold')

    # --- 置信度直方图 ---
    ax_conf = fig.add_subplot(gs[2, :2])
    if not valid_df.empty:
        ax_conf.hist(valid_df['mapping_confidence'], bins=25, color='#3498DB',
                     edgecolor='white', alpha=0.8)
        ax_conf.axvline(avg_conf, color='#E74C3C', linestyle='--', linewidth=2,
                        label=f'均值: {avg_conf:.3f}')
        ax_conf.axvline(0.35, color='#F39C12', linestyle=':', linewidth=2,
                        label='阈值: 0.35')
        ax_conf.legend(fontsize=10)
    ax_conf.set_xlabel("映射置信度", fontsize=11)
    ax_conf.set_title("置信度分布", fontsize=13, fontweight='bold')

    # --- 主题规模Top10 ---
    ax_top = fig.add_subplot(gs[2, 2:])
    topic_sizes = topic_info[topic_info['Topic'] != -1].nlargest(10, 'Count')
    id_to_macro = df_result.groupby('micro_topic_id')['macro_topic_name'].first().to_dict()

    bar_colors = [MACRO_ANCHORS.get(id_to_macro.get(t, ""), {}).get("color", "#95A5A6")
                  for t in topic_sizes['Topic']]
    ax_top.barh(range(len(topic_sizes)), topic_sizes['Count'].values[::-1],
                color=bar_colors[::-1], edgecolor='white', height=0.6)
    ax_top.set_yticks(range(len(topic_sizes)))
    ax_top.set_yticklabels([f"T{t}" for t in topic_sizes['Topic'].values[::-1]], fontsize=9)
    ax_top.set_xlabel("文档数", fontsize=11)
    ax_top.set_title("Top 10 主题规模", fontsize=13, fontweight='bold')

    fig.suptitle("BERTopic 模型训练效果总览", fontsize=22, fontweight='bold', y=1.01)
    fig.savefig(VIS_OUTPUT_DIR / "08_dashboard.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    logger.info("  ✅ 图8 已保存")


# -------------------------------------------------------
# 图9: 各宏观类别子主题占比堆叠图
# -------------------------------------------------------
def plot_macro_micro_breakdown(df_result, topic_model):
    """每个宏观类别下各子主题的占比"""
    logger.info("📊 绘制图9: 宏观-微观层级结构...")

    valid_df = df_result[df_result['macro_topic_name'] != "噪声数据(Outliers)"]

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for i, macro in enumerate(MACRO_ANCHORS):
        ax = axes[i]
        color = MACRO_ANCHORS[macro]["color"]

        subset = valid_df[valid_df['macro_topic_name'] == macro]
        if subset.empty:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14, color='#999')
            ax.set_title(macro, fontsize=12, fontweight='bold')
            continue

        micro_counts = subset.groupby('micro_topic_id')['note_id'].count().sort_values(ascending=True)
        total_in_macro = micro_counts.sum()

        # 生成渐变色
        n = len(micro_counts)
        from matplotlib.colors import to_rgba
        base_rgba = to_rgba(color)
        alphas = np.linspace(0.3, 0.95, n)
        bar_colors = [(base_rgba[0], base_rgba[1], base_rgba[2], a) for a in alphas]

        ax.barh(range(n), micro_counts.values, color=bar_colors, edgecolor='white', height=0.7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"T{t}" for t in micro_counts.index], fontsize=8)
        ax.set_xlabel("文档数", fontsize=9)
        ax.set_title(f"{macro}\n({total_in_macro}条)", fontsize=11, fontweight='bold', color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle("宏观类别内部子主题结构", fontsize=18, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / "09_macro_micro_breakdown.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    logger.info("  ✅ 图9 已保存")


# -------------------------------------------------------
# 图10: 主题关键词词云（如安装了wordcloud库）
# -------------------------------------------------------    """为每个宏观类别生成关键词词云"""
    logger.info("📊 绘制图10: 关键词词云...")

    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("  ⚠️ 未安装 wordcloud 库，跳过图10。安装命令: pip install wordcloud")
        return

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for i, macro in enumerate(MACRO_ANCHORS):
        ax = axes[i]
        color = MACRO_ANCHORS[macro]["color"]

        belonging = df_result[df_result['macro_topic_name'] == macro]['micro_topic_id'].unique()
        belonging = [t for t in belonging if t != -1]

        word_freq = {}
        for tid in belonging:
            topic_words = topic_model.get_topic(tid)
            if topic_words:
                for word, weight in topic_words[:20]:
                    word_freq[word] = word_freq.get(word, 0) + weight

        if word_freq:
            try:
                wc = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=40,
                    color_func=lambda *args, **kwargs: color,
                    font_path=None,  # 系统默认中文字体
                    prefer_horizontal=0.7
                ).generate_from_frequencies(word_freq)
                ax.imshow(wc, interpolation='bilinear')
            except Exception:
                # 如果中文字体问题，用纯色文字
                ax.text(0.5, 0.5, "\n".join(list(word_freq.keys())[:8]),
                        ha='center', va='center', fontsize=10, color=color)
        else:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14, color='#999')

        ax.set_title(macro, fontsize=12, fontweight='bold', color=color)
        ax.axis('off')

    fig.suptitle("各类别关键词词云", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / "10_wordclouds.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    logger.info("  ✅ 图10 已保存")
def plot_wordclouds(topic_model, df_result):
    """为每个宏观类别生成关键词词云"""
    logger.info("📊 绘制图10: 关键词词云...")

    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("  ⚠️ 未安装 wordcloud 库，跳过图10。安装命令: pip install wordcloud")
        return

    # ====== 指定中文字体路径 ======
    WC_FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
    if not os.path.exists(WC_FONT_PATH):
        # 兜底路径
        fallback_fonts = [
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\simkai.ttf",
            r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
        for fp in fallback_fonts:
            if os.path.exists(fp):
                WC_FONT_PATH = fp
                break
        else:
            logger.warning("  ⚠️ 未找到中文字体文件，词云中文将显示为方块")
            WC_FONT_PATH = None

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for i, macro in enumerate(MACRO_ANCHORS):
        ax = axes[i]
        color = MACRO_ANCHORS[macro]["color"]

        belonging = df_result[df_result['macro_topic_name'] == macro]['micro_topic_id'].unique()
        belonging = [t for t in belonging if t != -1]

        word_freq = {}
        for tid in belonging:
            topic_words = topic_model.get_topic(tid)
            if topic_words:
                for word, weight in topic_words[:20]:
                    word_freq[word] = word_freq.get(word, 0) + weight

        if word_freq:
            try:
                wc = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=40,
                    color_func=lambda *args, **kwargs: color,
                    font_path=WC_FONT_PATH,  # ← 改这里：从 None 改为实际路径
                    prefer_horizontal=0.7
                ).generate_from_frequencies(word_freq)
                ax.imshow(wc, interpolation='bilinear')
            except Exception as e:
                logger.warning(f"  ⚠️ 词云生成失败 ({macro}): {e}")
                ax.text(0.5, 0.5, "\n".join(list(word_freq.keys())[:8]),
                        ha='center', va='center', fontsize=10, color=color)
        else:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14, color='#999')

        ax.set_title(macro, fontsize=12, fontweight='bold', color=color)
        ax.axis('off')

    fig.suptitle("各类别关键词词云", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / "10_wordclouds.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    logger.info("  ✅ 图10 已保存")

# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("  BERTopic 模型训练效果 & 评估效果 可视化看板")
    print("=" * 60)

    # ====== 第1步：先设置 seaborn 样式 ======
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # ====== 第2步：再设置字体（必须在sns之后！）======
    print("\n🔤 配置中文字体...")
    font_name = setup_chinese_font()

    # ====== 第3步：双重保险，再显式设置一次 ======
    plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans'] if font_name else ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载数据
    topic_model, df_result, df_raw = load_all()

    print(f"\n📊 数据概览:")
    print(f"   总数据量: {len(df_result)}")
    print(f"   噪声数据: {(df_result['macro_topic_name'] == '噪声数据(Outliers)').sum()}")
    print(f"   有效主题: {len(topic_model.get_topic_info()) - 1}")

    # 逐个生成可视化
    print(f"\n🎨 开始生成可视化图表，输出目录: {VIS_OUTPUT_DIR}\n")

    plot_macro_distribution(df_result)               # 图1
    plot_micro_topic_sizes(df_result, topic_model)   # 图2
    plot_umap_projection(topic_model, df_result)     # 图3
    plot_confidence_distribution(df_result)           # 图4
    plot_keyword_comparison(topic_model, df_result)  # 图5
    plot_topic_distance_heatmap(topic_model, df_result)  # 图6
    plot_text_length_distribution(df_result, df_raw)  # 图7
    plot_dashboard(df_result, topic_model)            # 图8
    plot_macro_micro_breakdown(df_result, topic_model)  # 图9
    plot_wordclouds(topic_model, df_result)           # 图10

    print("\n" + "=" * 60)
    print("  ✅ 全部可视化图表生成完毕！")
    print(f"  📂 输出目录: {VIS_OUTPUT_DIR}")
    print("=" * 60)
