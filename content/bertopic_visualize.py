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

# Windows 控制台常默认 GBK，遇到 emoji/部分符号会 UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ==================== 环境配置 ====================
os.environ['NUMBA_CACHE_DIR'] = str(Path(os.environ.get('TEMP', '/tmp')) / 'numba_cache')
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
# Windows + BERTopic/UMAP 常在 import 阶段触发 numba JIT 编译，可能导致卡住很久。
# 这里默认禁用 JIT 以保证可视化脚本稳定可跑；如需启用可在运行前设置环境变量 NUMBA_DISABLE_JIT=0
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
# 避免在某些终端/IDE 中 numba 的彩色错误高亮触发控制台句柄问题
os.environ.setdefault("NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING", "1")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 可视化库 ====================
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，防止弹窗卡死
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import seaborn as sns

# ==================== 算法库 ====================
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity

# plotly（用于桑基图，可选依赖）
try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None

# ==================== 路径配置 ====================
WORK_DIR = Path(__file__).resolve().parent
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


# ==================== 配色模式 ====================
# 支持:
# - rainbow: 经典高区分彩虹配色（当前默认）
# - scientific: 低饱和科研配色
COLOR_STYLE = "rainbow"

SCIENTIFIC_MACRO_PALETTE = {
    "AI内容创作": "#3B5B92",      # muted blue
    "AI应用与测评": "#5EA3A3",    # desaturated teal
    "AI学习教程": "#8DAA5B",      # olive green
    "AI赋能工作生活": "#C18B5A",  # muted ochre
    "AI社会反思": "#8B6BAF",      # muted violet
}


def get_macro_color(macro_name: str, default: str = "#8F96A3") -> str:
    """根据配色模式返回宏观主题颜色。"""
    if COLOR_STYLE == "scientific" and macro_name in SCIENTIFIC_MACRO_PALETTE:
        return SCIENTIFIC_MACRO_PALETTE[macro_name]
    return MACRO_ANCHORS.get(macro_name, {}).get("color", default)


THEME_ACCENTS = {
    "blue": "#3498DB",
    "red": "#E74C3C",
    "green": "#2ECC71",
    "orange": "#F39C12",
    "purple": "#9B59B6",
    "teal": "#1ABC9C",
}


PAPER_NEUTRAL = {
    "bg": "#FFFFFF",
    "grid": "#D7DCE2",
    "axis": "#C4CAD3",
    "text": "#2F3A4A",
    "subtext": "#5B6778",
}


def apply_publication_theme():
    """统一论文风图表主题（字体、颜色、网格、导出观感）。"""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": PAPER_NEUTRAL["bg"],
        "axes.facecolor": PAPER_NEUTRAL["bg"],
        "axes.edgecolor": PAPER_NEUTRAL["axis"],
        "axes.labelcolor": PAPER_NEUTRAL["text"],
        "axes.titlecolor": PAPER_NEUTRAL["text"],
        "xtick.color": PAPER_NEUTRAL["subtext"],
        "ytick.color": PAPER_NEUTRAL["subtext"],
        "grid.color": PAPER_NEUTRAL["grid"],
        "grid.alpha": 0.55,
        "grid.linestyle": "--",
        "grid.linewidth": 0.65,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#D2D8E0",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "savefig.facecolor": PAPER_NEUTRAL["bg"],
        "savefig.edgecolor": "none",
    })


def style_axis(ax, grid_axis: str = "y"):
    """统一子图轴样式。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PAPER_NEUTRAL["axis"])
    ax.spines["bottom"].set_color(PAPER_NEUTRAL["axis"])
    ax.tick_params(colors=PAPER_NEUTRAL["subtext"])
    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, alpha=0.45, linestyle="--", linewidth=0.65)


def save_figure(fig, filename: str, dpi: int = 300):
    """统一导出参数，保证论文插图清晰度与留白。"""
    fig.savefig(
        VIS_OUTPUT_DIR / filename,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=PAPER_NEUTRAL["bg"],
        edgecolor="none",
    )


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

    # 双环图：
    # - 外环：仅有效数据中的宏观主题占比
    # - 内环：覆盖率（有效 vs 噪声）
    fig, ax = plt.subplots(figsize=(12, 7))

    macro_counts = df_result["macro_topic_name"].value_counts()

    # 分离噪声与有效数据
    noise_count = int(macro_counts.get("噪声数据(Outliers)", 0))
    valid_counts = macro_counts.drop("噪声数据(Outliers)", errors="ignore")

    total = int(len(df_result))
    valid_total = int(total - noise_count)
    coverage = (valid_total / total * 100) if total > 0 else 0.0

    macro_order = [m for m in MACRO_ANCHORS.keys() if m in valid_counts.index]
    other_macros = [m for m in valid_counts.index if m not in macro_order]
    macro_order.extend(other_macros)

    sizes = [int(valid_counts[m]) for m in macro_order]
    labels = list(macro_order)
    colors = [get_macro_color(m) for m in macro_order]

    if sum(sizes) == 0:
        ax.text(0.5, 0.5, "无有效数据", ha="center", va="center", fontsize=16, color="#666",
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        # 外环：宏观主题（仅有效样本）
        _, _, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.86,
            radius=1.0,
            wedgeprops=dict(width=0.30, edgecolor="white", linewidth=1.6),
            textprops={"fontsize": 11}
        )
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight("bold")
            t.set_color("#2C3E50")

        # 内环：覆盖率（有效 vs 噪声）
        inner_sizes = [valid_total, noise_count]
        inner_labels = ["有效样本", "噪声样本"]
        inner_colors = ["#6F879C", "#D9DEE5"]
        _, inner_texts = ax.pie(
            inner_sizes,
            labels=inner_labels,
            colors=inner_colors,
            startangle=90,
            labeldistance=0.45,
            radius=0.64,
            wedgeprops=dict(width=0.30, edgecolor="white", linewidth=1.2),
            textprops={"fontsize": 10, "color": "#374151"}
        )
        for t in inner_texts:
            t.set_fontsize(10)

        # 中心信息：覆盖率与样本数量
        center_text = f"{coverage:.1f}%\n覆盖率\n\nn={total:,}"
        ax.text(0, 0, center_text, ha="center", va="center",
                fontsize=13, fontweight="bold", color="#2C3E50", linespacing=1.2)

    ax.set_title("宏观主题分布与覆盖率（双环）", fontsize=18, fontweight="bold", pad=18)
    save_figure(fig, "01_macro_distribution.png")
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
        lambda x: get_macro_color(x)
    )
    topic_info = topic_info.sort_values('Count', ascending=True)

    fig, ax = plt.subplots(figsize=(16, max(9, len(topic_info) * 0.38)))

    ax.barh(
        range(len(topic_info)),
        topic_info['Count'].values,
        color=topic_info['color'].values,
        edgecolor='#F7F7F7',
        linewidth=0.8,
        alpha=0.92,
        height=0.7
    )

    # 关键词标注仅展示头部主题，避免过度拥挤
    top_label_ids = set(topic_info.nlargest(12, "Count")["Topic"].tolist())
    max_count = float(topic_info["Count"].max()) if len(topic_info) else 1.0
    for i, (idx, row) in enumerate(topic_info.iterrows()):
        tid = row['Topic']
        topic_words = topic_model.get_topic(tid)
        kw = ", ".join([w[0] for w in topic_words[:3]])
        if tid in top_label_ids:
            ax.text(row['Count'] + max(2, max_count * 0.01), i, f"T{tid}: {kw}",
                    va='center', fontsize=8.8, color='#374151')

    ax.set_yticks(range(len(topic_info)))
    ax.set_yticklabels([f"T{t}" for t in topic_info['Topic']], fontsize=9)
    ax.set_xlabel("文档数量", fontsize=12)
    ax.set_title("微观主题规模排名（Top主题附关键词）", fontsize=16, fontweight='bold')
    ax.set_xlim(0, max_count * 1.40)

    # 图例
    legend_patches = [
        mpatches.Patch(color=get_macro_color(k), label=k)
        for k in MACRO_ANCHORS.keys()
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10, framealpha=0.95, ncol=1)

    style_axis(ax, grid_axis="x")
    plt.tight_layout()
    save_figure(fig, "02_micro_topic_sizes.png")
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

        top_annotated = set(valid_topics.nlargest(16, "Count")["Topic"].tolist())
        for i, tid in enumerate(topic_ids):
            macro = id_to_macro.get(tid, "未知")
            color = get_macro_color(macro)
            count = valid_topics[valid_topics['Topic'] == tid]['Count'].values[0]
            size = max(70, count * 3.2)  # 点大小与文档数成正比

            ax.scatter(coords[i, 0], coords[i, 1], c=color, s=size,
                       alpha=0.78, edgecolors='white', linewidth=1.4, zorder=3)

            # 标注主题ID和关键词
            if tid in top_annotated:
                kw = ", ".join([w[0] for w in topic_model.get_topic(tid)[:2]])
                ax.annotate(f"T{tid}\n{kw}", (coords[i, 0], coords[i, 1]),
                            fontsize=7.2, ha='center', va='bottom',
                            xytext=(0, 8), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                      edgecolor=color, alpha=0.86))

        ax.set_title("主题语义空间分布（UMAP 2D投影）", fontsize=16, fontweight='bold')
        ax.set_xlabel("UMAP-1", fontsize=11)
        ax.set_ylabel("UMAP-2", fontsize=11)

        # 图例
        legend_patches = [
            mpatches.Patch(color=get_macro_color(k), label=k)
            for k in MACRO_ANCHORS.keys()
        ]
        ax.legend(handles=legend_patches, loc='upper left', fontsize=10, framealpha=0.9)

        style_axis(ax, grid_axis="both")

        plt.tight_layout()
        save_figure(fig, "03_umap_projection.png")
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
    ax1.hist(confidences, bins=30, color=THEME_ACCENTS["blue"], edgecolor='white', alpha=0.85)
    ax1.axvline(np.median(confidences), color=THEME_ACCENTS["red"], linestyle='--', linewidth=2,
                label=f'中位数: {np.median(confidences):.3f}')
    ax1.axvline(0.35, color=THEME_ACCENTS["orange"], linestyle=':', linewidth=2,
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
            box_colors.append(get_macro_color(macro))

    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.axhline(0.35, color=THEME_ACCENTS["orange"], linestyle=':', linewidth=1.5)
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
                        c=get_macro_color(macro), label=macro,
                        s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

    ax3.axhline(0.35, color=THEME_ACCENTS["orange"], linestyle=':', linewidth=1.5, label='低置信阈值')
    ax3.set_xlabel("主题文档数量", fontsize=11)
    ax3.set_ylabel("映射置信度", fontsize=11)
    ax3.set_title("主题规模 vs 置信度", fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower right')

    # --- 右下：低置信度主题详情 ---
    ax4 = axes[1, 1]
    low_conf = topic_stats[topic_stats['confidence'] < 0.35].sort_values('confidence')
    if not low_conf.empty:
        y_pos = range(len(low_conf))
        colors = [get_macro_color(row['macro_topic_name'])
                  for _, row in low_conf.iterrows()]
        bars = ax4.barh(y_pos, low_conf['confidence'], color=colors, edgecolor='white')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(
            [f"T{row['micro_topic_id']} → {row['macro_topic_name'][:6]}"
             for _, row in low_conf.iterrows()],
            fontsize=9
        )
        ax4.axvline(0.35, color=THEME_ACCENTS["orange"], linestyle=':', linewidth=1.5)
        ax4.set_xlabel("映射置信度", fontsize=11)
        ax4.set_title(f"低置信度主题 ({len(low_conf)} 个)", fontsize=13, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "所有主题置信度\n均高于阈值 ✓",
                 ha='center', va='center', fontsize=16, color=get_macro_color("AI学习教程"),
                 transform=ax4.transAxes)
        ax4.set_title("低置信度主题检查", fontsize=13, fontweight='bold')

    for ax in axes.flatten():
        style_axis(ax, grid_axis="y")

    fig.suptitle("映射置信度评估", fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "04_confidence_distribution.png")
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

    # 2x3 布局，显著提升中文标签可读性
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    for i, (macro, kw_list) in enumerate(macro_keywords.items()):
        ax = axes[i]
        color = get_macro_color(macro)

        if kw_list:
            words = [w[0] for w in kw_list][:12][::-1]
            weights = [w[1] for w in kw_list][:12][::-1]

            # 归一化权重用于颜色深浅
            max_w = max(weights) if weights else 1
            bar_colors = [plt.cm.colors.to_rgba(color, alpha=0.4 + 0.6 * w / max_w)
                          for w in weights]

            ax.barh(range(len(words)), weights, color=bar_colors, edgecolor='white', height=0.7)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=10)
            ax.set_xlabel("累积权重", fontsize=10)
            for j, w in enumerate(weights):
                ax.text(w + max(weights) * 0.02, j, f"{w:.2f}", va='center', fontsize=8, color="#4B5563")
        else:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14, color='#999')

        ax.set_title(macro, fontsize=13, fontweight='bold', color=color)
        style_axis(ax, grid_axis="x")
        ax.set_xlim(0, max(weights) * 1.30 if kw_list else 1)

    # 删除空白子图（第6格）
    for j in range(len(macro_keywords), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("五大宏观类别 Top 关键词对比", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "05_keyword_comparison.png")
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
        sim_matrix_sorted = sim_matrix[np.ix_(sort_idx, sort_idx)]
        labels_sorted = [topic_labels[i] for i in sort_idx]
        n = len(valid_topics)

        fig_size = max(10, n * 0.45)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        cmap = LinearSegmentedColormap.from_list('custom',
                                                  ['#F5F7FA', '#8FAFCE', '#2E4E78'])
        im = ax.imshow(sim_matrix_sorted, cmap=cmap, aspect='auto', vmin=0.2, vmax=1.0)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels_sorted, fontsize=7, rotation=90)
        ax.set_yticklabels(labels_sorted, fontsize=7)

        # 添加宏观类别颜色条标记
        id_to_sorted_idx = {sort_idx[i]: i for i in range(n)}
        for i, tid in enumerate([valid_topics[j] for j in sort_idx]):
            macro = id_to_macro.get(tid, "")
            color = get_macro_color(macro)
            ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 0.4, 1,
                                        facecolor=color, clip_on=False, transform=ax.transData))
            ax.add_patch(plt.Rectangle((i - 0.5, n - 0.1), 1, 0.4,
                                        facecolor=color, clip_on=False, transform=ax.transData))

        cbar = plt.colorbar(im, ax=ax, shrink=0.7, label="余弦相似度")
        cbar.outline.set_edgecolor(PAPER_NEUTRAL["axis"])
        ax.set_title("主题间语义距离热力图", fontsize=16, fontweight='bold', pad=15)
        style_axis(ax, grid_axis="none")

        plt.tight_layout()
        save_figure(fig, "06_topic_distance_heatmap.png")
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
            box_colors.append(get_macro_color(macro))

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
        ax2.hist(valid_lens, bins=45, color=THEME_ACCENTS["blue"], alpha=0.58, label=f'有效数据 (均值:{valid_lens.mean():.0f})',
                 edgecolor='white', density=True)
        ax2.hist(noise_lens, bins=45, color=THEME_ACCENTS["red"], alpha=0.58, label=f'噪声数据 (均值:{noise_lens.mean():.0f})',
                 edgecolor='white', density=True)
        ax2.set_xlabel("文本长度（字符数）", fontsize=11)
        ax2.set_ylabel("密度", fontsize=11)
        ax2.set_title("有效数据 vs 噪声：文本长度对比", fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, "数据不足", ha='center', va='center', fontsize=14)

    style_axis(ax1, grid_axis="y")
    style_axis(ax2, grid_axis="y")
    fig.suptitle("数据质量评估", fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "07_text_length.png")
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
        ("总数据量", f"{total:,}", THEME_ACCENTS["blue"]),
        ("覆盖率", f"{coverage:.1f}%", THEME_ACCENTS["green"] if coverage > 80 else THEME_ACCENTS["red"]),
        ("微观主题", f"{n_micro}", THEME_ACCENTS["purple"]),
        ("宏观类别", f"{n_macro}/5", THEME_ACCENTS["orange"]),
        ("平均置信度", f"{avg_conf:.3f}", THEME_ACCENTS["teal"]),
        ("低置信主题", f"{low_conf_count}", THEME_ACCENTS["red"] if low_conf_count > 0 else THEME_ACCENTS["green"]),
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
            spine.set_edgecolor('#D8DEE7')
            spine.set_linewidth(1.2)

    # --- 宏观分布环图 ---
    ax_pie = fig.add_subplot(gs[1, 2:])
    macro_counts = valid_df['macro_topic_name'].value_counts()
    colors = [get_macro_color(n) for n in macro_counts.index]
    _, _, _ = ax_pie.pie(
        macro_counts.values, labels=macro_counts.index, colors=colors,
        autopct='%1.0f%%', startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5),
        textprops={'fontsize': 9}
    )
    ax_pie.set_title("宏观分布", fontsize=13, fontweight='bold')

    # --- 置信度直方图 ---
    ax_conf = fig.add_subplot(gs[2, :2])
    if not valid_df.empty:
        ax_conf.hist(valid_df['mapping_confidence'], bins=25, color=THEME_ACCENTS["blue"],
                     edgecolor='white', alpha=0.8)
        ax_conf.axvline(avg_conf, color=THEME_ACCENTS["red"], linestyle='--', linewidth=2,
                        label=f'均值: {avg_conf:.3f}')
        ax_conf.axvline(0.35, color=THEME_ACCENTS["orange"], linestyle=':', linewidth=2,
                        label='阈值: 0.35')
        ax_conf.legend(fontsize=10)
    ax_conf.set_xlabel("映射置信度", fontsize=11)
    ax_conf.set_title("置信度分布", fontsize=13, fontweight='bold')

    # --- 主题规模Top10 ---
    ax_top = fig.add_subplot(gs[2, 2:])
    topic_sizes = topic_info[topic_info['Topic'] != -1].nlargest(10, 'Count')
    id_to_macro = df_result.groupby('micro_topic_id')['macro_topic_name'].first().to_dict()

    bar_colors = [get_macro_color(id_to_macro.get(t, ""))
                  for t in topic_sizes['Topic']]
    ax_top.barh(range(len(topic_sizes)), topic_sizes['Count'].values[::-1],
                color=bar_colors[::-1], edgecolor='white', height=0.6)
    ax_top.set_yticks(range(len(topic_sizes)))
    ax_top.set_yticklabels([f"T{t}" for t in topic_sizes['Topic'].values[::-1]], fontsize=9)
    ax_top.set_xlabel("文档数", fontsize=11)
    ax_top.set_title("Top 10 主题规模", fontsize=13, fontweight='bold')
    style_axis(ax_conf, grid_axis="y")
    style_axis(ax_top, grid_axis="x")

    fig.suptitle("BERTopic 模型训练效果总览", fontsize=22, fontweight='bold', y=1.01)
    save_figure(fig, "08_dashboard.png")
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
        color = get_macro_color(macro)

        subset = valid_df[valid_df['macro_topic_name'] == macro]
        if subset.empty:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14, color='#999')
            ax.set_title(macro, fontsize=12, fontweight='bold')
            continue

        micro_counts = subset.groupby('micro_topic_id')['note_id'].count().sort_values(ascending=False)
        if len(micro_counts) > 8:
            top = micro_counts.head(8).sort_values(ascending=True)
            other = micro_counts.iloc[8:].sum()
            micro_counts = pd.concat([pd.Series({"其他": other}), top]).sort_values(ascending=True)
        else:
            micro_counts = micro_counts.sort_values(ascending=True)
        total_in_macro = micro_counts.sum()

        # 生成渐变色
        n = len(micro_counts)
        base_rgba = to_rgba(color)
        alphas = np.linspace(0.35, 0.95, n)
        bar_colors = [(base_rgba[0], base_rgba[1], base_rgba[2], a) for a in alphas]

        ax.barh(range(n), micro_counts.values, color=bar_colors, edgecolor='white', height=0.7)
        ax.set_yticks(range(n))
        y_labels = [f"T{t}" if str(t).isdigit() else str(t) for t in micro_counts.index]
        ax.set_yticklabels(y_labels, fontsize=8.5)
        ax.set_xlabel("文档数", fontsize=9)
        ax.set_title(f"{macro}\n({total_in_macro}条)", fontsize=11, fontweight='bold', color=color)
        style_axis(ax, grid_axis="x")

    fig.suptitle("宏观类别内部子主题结构", fontsize=18, fontweight='bold', y=1.03)
    plt.tight_layout()
    save_figure(fig, "09_macro_micro_breakdown.png")
    plt.close(fig)
    logger.info("  ✅ 图9 已保存")


# -------------------------------------------------------
# 图10: 主题关键词词云（如安装了wordcloud库）
# -------------------------------------------------------
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

    # 五个宏观主题单行横向排布
    n_macro = len(MACRO_ANCHORS)
    fig, axes = plt.subplots(1, n_macro, figsize=(32, 7.5))
    axes = np.atleast_1d(axes).ravel()

    for i, macro in enumerate(MACRO_ANCHORS):
        ax = axes[i]
        color = get_macro_color(macro)

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
                # 通过更高 min_font_size + 更少词数，提高可读性
                boosted_freq = {k: (v ** 0.65) for k, v in word_freq.items()}
                wc = WordCloud(
                    width=480,
                    height=340,
                    background_color='white',
                    max_words=24,
                    min_font_size=14,
                    max_font_size=110,
                    relative_scaling=0.72,
                    color_func=lambda *args, **kwargs: color,
                    font_path=WC_FONT_PATH,  # ← 改这里：从 None 改为实际路径
                    prefer_horizontal=0.9,
                    collocations=False,
                    repeat=True
                ).generate_from_frequencies(boosted_freq)
                ax.imshow(wc, interpolation='bilinear')
            except Exception as e:
                logger.warning(f"  ⚠️ 词云生成失败 ({macro}): {e}")
                ax.text(0.5, 0.5, "\n".join(list(word_freq.keys())[:8]),
                        ha='center', va='center', fontsize=10, color=color)
        else:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14, color='#999')

        ax.set_title(macro, fontsize=26, fontweight='bold', color=color, pad=14)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#E1E6EF")
            spine.set_linewidth(0.8)

    fig.suptitle("各类别关键词词云", fontsize=18, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    save_figure(fig, "10_wordclouds.png")
    plt.close(fig)
    logger.info("  ✅ 图10 已保存")


# -------------------------------------------------------
# 图11: 主题桑基图（原始笔记 → 聚类主题 → 五大宏观）
# -------------------------------------------------------
def _hex_to_rgba_str(hex_color: str, alpha: float = 0.65) -> str:
    try:
        from matplotlib.colors import to_rgb
        r, g, b = to_rgb(hex_color)
        return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})"
    except Exception:
        return hex_color


def plot_topic_sankey(df_result, topic_model, max_micro_nodes: int = 18):
    """
    三段桑基图（从左到右）：
    1) 左列仅一个节点「原始笔记」（汇总全部有效笔记流量）
    2) 中列：BERTopic 微观主题（文档数 Top-N，其余合并为「其他微观主题」以控制纵向高度）
    3) 右列：五大 AI 宏观主题（标签中带整体占比%）

    输出：
    - 11_topic_sankey.html（交互式）
    - 11_topic_sankey.png（如果安装了 kaleido）
    """
    logger.info("📊 绘制图11: 主题桑基图（原始笔记 → 聚类主题 → 五大宏观）...")

    if go is None:
        logger.warning("  ⚠️ 未安装 plotly，跳过图11。安装命令: pip install plotly kaleido")
        return

    valid_df = df_result[df_result["macro_topic_name"] != "噪声数据(Outliers)"].copy()
    valid_df = valid_df[valid_df["micro_topic_id"] != -1]
    if valid_df.empty:
        logger.warning("  ⚠️ 无有效数据，跳过图11")
        return

    total_notes = len(valid_df)

    OTHER_MICRO_ID = -999
    topic_counts = valid_df.groupby("micro_topic_id").size().sort_values(ascending=False)
    keep_ids = set(topic_counts.head(max(1, int(max_micro_nodes))).index.astype(int))
    plot_df = valid_df.copy()
    plot_df["_mid"] = plot_df["micro_topic_id"].astype(int)
    plot_df.loc[~plot_df["_mid"].isin(keep_ids), "_mid"] = OTHER_MICRO_ID

    def _micro_label(tid: int) -> str:
        if tid == OTHER_MICRO_ID:
            return "其他微观主题"
        words = topic_model.get_topic(int(tid)) or []
        kw = "、".join([w[0] for w in words[:3]]) if words else ""
        return f"T{int(tid)} {kw}".strip()

    macro_order = list(MACRO_ANCHORS.keys())
    macro_counts = (
        valid_df["macro_topic_name"].value_counts().reindex(macro_order, fill_value=0)
    )
    macro_pct = {m: 100.0 * int(macro_counts[m]) / total_notes for m in macro_order}
    macro_labels = [f"{m}<br>{macro_pct[m]:.1f}%" for m in macro_order]

    NOTE_ROOT = "原始笔记"
    note_labels = [NOTE_ROOT]
    n_note = 1

    micro_ids = sorted(plot_df["_mid"].astype(int).unique().tolist())
    micro_labels = [_micro_label(tid) for tid in micro_ids]
    micro_id_to_label = {tid: lab for tid, lab in zip(micro_ids, micro_labels)}

    node_labels = note_labels + micro_labels + macro_labels
    n_micro = len(micro_labels)
    n_macro = len(macro_order)

    note_label_to_idx = {NOTE_ROOT: 0}
    micro_label_to_idx = {lab: n_note + i for i, lab in enumerate(micro_labels)}

    # 三列节点 y 均摊在 (margin, 1-margin)，避免中列因节点多而视觉“鼓包”过高
    margin = 0.06

    def _ys_spread(n: int) -> list:
        if n <= 0:
            return []
        if n == 1:
            return [0.5]
        return [margin + (1 - 2 * margin) * i / (n - 1) for i in range(n)]

    ys_note = _ys_spread(n_note)
    ys_micro = _ys_spread(n_micro)
    ys_macro = _ys_spread(n_macro)
    node_x = [0.0] * n_note + [0.5] * n_micro + [1.0] * n_macro
    node_y = ys_note + ys_micro + ys_macro

    # 中列不用浅蓝填充，避免与灰色流量衔接处形成整条蓝色竖边（观感像瑕疵）
    micro_fill = "#D5DBDB"
    node_colors = (
        ["#BDC3C7"] * n_note
        + [micro_fill] * n_micro
        + [get_macro_color(m) for m in macro_order]
    )

    sources = []
    targets = []
    values = []
    link_colors = []

    # 原始笔记（汇总）→ 各微观主题（已合并长尾）
    root_i = note_label_to_idx[NOTE_ROOT]
    note_to_micro = (
        plot_df.groupby("_mid")
        .size()
        .reset_index(name="count")
    )
    for _, row in note_to_micro.iterrows():
        tid = int(row["_mid"])
        cnt = int(row["count"])
        mlab = micro_id_to_label.get(tid)
        if mlab is None:
            continue
        ti = micro_label_to_idx.get(mlab)
        if ti is None:
            continue
        sources.append(root_i)
        targets.append(ti)
        values.append(cnt)
        link_colors.append("rgba(149,165,166,0.4)")

    # 微观主题 → 五大宏观
    mm = (
        plot_df.groupby(["_mid", "macro_topic_name"])
        .size()
        .reset_index(name="count")
    )
    macro_idx_by_name = {m: n_note + n_micro + i for i, m in enumerate(macro_order)}
    for _, row in mm.iterrows():
        tid = int(row["_mid"])
        macro = row["macro_topic_name"]
        cnt = int(row["count"])
        if macro not in macro_idx_by_name or tid not in micro_id_to_label:
            continue
        si = micro_label_to_idx[micro_id_to_label[tid]]
        ti = macro_idx_by_name[macro]
        sources.append(si)
        targets.append(ti)
        values.append(cnt)
        base = get_macro_color(macro)
        if isinstance(base, str) and base.startswith("#") and len(base) == 7:
            link_colors.append(_hex_to_rgba_str(base, alpha=0.55))
        else:
            link_colors.append(base)

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=2,
                thickness=7,
                line=dict(color="rgba(255,255,255,0.45)", width=0.5),
                label=node_labels,
                color=node_colors,
                x=node_x,
                y=node_y,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            )
        )]
    )

    fig.update_layout(
        title_text=(
            f"主题桑基图：原始笔记 → BERTopic 主题 → 五大AI主题"
            f"<br><sup>有效笔记 {total_notes:,} 条；中列展示文档数 Top-{max_micro_nodes} 微观主题，其余并入「其他微观主题」</sup>"
        ),
        font=dict(size=11, family="Microsoft YaHei, SimHei, Arial"),
        width=1500,
        height=820,
        margin=dict(l=24, r=24, t=88, b=24),
    )

    html_path = VIS_OUTPUT_DIR / "11_topic_sankey.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info("  ✅ 图11 HTML 已保存")

    try:
        png_path = VIS_OUTPUT_DIR / "11_topic_sankey.png"
        fig.write_image(str(png_path), scale=2)
        logger.info("  ✅ 图11 PNG 已保存")
    except Exception as e:
        logger.warning(f"  ⚠️ 图11 PNG 导出失败（可能未安装 kaleido）: {e}")

# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("  BERTopic 模型训练效果 & 评估效果 可视化看板")
    print("=" * 60)

    # ====== 第1步：设置统一论文风主题 ======
    apply_publication_theme()

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
    plot_topic_sankey(df_result, topic_model)         # 图11

    print("\n" + "=" * 60)
    print("  ✅ 全部可视化图表生成完毕！")
    print(f"  📂 输出目录: {VIS_OUTPUT_DIR}")
    print("=" * 60)
