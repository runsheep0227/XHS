import pandas as pd
import numpy as np
import os
import sys
import tempfile
import torch
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings("ignore")

# ==================== 环境与镜像配置 ====================
os.environ['NUMBA_CACHE_DIR'] = str(Path(tempfile.gettempdir()) / 'numba_cache')
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 核心算法库 ====================
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================== 路径与基础配置 ====================
WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR / "bertopic_ready_data"
OUTPUT_DIR = WORK_DIR / "bertopic_results_optimized"
MODEL_DIR = OUTPUT_DIR / "saved_model"
LOG_DIR = WORK_DIR / "logs"

INPUT_FILE = DATA_DIR / "bertopic_train_data.csv"
RANDOM_STATE = 42

# ==================== 【优化1】调整目标主题数 ====================
TARGET_MICRO_TOPICS_MIN = 20
TARGET_MICRO_TOPICS_MAX = 40

# ==================== 【优化2】扩充宏观锚点与种子词 ====================
MACRO_ANCHORS = {
    "AI内容创作": {
        "definition": "核心词：绘画 插画 视频 建模 设计 生成 创作 写小说 网文 写作 音乐 短剧 脚本 IP 形象 Midjourney Stable Diffusion SD MJ ComfyUI 工作流 LoRA ControlNet。场景定义：利用人工智能生成图像、视频、音频、三维模型或文学作品的直接创作过程与作品展示。",
        "keywords": ["绘画", "插画", "视频", "建模", "生成", "创作", "小说", "网文",
                     "写作", "音乐", "短剧", "脚本", "Midjourney", "SD", "ComfyUI",
                     "工作流", "LoRA", "ControlNet", "出图", "生图", "画风", "IP",
                     "形象设计", "头像", "壁纸", "表情包", "动漫", "二次元", "摄影",
                     "修图", "海报", "Logo", "排版", "UI设计", "三维", "C4D", "Blender"]
    },
    "AI应用与测评": {
        "definition": "核心词：测评 推荐 对比 红黑榜 体验 智能体 机器人 星野 猫箱 APP 硬件 大模型 GPT Claude Kimi 通义 文心 DeepSeek。场景定义：对具体的AI软件、应用、平台、硬件设备或虚拟角色伴侣进行功能测试、优缺点评价与盘点。",
        "keywords": ["测评", "推荐", "对比", "体验", "智能体", "星野", "猫箱", "APP",
                     "硬件", "大模型", "GPT", "Claude", "Kimi", "通义", "文心", "DeepSeek",
                     "红黑榜", "避雷", "安利", "好用", "不好用", "功能", "更新", "版本",
                     "插件", "工具", "平台", "网站", "API", "免费", "付费", "会员",
                     "豆包", "天工", "讯飞", "百川", "智谱", "GLM", "Gemini", "Copilot"]
    },
    "AI学习教程": {
        "definition": "核心词：教程 学习 入门 提示词 指令 课程 培训 指南 考证 技巧 零基础 prompt 咒语 话术。场景定义：教授用户如何使用AI工具的实操指南、指令词编写方法、学习路线与教育培训内容。",
        "keywords": ["教程", "学习", "入门", "提示词", "指令", "课程", "培训", "指南",
                     "考证", "技巧", "零基础", "prompt", "咒语", "话术", "怎么用",
                     "手把手", "保姆级", "详细", "步骤", "方法", "学会", "小白",
                     "新手", "进阶", "精通", "实操", "练习", "模板", "框架",
                     "公式", "结构", "万能", "高效", "准确", "精准", "提问", "对话"]
    },
    "AI赋能工作生活": {
        "definition": "核心词：效率 办公 搞钱 副业 变现 论文 翻译 简历 英语 口语 旅游 算命 求职 PPT Excel 数据分析。场景定义：将AI作为辅助工具解决工作效率提升、学术研究、赚钱变现或日常生活具体场景的问题。",
        "keywords": ["效率", "办公", "副业", "变现", "论文", "翻译", "简历", "英语",
                     "口语", "旅游", "算命", "求职", "PPT", "Excel", "数据分析",
                     "赚钱", "搞钱", "收入", "月入", "兼职", "自由职业", "自媒体",
                     "运营", "文案", "营销", "客服", "自动化", "流程", "提效",
                     "汇报", "周报", "总结", "邮件", "合同", "法律", "财务",
                     "代码", "编程", "开发", "debug", "程序", "Python", "爬虫"]
    },
    "AI社会反思": {
        "definition": "核心词：失业 焦虑 取代 版权 侵权 法律 伦理 诈骗 深度伪造 威胁 监管 维权 画师 设计师 失业潮。场景定义：反思AI带来的负面影响，包括画师维权、人类职业被取代的恐慌、数据隐私与科技伦理争议。",
        "keywords": ["失业", "焦虑", "取代", "版权", "侵权", "法律", "伦理", "诈骗",
                     "深度伪造", "维权", "画师", "设计师", "隐私", "数据安全", "歧视",
                     "偏见", "公平", "透明", "AI画图", "AI写作", "抄袭", "剽窃",
                     "原创", "知识产权", "人类", "机器", "未来", "趋势", "影响"]
    }
}
MACRO_TOPIC_NAMES = list(MACRO_ANCHORS.keys())

# 【优化2】从扩充后的关键词中提取种子词，每个类别取前20个
SEED_WORDS = [anchor["keywords"][:20] for anchor in MACRO_ANCHORS.values()]

# ==================== 日志初始化 ====================
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "bertopic_optimized.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


# ==================== 【优化3】数据加载增强 ====================
def load_data(file_path: Path):
    logger.info(f"📂 正在加载数据: {file_path}")
    df = pd.read_csv(file_path).dropna(subset=["cleaned_full_text", "cleaned_seg_text"])
    df = df[df["cleaned_seg_text"].str.strip() != ""]
    df = df[df["cleaned_full_text"].str.strip() != ""]

    # 【新增】数据质量检查
    logger.info(f"📊 有效数据量: {len(df)} 条")
    avg_len = df["cleaned_full_text"].str.len().mean()
    logger.info(f"📊 平均文本长度: {avg_len:.0f} 字符")

    # 【新增】过滤过短文本（<10字符的文本无法提供有效语义）
    short_mask = df["cleaned_full_text"].str.len() < 10
    if short_mask.sum() > 0:
        logger.warning(f"⚠️ 发现 {short_mask.sum()} 条过短文本（<10字符），将被过滤")
        df = df[~short_mask]

    return df["note_id"].tolist(), df["cleaned_full_text"].tolist(), df["cleaned_seg_text"].tolist()


# ==================== 【优化4】初始化流水线（最终调整） ====================
def initialize_pipeline():
    logger.info("🔧 初始化最终优化版 BERTopic 流水线...")

    # 1. 文本向量化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🤖 正在加载 BAAI/bge-large-zh-v1.5 模型，使用设备: {device}")
    embedding_model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device=device)
    embedding_model.max_seq_length = 512

    # 2. UMAP降维：【最终调整】平衡信息保留与降维效果
    umap_model = UMAP(
        n_neighbors=12,        # 【最终调整】降低到12，更关注局部结构
        n_components=12,       # 【最终调整】适度压缩，保留关键信息
        min_dist=0.0,
        metric='cosine',
        random_state=RANDOM_STATE
    )

    # 3. HDBSCAN：【最终调整】大幅降低门槛，让绝大多数文档被归类
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,    # 【最终调整】降低到5，允许极小主题
        min_samples=1,         # 【最终调整】降低到1，几乎任何点都可成为核心点
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=-1
    )

    # 4. 词频与TF-IDF：【最终调整】适度过滤
    vectorizer_model = CountVectorizer(
        min_df=2,              # 【最终调整】降低到2，允许更低频词
        max_df=0.92,           # 【最终调整】提高到0.92，保留更多高频词
        ngram_range=(1, 2)
    )
    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True,
        reduce_frequent_words=True
    )

    return embedding_model, umap_model, hdbscan_model, vectorizer_model, ctfidf_model


# ==================== 【优化5】训练与多策略优化（最终版） ====================
def train_and_optimize(full_texts, seg_texts, pipeline_models):
    emb_mdl, umap_mdl, hdb_mdl, vec_mdl, ctfidf_mdl = pipeline_models

    # ================================================================
    # 阶段1：引导式BERTopic聚类
    # ================================================================
    logger.info("🚀 阶段1：开始执行 Guided BERTopic (先验引导聚类)...")

    topic_model = BERTopic(
        embedding_model=emb_mdl,
        umap_model=umap_mdl,
        hdbscan_model=hdb_mdl,
        vectorizer_model=vec_mdl,
        ctfidf_model=ctfidf_mdl,
        seed_topic_list=SEED_WORDS,
        nr_topics=TARGET_MICRO_TOPICS_MAX,  # 先设上限，后续自动收敛
        verbose=True,
        calculate_probabilities=True
    )

    topics, probs = topic_model.fit_transform(full_texts)

    noise_count_initial = topics.count(-1)
    total_count = len(topics)
    noise_ratio = noise_count_initial / total_count * 100
    logger.info(f"📊 初始聚类完成！噪声数据: {noise_count_initial} 条 ({noise_ratio:.1f}%)")
    logger.info(f"📊 发现有效主题数: {len(set(topics)) - (1 if -1 in topics else 0)} 个")

    # ================================================================
    # 阶段2：【最终调整】离群点回收策略（大幅提高阈值）
    # ================================================================
    logger.info("🔄 阶段2：执行离群点回收（只回收高置信度文档）...")

    if noise_count_initial > 0:
        # 策略A：基于文档嵌入相似度回收 - 大幅提高阈值
        logger.info("   → 策略A: 基于嵌入向量相似度回收（高阈值）...")
        topics_after_embed = topic_model.reduce_outliers(
            full_texts, topics,
            strategy="embeddings",
            threshold=0.6  # 【最终调整】大幅提高到0.6，只回收非常明确的匹配
        )

        # 策略B：基于c-TF-IDF分布回收 - 大幅提高阈值
        logger.info("   → 策略B: 基于c-TF-IDF分布回收（高阈值）...")
        topics_after_ctfidf = topic_model.reduce_outliers(
            full_texts, topics_after_embed,
            strategy="c-tf-idf",
            threshold=0.5  # 【最终调整】大幅提高到0.5
        )

        # 策略C：基于概率分布回收 - 保持中等阈值
        logger.info("   → 策略C: 基于概率分布兜底回收...")
        topics_final = topic_model.reduce_outliers(
            full_texts, topics_after_ctfidf,
            strategy="distributions",
            threshold=0.2  # 【最终调整】提高到0.2
        )

        # 更新主题分配
        topic_model.update_topics(full_texts, topics=topics_final)
        topics = topics_final

        noise_count_final = topics.count(-1)
        noise_ratio_final = noise_count_final / total_count * 100
        recovered = noise_count_initial - noise_count_final
        logger.info(f"✅ 离群点回收完成！回收 {recovered} 条，剩余噪声: {noise_count_final} 条 ({noise_ratio_final:.1f}%)")
    else:
        logger.info("✅ 初始聚类无噪声数据，跳过回收步骤")

    # ================================================================
    # 阶段3：【最终调整】主题合并（减少碎片化）
    # ================================================================
    logger.info("🎯 阶段3：执行主题合并，减少碎片化...")

    # 获取主题信息
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1].copy()

    if len(valid_topics) > TARGET_MICRO_TOPICS_MAX:
        logger.info(f"   发现 {len(valid_topics)} 个主题，超过上限 {TARGET_MICRO_TOPICS_MAX}，执行合并...")

        # 计算主题间的相似度
        topic_vectors = []
        topic_ids = []
        for tid in valid_topics['Topic']:
            # 使用主题的关键词构建向量
            topic_words = topic_model.get_topic(tid)
            if topic_words:
                words_str = " ".join([w[0] for w in topic_words[:15]])
                topic_vectors.append(emb_mdl.encode([words_str], normalize_embeddings=True)[0])
                topic_ids.append(tid)

        if topic_vectors:
            topic_sim_matrix = cosine_similarity(topic_vectors)

            # 合并相似主题
            merged_topics = set()
            for i, tid1 in enumerate(topic_ids):
                if tid1 in merged_topics:
                    continue
                for j, tid2 in enumerate(topic_ids):
                    if i != j and tid2 not in merged_topics:
                        if topic_sim_matrix[i][j] > 0.7:  # 相似度阈值
                            # 合并主题：将tid2的文档分配到tid1
                            for idx, topic_id in enumerate(topics):
                                if topic_id == tid2:
                                    topics[idx] = tid1
                            merged_topics.add(tid2)
                            logger.info(f"   合并主题 T{tid2} -> T{tid1} (相似度: {topic_sim_matrix[i][j]:.3f})")

        # 重新计算主题分配
        topic_model.update_topics(full_texts, topics=topics)
        logger.info(f"   主题合并完成，当前主题数: {len(set(topics)) - (1 if -1 in topics else 0)}")

    # ================================================================
    # 阶段4：【最终调整】微观→宏观语义映射（调整权重）
    # ================================================================
    logger.info("🗺️ 阶段4：执行微观→宏观语义映射...")

    # 获取5大宏观锚点的向量
    macro_descriptions = [anchor["definition"] for anchor in MACRO_ANCHORS.values()]
    macro_embeddings = emb_mdl.encode(macro_descriptions, normalize_embeddings=True)

    # 同时使用关键词向量作为辅助判断
    macro_keyword_texts = [" ".join(anchor["keywords"]) for anchor in MACRO_ANCHORS.values()]
    macro_kw_embeddings = emb_mdl.encode(macro_keyword_texts, normalize_embeddings=True)

    mapping_dict = {}
    confidence_dict = {}
    topic_info = topic_model.get_topic_info()

    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            mapping_dict[topic_id] = "噪声数据(Outliers)"
            confidence_dict[topic_id] = 0.0
            continue

        # 提取微观主题的Top20核心词
        top_words = " ".join([word for word, weight in topic_model.get_topic(topic_id)[:20]])

        # 提取微观主题最具代表性的3篇原味笔记
        rep_docs = topic_model.get_representative_docs(topic_id)
        if not rep_docs:
            rep_docs = [""]

        # 组合成微观锚点句
        micro_anchor = f"核心词：{top_words}。场景内容：{' '.join(rep_docs)[:400]}"
        micro_vector = emb_mdl.encode([micro_anchor], normalize_embeddings=True)

        # 【最终调整】双重相似度计算：关键词权重提高到0.7
        semantic_sims = cosine_similarity(micro_vector, macro_embeddings)[0]
        keyword_sims = cosine_similarity(micro_vector, macro_kw_embeddings)[0]

        # 【最终调整】加权融合：关键词权重0.7，语义权重0.3
        combined_sims = 0.3 * semantic_sims + 0.7 * keyword_sims

        best_macro_idx = np.argmax(combined_sims)
        best_confidence = combined_sims[best_macro_idx]

        mapping_dict[topic_id] = MACRO_TOPIC_NAMES[best_macro_idx]
        confidence_dict[topic_id] = float(best_confidence)

    # 【优化】低置信度主题的二次检查
    low_conf_topics = [tid for tid, conf in confidence_dict.items()
                       if tid != -1 and conf < 0.35]
    if low_conf_topics:
        logger.warning(f"⚠️ 发现 {len(low_conf_topics)} 个低置信度主题（<0.35），可能需要人工审查")
        for tid in low_conf_topics:
            logger.warning(f"   主题 {tid} -> {mapping_dict[tid]} (置信度: {confidence_dict[tid]:.3f})")

    return topic_model, topics, mapping_dict, confidence_dict


# ==================== 【优化6】结果保存（增强版） ====================
def save_results(topic_model, topics, note_ids, full_texts, seg_texts, mapping_dict, confidence_dict):
    logger.info("💾 阶段5：正在保存结果与本地微调模型...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 保存模型
    try:
        topic_model.save(str(MODEL_DIR), serialization="safetensors", save_ctfidf=True)
        logger.info(f"🏆 模型已保存至：{MODEL_DIR}")
    except Exception as e:
        logger.error(f"❌ 模型保存失败: {e}")

    # 2. 提取微观主题关键词
    micro_keywords = {}
    micro_top_words_full = {}
    for tid in mapping_dict.keys():
        if tid == -1:
            micro_keywords[tid] = "无明确语义"
            micro_top_words_full[tid] = ""
        else:
            topic_words = topic_model.get_topic(tid)
            micro_keywords[tid] = ", ".join([w[0] for w in topic_words[:8]])
            micro_top_words_full[tid] = ", ".join([w[0] for w in topic_words[:20]])

    # 3. 输出分类结果CSV
    df_result = pd.DataFrame({
        "note_id": note_ids,
        "content": full_texts,
        "segmented_text": seg_texts,
        "micro_topic_id": topics,
        "micro_topic_keywords": [micro_keywords[t] for t in topics],
        "micro_topic_full_keywords": [micro_top_words_full[t] for t in topics],
        "macro_topic_name": [mapping_dict[t] for t in topics],
        "mapping_confidence": [confidence_dict.get(t, 0.0) for t in topics]
    })

    output_csv = OUTPUT_DIR / "final_pro_topics.csv"
    df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"📄 分类结果CSV已保存至：{output_csv}")

    # 4. 生成详细的层级审查报告
    report_file = OUTPUT_DIR / "pro_mapping_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  AIGC 小红书笔记 (最终优化版) 聚类映射报告\n")
        f.write(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # 总览统计
        total = len(topics)
        noise_count = topics.count(-1)
        valid_count = total - noise_count
        unique_micro = len(set(topics)) - (1 if -1 in topics else 0)
        unique_macro = len(set(mapping_dict.values()) - {"噪声数据(Outliers)"})

        f.write(f"📊 总览统计:\n")
        f.write(f"   总数据量: {total} 条\n")
        f.write(f"   有效分类: {valid_count} 条 ({valid_count/total*100:.1f}%)\n")
        f.write(f"   噪声数据: {noise_count} 条 ({noise_count/total*100:.1f}%)\n")
        f.write(f"   微观主题数: {unique_micro} 个\n")
        f.write(f"   宏观大类数: {unique_macro} 个\n\n")

        # 各宏观大类详情
        for macro in MACRO_TOPIC_NAMES:
            belonging_micros = [(tid, confidence_dict.get(tid, 0.0)) for tid, m_name in mapping_dict.items()
                    if m_name == macro and tid != -1]
            belonging_micros.sort(key=lambda x: topics.count(x[0]), reverse=True)

            macro_count = sum(topics.count(tid) for tid, _ in belonging_micros)
            f.write(f"{'='*60}\n")
            f.write(f"🌟 宏观大类：【{macro}】 (共 {macro_count} 条, {macro_count/total*100:.1f}%)\n")
            f.write(f"{'='*60}\n")

            for tid, conf in belonging_micros:
                count = topics.count(tid)
                f.write(f"   ├─ 子主题 {tid} (频次:{count}, 置信度:{conf:.3f})\n")
                f.write(f"   │  关键词: {micro_keywords[tid]}\n")

                # 输出该子主题的2篇代表性笔记摘要
                rep_docs = topic_model.get_representative_docs(tid)
                for i, doc in enumerate(rep_docs[:2]):
                    doc_preview = doc[:80].replace('\n', ' ')
                    f.write(f"   │  样例{i+1}: {doc_preview}...\n")
                f.write(f"   │\n")

            if not belonging_micros:
                f.write(f"   (无子主题)\n")
            f.write("\n")

        # 噪声数据统计
        f.write(f"{'='*60}\n")
        f.write(f"🗑️ 隔离的边缘/无效噪声数据 (Topic -1): {noise_count} 条\n")
        f.write(f"{'='*60}\n")

        # 低置信度主题提醒
        low_conf = [(tid, mapping_dict[tid], confidence_dict[tid])
                    for tid in confidence_dict if tid != -1 and confidence_dict[tid] < 0.35]
        if low_conf:
            f.write(f"\n⚠️ 低置信度主题（建议人工审查）:\n")
            for tid, macro, conf in low_conf:
                f.write(f"   主题 {tid} -> {macro} (置信度: {conf:.3f})\n")

    logger.info(f"✅ 详细审查报告已保存至：{report_file}")

    # 5. 输出主题分布统计CSV
    stats_data = []
    for macro in MACRO_TOPIC_NAMES:
        belonging = [tid for tid, m in mapping_dict.items() if m == macro and tid != -1]
        for tid in belonging:
            stats_data.append({
                "macro_topic": macro,
                "micro_topic_id": tid,
                "note_count": topics.count(tid),
                "keywords": micro_keywords[tid],
                "confidence": confidence_dict.get(tid, 0.0)
            })

    df_stats = pd.DataFrame(stats_data).sort_values(
        ["macro_topic", "note_count"], ascending=[True, False]
    )
    stats_csv = OUTPUT_DIR / "topic_distribution_stats.csv"
    df_stats.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    logger.info(f"📊 主题分布统计已保存至：{stats_csv}")


# ==================== 【优化7】聚类质量评估函数 ====================
def evaluate_clustering(topic_model, topics, mapping_dict, full_texts):
    """输出聚类质量的量化评估指标"""
    logger.info("📈 正在评估聚类质量...")
    total = len(topics)
    noise_count = topics.count(-1)

    print("\n" + "=" * 50)
    print("  聚类质量评估报告")
    print("=" * 50)

    # 1. 覆盖率 = 非噪声数据 / 总数据
    coverage = (total - noise_count) / total * 100
    print(f"  📊 数据覆盖率: {coverage:.1f}%")
    print(f"     (非噪声数据 {total - noise_count} / 总数据 {total})")

    # 2. 主题分布均匀度
    valid_topics = [t for t in set(topics) if t != -1]
    if valid_topics:
        counts = [topics.count(t) for t in valid_topics]
        max_count = max(counts)
        min_count = min(counts)
        avg_count = np.mean(counts)
        print(f"  📊 主题分布: 最大={max_count}, 最小={min_count}, 平均={avg_count:.0f}")
        print(f"     (比值 最大/最小 = {max_count/min_count:.1f}，理想值<10)")

    # 3. 宏观类别覆盖
    macro_coverage = set(mapping_dict.values()) - {"噪声数据(Outliers)"}
    print(f"  📊 宏观类别覆盖: {len(macro_coverage)} / {len(MACRO_TOPIC_NAMES)}")
    uncovered = set(MACRO_TOPIC_NAMES) - macro_coverage
    if uncovered:
        print(f"     ⚠️ 未覆盖的类别: {uncovered}")

    # 4. 主题纯度（宏观类别内聚性）
    for macro in MACRO_TOPIC_NAMES:
        belonging = [tid for tid, m in mapping_dict.items() if m == macro and tid != -1]
        total_in_macro = sum(topics.count(tid) for tid in belonging)
        print(f"  📊 {macro}: {len(belonging)}个子主题, {total_in_macro}条数据")

    print("=" * 50 + "\n")


# ==================== 主函数 ====================
if __name__ == "__main__":
    # 加载数据
    note_ids, full_texts, seg_texts = load_data(INPUT_FILE)

    # 初始化流水线
    pipeline_models = initialize_pipeline()

    # 训练与优化
    topic_model, topics, mapping_dict, confidence_dict = train_and_optimize(
        full_texts, seg_texts, pipeline_models
    )

    # 质量评估
    evaluate_clustering(topic_model, topics, mapping_dict, full_texts)

    # 保存结果
    save_results(
        topic_model, topics, note_ids, full_texts, seg_texts,
        mapping_dict, confidence_dict
    )

    logger.info("🎉 全部流程执行完毕！")
