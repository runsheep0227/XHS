import pandas as pd
import numpy as np
import os
import sys
import tempfile
import torch
from pathlib import Path
from datetime import datetime
import logging
import time
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
RANDOM_STATE = 13
TARGET_MICRO_TOPICS_MAX = 30

# ==================== 扩充宏观锚点与种子词 ====================
MACRO_ANCHORS = {
    "AI内容创作": {
        "definition": "核心词：绘画 插画 视频 建模 设计 生成 创作 写小说 网文 写作 音乐 短剧 脚本 IP 形象 Midjourney Stable Diffusion SD MJ ComfyUI 工作流 LoRA ControlNet。场景定义：利用人工智能生成图像、视频、音频、三维模型或文学作品的直接创作过程与作品展示。",
        "keywords": list(dict.fromkeys([
            "绘画", "插画", "视频", "建模", "生成", "创作", "小说", "网文",
            "写作", "音乐", "短剧", "脚本", "Midjourney", "SD", "ComfyUI",
            "工作流", "LoRA", "ControlNet", "出图", "生图", "画风", "IP",
            "形象设计", "头像", "壁纸", "表情包", "动漫", "二次元", "摄影",
            "修图", "海报", "Logo", "排版", "UI设计", "三维", "C4D", "Blender"
        ]))
    },
    "AI应用与测评": {
        "definition": "核心词：测评 推荐 对比 红黑榜 体验 智能体 机器人 星野 猫箱 APP 硬件 大模型 GPT Claude Kimi 通义 文心 DeepSeek 即梦 造点 liblib。场景定义：对具体的AI软件、应用、平台、硬件设备或虚拟角色伴侣进行功能测试、优缺点评价与盘点。",
        "keywords": list(dict.fromkeys([
            "测评", "推荐", "对比", "体验", "智能体", "星野", "猫箱", "APP",
            "Cursor", "大模型", "GPT", "Claude", "Kimi", "通义", "文心", "DeepSeek",
            "红黑榜", "避雷", "安利", "好用", "不好用", "功能", "更新", "版本",
            "插件", "工具", "平台", "网站", "API", "免费", "付费", "会员",
            "豆包", "天工", "讯飞", "百川", "智谱", "GLM", "Gemini", "Copilot",
            "即梦", "造点", "liblib", "nanobanana pro", "软件","龙虾","openai","openclaw"
            "千问","Qwen"
        ]))
    },
    "AI学习教程": {
        "definition": "核心词：教程 学习 入门 提示词 指令 课程 培训 指南 考证 技巧 零基础 prompt 咒语 话术。场景定义：教授用户如何使用AI工具的实操指南、指令词编写方法、学习路线与教育培训内容。",
        "keywords": list(dict.fromkeys([
            "教程", "学习", "入门", "提示词", "指令", "课程", "培训", "指南",
            "考证", "技巧", "零基础", "prompt", "咒语", "话术", "怎么用",
            "手把手", "保姆级", "详细", "步骤", "方法", "学会", "小白",
            "新手", "进阶", "精通", "实操", "练习", "模板", "框架",
            "公式", "结构", "万能", "高效", "准确", "精准", "提问", "对话"
        ]))
    },
    "AI赋能工作生活": {
        "definition": "核心词：效率 办公 搞钱 副业 变现 论文 翻译 简历 英语 口语 旅游 算命 求职 PPT Excel 数据分析。场景定义：将AI作为辅助工具解决工作效率提升、学术研究、赚钱变现或日常生活具体场景的问题。",
        "keywords": list(dict.fromkeys([
            "效率", "办公", "副业", "变现", "论文", "翻译", "简历", "英语",
            "口语", "旅游", "算命", "求职", "PPT", "Excel", "数据分析",
            "赚钱", "搞钱", "收入", "月入", "兼职", "自由职业", "自媒体",
            "运营", "文案", "营销", "客服", "自动化", "流程", "提效",
            "汇报", "周报", "总结", "邮件", "合同", "法律", "财务",
            "代码", "编程", "开发", "debug", "程序", "Python", "爬虫"
        ]))
    },
    "AI社会反思": {
        "definition": "核心词：失业 版权 伦理 诈骗 深度伪造 维权 数字鸿沟 学术造假 数据投毒 能源消耗 失控。场景定义：反思AI带来的负面与深层影响，包括职业取代恐慌、版权隐私争议、深度伪造造谣、教育作弊与思考能力退化、数字鸿沟与资本垄断，以及AI算力带来的能源环境危机。",
        "keywords": list(dict.fromkeys([
            "失业", "焦虑", "取代", "版权", "侵权", "维权", "画师", "裁员", "抄袭", "原创",
            "法律", "伦理", "知识产权", "人类", "职业", "失业潮",
            "诈骗", "欺骗", "深度伪造", "造谣", "黄谣", "换脸", "虚假信息", "伪造",
            "作弊", "代写", "学术造假", "思考能力", "退化", "独立思考", "洗稿",
            "沉迷", "异化", "情感剥削", "丧失人性", "信息茧房", "偏见", "歧视",
            "数字鸿沟", "垄断", "巨头", "贫富差距", "资本", "不公平", "透明",
            "隐私", "数据安全", "投毒", "模型崩塌", "数据污染", "未经授权",
            "耗电", "能源", "环境", "碳排放", "失控", "觉醒", "硅基生物", "威胁"
        ]))
    }
}
MACRO_TOPIC_NAMES = list(MACRO_ANCHORS.keys())
SEED_WORDS = [anchor["keywords"][:60] for anchor in MACRO_ANCHORS.values()]

KEYWORD_TO_MACROS = {}
for macro, anchor in MACRO_ANCHORS.items():
    seen = set()
    for idx, kw in enumerate(anchor["keywords"]):
        if kw in seen: continue
        seen.add(kw)
        weight = 3.0 if idx < 15 else (2.0 if idx < 35 else 1.5)
        if kw not in KEYWORD_TO_MACROS: KEYWORD_TO_MACROS[kw] = {}
        KEYWORD_TO_MACROS[kw][macro] = weight

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

# ==================== 数据加载 ====================
def load_data(file_path: Path):
    logger.info(f"📂 正在加载数据: {file_path}")
    df = pd.read_csv(file_path).dropna(subset=["cleaned_full_text", "cleaned_seg_text"])
    df = df[df["cleaned_seg_text"].str.strip() != ""]
    df = df[df["cleaned_full_text"].str.strip() != ""]

    logger.info(f"📊 原始有效数据量: {len(df)} 条")
    short_mask = df["cleaned_full_text"].str.len() < 5
    if short_mask.sum() > 0:
        logger.warning(f"⚠️ 过滤 {short_mask.sum()} 条过短文本（<5字符）")
        df = df[~short_mask]

    if len(df) == 0:
        logger.error("❌ 过滤后无有效数据，程序终止")
        sys.exit(1)

    return df["note_id"].tolist(), df["cleaned_full_text"].tolist(), df["cleaned_seg_text"].tolist()

def space_tokenizer(text):
    return text.split() if isinstance(text, str) else[]

# ==================== 初始化流水线 (已优化防膨胀) ====================
def initialize_pipeline(data_size):
    logger.info("🔧 初始化 BERTopic 流水线...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🤖 使用设备: {device}")

    embedding_model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device=device)
    embedding_model.max_seq_length = 512

    umap_model = UMAP(n_neighbors=30, n_components=15, min_dist=0.1, metric='cosine', random_state=RANDOM_STATE)
    
    dynamic_min_cluster = max(25, int((data_size / TARGET_MICRO_TOPICS_MAX) * 0.25))
    logger.info(f"📐 动态设定 HDBSCAN 最小聚类体积: {dynamic_min_cluster}")

    # 【降噪调整1】降低孤立点门槛 (从 //4 降到 //5)，使得稍微稀疏的数据也能成团而不是直接变噪声
    min_samples_val = max(5, dynamic_min_cluster // 5) 
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=dynamic_min_cluster,
        min_samples=min_samples_val, 
        metric='euclidean', 
        cluster_selection_method='eom', 
        prediction_data=True, 
        core_dist_n_jobs=-1
    )
    
    vectorizer_model = CountVectorizer(tokenizer=space_tokenizer, min_df=3, max_df=0.90, ngram_range=(1, 2))
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

    return embedding_model, umap_model, hdbscan_model, vectorizer_model, ctfidf_model

# ==================== 计算映射与置信度 ====================
def compute_mappings(topic_model, emb_mdl):
    mapping_dict = {}
    confidence_dict = {}
    topic_info = topic_model.get_topic_info()
    
    macro_descriptions = [anchor["definition"] for anchor in MACRO_ANCHORS.values()]
    macro_embeddings = emb_mdl.encode(macro_descriptions, normalize_embeddings=True)

    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            mapping_dict[topic_id] = "噪声数据(Outliers)"
            confidence_dict[topic_id] = 0.0
            continue

        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            mapping_dict[topic_id] = "噪声数据(Outliers)"
            confidence_dict[topic_id] = 0.0
            continue

        topic_keywords = [w[0] for w in topic_words[:30]]
        topic_weights =[w[1] for w in topic_words[:30]]

        # Layer 1: 关键词加权
        macro_weighted_scores = {macro: 0.0 for macro in MACRO_TOPIC_NAMES}
        total_keyword_weight = 0.0

        for kw, tfidf_weight in zip(topic_keywords[:15], topic_weights[:15]):
            if kw in KEYWORD_TO_MACROS:
                for macro, kw_importance in KEYWORD_TO_MACROS[kw].items():
                    score = tfidf_weight * kw_importance
                    macro_weighted_scores[macro] += score
                    total_keyword_weight += score

        if total_keyword_weight > 0:
            for macro in macro_weighted_scores:
                macro_weighted_scores[macro] /= total_keyword_weight
        else:
            for macro in macro_weighted_scores:
                macro_weighted_scores[macro] = 1.0 / len(MACRO_TOPIC_NAMES)

        # Layer 2: 语义相似度匹配
        rep_docs = topic_model.get_representative_docs(topic_id)
        rep_docs = rep_docs if isinstance(rep_docs, list) else[]
        top_words_str = " ".join(topic_keywords[:15])
        docs_str = " ".join(rep_docs)[:300]
        micro_anchor = f"主题关键词：{top_words_str}。内容摘要：{docs_str}"

        if not micro_anchor.strip():
            semantic_scores = {macro: 0.0 for macro in MACRO_TOPIC_NAMES}
        else:
            micro_vector = emb_mdl.encode([micro_anchor], normalize_embeddings=True)
            semantic_sims = cosine_similarity(micro_vector, macro_embeddings)[0]
            semantic_scores = {macro: max(0.0, float(s)) for macro, s in zip(MACRO_TOPIC_NAMES, semantic_sims)}

        # Layer 3: 融合计算 
        combined_scores = {}
        for macro in MACRO_TOPIC_NAMES:
            combined_scores[macro] = (semantic_scores[macro] * 0.6 + macro_weighted_scores[macro] * 0.4)

        best_macro = max(combined_scores, key=combined_scores.get)
        best_score = combined_scores[best_macro]
        sorted_scores = sorted(combined_scores.values(), reverse=True)

        final_confidence = min(best_score * (1.0 + (sorted_scores[0] - sorted_scores[1])), 1.0) if len(sorted_scores) > 1 else best_score
        mapping_dict[topic_id] = best_macro
        confidence_dict[topic_id] = round(final_confidence, 4)

    return mapping_dict, confidence_dict

# ==================== 训练与优化 ====================
def train_and_optimize(full_texts, seg_texts, pipeline_models):
    start_time = time.time()
    emb_mdl, umap_mdl, hdb_mdl, vec_mdl, ctfidf_mdl = pipeline_models

    logger.info("🚀 阶段0：预计算文档完整语义 Embedding...")
    embeddings = emb_mdl.encode(full_texts, show_progress_bar=True, batch_size=32)

    logger.info("🚀 阶段1：开始执行 Guided BERTopic 聚类...")
    topic_model = BERTopic(
        embedding_model=emb_mdl,
        umap_model=umap_mdl,
        hdbscan_model=hdb_mdl,
        vectorizer_model=vec_mdl,
        ctfidf_model=ctfidf_mdl,
        seed_topic_list=SEED_WORDS,
        verbose=True,
        calculate_probabilities=False
    )

    topics, _ = topic_model.fit_transform(seg_texts, embeddings=embeddings)
    initial_topic_count = len(set(topics)) - (1 if -1 in topics else 0)
    logger.info(f"📊 初始主题数: {initial_topic_count} 个")

    if initial_topic_count > TARGET_MICRO_TOPICS_MAX:
        logger.info(f"🔄 阶段2.1：微调合并控制主题数不超过 {TARGET_MICRO_TOPICS_MAX}...")
        topic_model.reduce_topics(seg_texts, nr_topics=TARGET_MICRO_TOPICS_MAX)
        topics = topic_model.topics_

    # 【降噪调整2】适度放宽离群点回收门槛 (0.75 -> 0.55)，大幅度挽回有效数据
    logger.info("🔄 阶段2.2：执行离群点归队回收（适中阈值: 0.55）...")
    topics_recovered = topic_model.reduce_outliers(seg_texts, topics, strategy="embeddings", embeddings=embeddings, threshold=0.55)
    topic_model.update_topics(seg_texts, topics=topics_recovered)
    topics = topic_model.topics_

    mapping_dict, confidence_dict = compute_mappings(topic_model, emb_mdl)

    logger.info("🎯 阶段3：执行低置信度主题处理...")
    macro_topics = {}
    for tid, macro in mapping_dict.items():
        if tid != -1 and confidence_dict[tid] >= 0.5:
            macro_topics.setdefault(macro,[]).append(tid)

    merge_groups = {}
    for tid, conf in list(confidence_dict.items()):
        if tid == -1: continue

        # 【降噪调整3】降低弱主题直接死刑的门槛 (0.35 -> 0.30)
        if conf < 0.30:
            mapping_dict[tid] = "噪声数据(Outliers)"
            continue

        if 0.30 <= conf < 0.5:
            current_macro = mapping_dict[tid]
            sibling_topics = macro_topics.get(current_macro,[])
            if not sibling_topics: continue

            tid_words = topic_model.get_topic(tid)
            if not tid_words: continue

            tid_vector = emb_mdl.encode([" ".join([w[0] for w in tid_words[:10]])], normalize_embeddings=True)
            best_sibling, best_sim = None, 0

            for sib_tid in sibling_topics:
                sib_words = topic_model.get_topic(sib_tid)
                if not sib_words: continue
                sib_vector = emb_mdl.encode([" ".join([w[0] for w in sib_words[:10]])], normalize_embeddings=True)
                sim = cosine_similarity(tid_vector, sib_vector)[0][0]
                if sim > best_sim:
                    best_sim, best_sibling = sim, sib_tid

            # 【降噪调整4】放宽合并门槛 (0.70 -> 0.65)，鼓励弱主题被同类吸收而非变噪声
            if best_sibling is not None and best_sim > 0.65:
                merge_groups.setdefault(best_sibling,[]).append(tid)
                logger.info(f"   🔗 计划合并弱主题 T{tid} -> T{best_sibling} (相似度: {best_sim:.3f})")
            else:
                mapping_dict[tid] = "噪声数据(Outliers)" 

    topics_to_merge = [[target] + sources for target, sources in merge_groups.items()]

    if topics_to_merge:
        logger.info(f"   执行最终合并，共涉及 {len(topics_to_merge)} 组聚类操作...")
        topic_model.merge_topics(seg_texts, topics_to_merge)
        topics = topic_model.topics_
        mapping_dict, confidence_dict = compute_mappings(topic_model, emb_mdl)

        for tid, conf in confidence_dict.items():
            if tid != -1 and conf < 0.30:
                mapping_dict[tid] = "噪声数据(Outliers)"

    logger.info(f"📊 训练完成，总耗时: {time.time() - start_time:.2f}秒")
    return topic_model, topics, mapping_dict, confidence_dict

# ==================== 结果评估与保存 ====================
def evaluate_clustering(topics, mapping_dict, confidence_dict):
    logger.info("📈 正在评估聚类质量...")
    total = len(topics)
    noise_count = sum(1 for t in topics if t == -1 or mapping_dict.get(t) == "噪声数据(Outliers)")

    print("\n" + "=" * 60)
    print("  聚类质量评估报告")
    print("=" * 60)
    print(f"  📊 有效数据覆盖率: {(total - noise_count) / total * 100:.1f}% ({total - noise_count}/{total})")

    valid_topics =[t for t in topics if t != -1 and mapping_dict.get(t) != "噪声数据(Outliers)"]
    if valid_topics:
        topic_counts = pd.Series(valid_topics).value_counts()
        print(f"  📊 分布健康度监控:")
        print(f"     最大主题占比: {topic_counts.max() / len(valid_topics) * 100:.1f}% (建议<40%)")
        print(f"     最小主题包含数: {topic_counts.min()} 条")

    for macro in MACRO_TOPIC_NAMES:
        belonging =[tid for tid, m in mapping_dict.items() if m == macro and tid != -1]
        total_in_macro = sum(topics.count(tid) for tid in belonging)
        avg_conf = np.mean([confidence_dict.get(tid, 0.0) for tid in belonging]) if belonging else 0
        print(f"  📌 {macro}: {len(belonging)}个子主题, 共 {total_in_macro}条, 平均置信度:{avg_conf:.3f}")

    valid_confidences =[
        confidence_dict[tid] for tid in confidence_dict 
        if tid != -1 and mapping_dict.get(tid) != "噪声数据(Outliers)"
    ]
    
    if valid_confidences:
        avg_confidence = np.mean(valid_confidences)
        high_conf = sum(1 for c in valid_confidences if c >= 0.7)
        medium_conf = sum(1 for c in valid_confidences if 0.5 <= c < 0.7)
        low_conf = sum(1 for c in valid_confidences if c < 0.5)

        print(f"\n  📊 有效分类置信度分布 (不含噪声):")
        print(f"     平均置信度: {avg_confidence:.3f}")
        print(f"     高置信(≥0.7): {high_conf}个 ({high_conf/len(valid_confidences)*100:.1f}%)")
        print(f"     中置信(0.5-0.7): {medium_conf}个 ({medium_conf/len(valid_confidences)*100:.1f}%)")
        print(f"     低置信(<0.5): {low_conf}个 ({low_conf/len(valid_confidences)*100:.1f}%)")
    print("=" * 60 + "\n")

def save_results(topic_model, topics, note_ids, full_texts, seg_texts, mapping_dict, confidence_dict):
    logger.info("💾 阶段5：正在保存完整结果...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        topic_model.save(str(MODEL_DIR), serialization="safetensors", save_ctfidf=True)
    except Exception as e:
        logger.error(f"❌ 模型保存失败: {e}")

    micro_keywords, micro_rep_docs = {}, {}
    for tid in set(topics):
        if tid == -1 or mapping_dict.get(tid) == "噪声数据(Outliers)":
            micro_keywords[tid], micro_rep_docs[tid] = "无明确语义",[]
        else:
            words = topic_model.get_topic(tid)
            micro_keywords[tid] = ", ".join([w[0] for w in words[:15]]) if words else ""
            rep_docs = topic_model.get_representative_docs(tid)
            micro_rep_docs[tid] = rep_docs if isinstance(rep_docs, list) else[]

    df_result = pd.DataFrame({
        "note_id": note_ids,
        "content": full_texts,
        "segmented_text": seg_texts,
        "micro_topic_id": topics,
        "micro_topic_keywords":[micro_keywords.get(t, "") for t in topics],
        "macro_topic_name":[mapping_dict.get(t, "未知") for t in topics],
        "mapping_confidence":[confidence_dict.get(t, 0.0) for t in topics],
        "is_noise":[(t == -1 or mapping_dict.get(t) == "噪声数据(Outliers)") for t in topics]
    })
    df_result.to_csv(OUTPUT_DIR / "final_pro_topics.csv", index=False, encoding="utf-8-sig")

    report_file = OUTPUT_DIR / "pro_mapping_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n  AIGC 小红书笔记 聚类映射报告\n" + "=" * 80 + "\n\n")

        total = len(topics)
        noise_count = sum(1 for t in topics if t == -1 or mapping_dict.get(t) == "噪声数据(Outliers)")
        valid_count = total - noise_count
        unique_micro = len([t for t in set(topics) if mapping_dict.get(t) != "噪声数据(Outliers)" and t != -1])

        f.write(f"📊 总览统计:\n")
        f.write(f"   总数据量: {total} 条\n")
        f.write(f"   有效分类: {valid_count} 条 ({valid_count/total*100:.1f}%)\n")
        f.write(f"   微观主题数: {unique_micro} 个\n\n")

        for macro in MACRO_TOPIC_NAMES:
            belonging_micros =[(tid, confidence_dict.get(tid, 0.0)) for tid, m_name in mapping_dict.items() if m_name == macro]
            belonging_micros.sort(key=lambda x: topics.count(x[0]), reverse=True)

            macro_count = sum(topics.count(tid) for tid, _ in belonging_micros)
            avg_macro_conf = np.mean([conf for _, conf in belonging_micros]) if belonging_micros else 0

            f.write(f"🌟 宏观大类：【{macro}】 (共 {macro_count} 条, 平均置信度: {avg_macro_conf:.3f})\n{'-'*70}\n")
            for tid, conf in belonging_micros:
                f.write(f"   ├─ 子主题 {tid} (频次:{topics.count(tid)}, 置信度:{conf:.3f})\n")
                f.write(f"   │  关键词: {micro_keywords.get(tid, '')}\n")
            f.write("\n")
            
        f.write(f"{'='*70}\n🗑️ 隔离的边缘/无效噪声数据: {noise_count} 条\n{'='*70}\n")
        
    logger.info(f"✅ 处理全流程结束，所有文件均已输出至：{OUTPUT_DIR}")

# ==================== 主函数 ====================
if __name__ == "__main__":
    note_ids, full_texts, seg_texts = load_data(INPUT_FILE)
    
    pipeline_models = initialize_pipeline(data_size=len(full_texts))
    
    topic_model, topics, mapping_dict, confidence_dict = train_and_optimize(full_texts, seg_texts, pipeline_models)
    evaluate_clustering(topics, mapping_dict, confidence_dict)
    save_results(topic_model, topics, note_ids, full_texts, seg_texts, mapping_dict, confidence_dict)
    
    logger.info("🎉 全部流程执行完毕！")