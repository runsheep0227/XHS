"""
bertopic_train.py - BERTopic主题建模训练脚本
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# ==================== 环境兼容处理 ====================
# 解决numba缓存问题
os.environ['NUMBA_CACHE_DIR'] = str(Path(tempfile.gettempdir()) / 'numba_cache')
os.environ['NUMBA_THREADING_LAYER'] = 'omp'

# ==================== 国内镜像 ====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== BERTopic ====================
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# ==================== 配置 ====================
WORK_DIR = Path(r"E:\document\PG\studio\content")
DATA_DIR = WORK_DIR / "bertopic_ready_data"
OUTPUT_DIR = WORK_DIR / "bertopic_results"
MODEL_DIR = OUTPUT_DIR / "bertopic_model"
LOG_DIR = WORK_DIR / "logs"

INPUT_FILE = DATA_DIR / "bertopic_train_data.csv"

# 随机种子
RANDOM_STATE = 42

# 嵌入模型
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 256

# UMAP降维参数
# 降维到2维便于K-Means聚类
UMAP_PARAMS = {
    "n_neighbors": 15,
    "n_components": 2,  # K-Means只需要2维
    "metric": "cosine",
    "min_dist": 0.1,     # 保持一定距离
    "random_state": RANDOM_STATE,
    "n_epochs": 200,
    "learning_rate": 0.01,
    "init": "spectral"
}

# ==================== 核心参数 ====================
N_TOPICS = 5  # 强制5个主题

# 主题名称映射（按您的5个维度）
TOPIC_NAMES = {
    0: "AI内容创作",
    1: "AI工具测评", 
    2: "AI学习教程",
    3: "AI生活融合",
    4: "AI社会反思"
}

# 种子词（用于帮助模型识别每个主题的关键词，不参与聚类）
SEED_TOPICS = {
    "AI内容创作": [
        "AI绘画", "AI视频", "AI动画", "AI建模", "AI设计", "AI插画",
        "AI创作", "AI生成", "Midjourney", "Stable_Diffusion", "文生图", "AIGC"
    ],
    "AI工具测评": [
        "ChatGPT", "Claude", "测评", "对比", "推荐", "工具",
        "使用体验", "神器", "评测", "红黑榜", "NotionAI"
    ],
    "AI学习教程": [
        "教程", "学习", "入门", "课程", "培训", "Prompt", "提示词",
        "技巧", "方法", "零基础", "小白", "实战", "训练"
    ],
    "AI生活融合": [
        "AI办公", "效率", "工作", "生活", "副业", "变现",
        "AI写作", "AI翻译", "AI修图", "职场", "打工"
    ],
    "AI社会反思": [
        "AI伦理", "焦虑", "失业", "威胁", "未来", "反思",
        "AI时代", "人类", "讨论", "监管", "安全", "隐私"
    ]
}

# ==================== 日志 ====================
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "bertopic_train.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


# ==================== 数据加载 ====================

def load_and_validate_data(file_path: Path) -> Tuple[List[str], List[str], List[str]]:
    """加载并验证训练数据"""
    logger.info(f"📂 加载数据: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"❌ 数据文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"   原始数据: {len(df)} 条")
    
    required_cols = ["note_id", "cleaned_full_text", "cleaned_seg_text"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ 缺少必要字段: {missing_cols}")
    
    # 过滤空值
    df = df.dropna(subset=["cleaned_full_text", "cleaned_seg_text"])
    df = df[df["cleaned_full_text"].str.strip() != ""]
    df = df[df["cleaned_seg_text"].str.strip() != ""]
    
    logger.info(f"   有效数据: {len(df)} 条")
    
    return df["note_id"].tolist(), df["cleaned_full_text"].tolist(), df["cleaned_seg_text"].tolist()


# ==================== 模型初始化 ====================

def initialize_models():
    """初始化BERTopic各子模型"""
    logger.info("=" * 60)
    logger.info("🔧 初始化子模型")
    logger.info("-" * 60)
    
    # 1. 嵌入模型
    logger.info(f"\n🤖 嵌入模型: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_model.max_seq_length = MAX_SEQ_LENGTH
    embed_dim = embedding_model.get_sentence_embedding_dimension()
    logger.info(f"   嵌入维度: {embed_dim}")
    
    # 2. UMAP降维
    logger.info(f"\n📉 UMAP降维")
    logger.info(f"   近邻数: {UMAP_PARAMS['n_neighbors']}")
    logger.info(f"   降维维度: {UMAP_PARAMS['n_components']}")
    logger.info(f"   距离度量: {UMAP_PARAMS['metric']}")
    umap_model = UMAP(**UMAP_PARAMS)
    
    # 3. K-Means聚类（强制5类）
    logger.info(f"\n🎯 K-Means聚类")
    logger.info(f"   聚类数量: {N_TOPICS} (强制)")
    kmeans_model = KMeans(
        n_clusters=N_TOPICS,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300
    )
    
    # 4. 词向量模型
    logger.info(f"\n📊 CountVectorizer")
    vectorizer_model = CountVectorizer(
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words=None
    )
    
    # 5. c-TF-IDF
    logger.info(f"\n📈 c-TF-IDF")
    from bertopic.vectorizers import ClassTfidfTransformer
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,
        bm25_weighting=True
    )
    
    return embedding_model, umap_model, kmeans_model, vectorizer_model, ctfidf_model


# ==================== 训练 ====================

def train_topic_model(
    full_texts: List[str],
    embedding_model,
    umap_model,
    kmeans_model,
    vectorizer_model,
    ctfidf_model
) -> Tuple[BERTopic, List[int], np.ndarray]:
    """执行BERTopic训练（使用K-Means）"""
    logger.info("🚀 开始训练")
    logger.info("-" * 60)
    logger.info(f"   文档数量: {len(full_texts)}")
    logger.info(f"   目标主题数: {N_TOPICS}")
    
    # 创建模型 - 注意：不使用seed_topic_list
    # 因为K-Means已经强制5类，seed_topic反而会干扰
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=kmeans_model,  # 传入K-Means替代HDBSCAN
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        verbose=True,
        calculate_probabilities=True,
        nr_topics=N_TOPICS,  # 强制5个主题
        top_n_words=15,       # 每个主题15个关键词
        # 不使用seed_topic_list，因为K-Means已经强制分类
    )
    
    start_time = datetime.now()
    logger.info(f"\n⏰ 开始时间: {start_time}")
    
    # 训练
    topics, probs = topic_model.fit_transform(full_texts)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\n⏰ 训练完成! 耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
    
    # 统计
    unique_topics = set(topics)
    n_clusters = len(unique_topics)
    
    logger.info(f"\n📊 训练结果:")
    logger.info(f"   主题数: {n_clusters}")
    
    return topic_model, topics, probs


# ==================== 结果分析 ====================

def analyze_results(
    topic_model: BERTopic,
    topics: List[int],
    note_ids: List[str],
    seg_texts: List[str]
) -> Dict:
    """分析主题建模结果"""
    logger.info("📊 主题分析")
    logger.info("-" * 60)
    
    # 主题信息
    topic_info = topic_model.get_topic_info()
    logger.info(f"\n主题概览:\n{topic_info.to_string()}")
    
    # 每个主题的关键词
    topics_keywords = {}
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        if words:
            topics_keywords[topic_id] = [
                {"word": w[0], "weight": float(w[1])} 
                for w in words[:15]
            ]
    
    # 主题分布统计
    topic_counts = pd.Series(topics).value_counts().sort_index()
    
    logger.info("\n📈 主题分布:")
    topic_distribution = {}
    for topic_id, count in topic_counts.items():
        topic_name = TOPIC_NAMES.get(topic_id, f"主题{topic_id}")
        ratio = count / len(topics) * 100
        topic_distribution[topic_id] = {
            "name": topic_name,
            "count": int(count),
            "ratio": float(ratio)
        }
        logger.info(f"   主题{topic_id} [{topic_name}]: {count} ({ratio:.1f}%)")
    
    return {
        "topic_info": topic_info,
        "topics_keywords": topics_keywords,
        "topic_distribution": topic_distribution
    }


# ==================== 保存结果 ====================

def save_results(
    topic_model: BERTopic,
    topics: List[int],
    probs: np.ndarray,
    note_ids: List[str],
    seg_texts: List[str],
    analysis_result: Dict
):
    """保存所有结果"""
    logger.info("=" * 50)
    logger.info("💾 保存结果")
    logger.info("=" * 50)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 文档-主题归属
    doc_topic_df = pd.DataFrame({
        "note_id": note_ids,
        "topic": topics,
        "topic_name": [TOPIC_NAMES.get(t, f"主题{t}") for t in topics],
        "seg_text": seg_texts
    })
    
    # 添加概率
    if probs is not None and len(probs) > 0:
        prob_cols = [f"prob_topic_{i}" for i in range(probs.shape[1])]
        prob_df = pd.DataFrame(probs, columns=prob_cols)
        doc_topic_df = pd.concat([doc_topic_df, prob_df], axis=1)
    
    doc_file = OUTPUT_DIR / "document_topic_assignment.csv"
    doc_topic_df.to_csv(doc_file, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 文档-主题: {doc_file.name}")
    
    # 2. 主题信息
    topic_info_file = OUTPUT_DIR / "topic_info.csv"
    analysis_result["topic_info"].to_csv(topic_info_file, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 主题信息: {topic_info_file.name}")
    
    # 3. 主题关键词
    keywords_file = OUTPUT_DIR / "topic_keywords.json"
    with open(keywords_file, "w", encoding="utf-8") as f:
        json.dump(analysis_result["topics_keywords"], f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 主题关键词: {keywords_file.name}")
    
    # 4. 模型序列化
    model_file = MODEL_DIR / "bertopic_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(topic_model, f)
    logger.info(f"✅ BERTopic模型: {model_file.name}")
    
    # 5. 模型配置
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "max_seq_length": MAX_SEQ_LENGTH,
        "umap_params": UMAP_PARAMS,
        "n_topics": N_TOPICS,
        "clustering": "KMeans",
        "topic_names": TOPIC_NAMES,
        "seed_topics": SEED_TOPICS,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_documents": len(topics)
    }
    config_file = MODEL_DIR / "model_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 模型配置: {config_file.name}")
    
    # 6. 训练报告（保存到logs目录）
    report_lines = [
        "=" * 60,
        "小红书AI笔记 BERTopic 主题建模训练报告",
        "-" * 60,
        "",
        "【一、基本信息】",
        f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"数据量: {len(topics)} 条",
        f"嵌入模型: {EMBEDDING_MODEL}",
        "",
        "【二、模型配置】",
        f"聚类方法: K-Means (强制{N_TOPICS}类)",
        f"UMAP降维: n_neighbors={UMAP_PARAMS['n_neighbors']}, n_components={UMAP_PARAMS['n_components']}",
        "",
        "【三、主题分布】",
    ]
    
    for topic_id, info in analysis_result["topic_distribution"].items():
        report_lines.append(f"主题{topic_id} [{info['name']}]: {info['count']}条 ({info['ratio']:.1f}%)")
    
    report_lines.extend([
        "",
        "【四、主题关键词】",
    ])
    
    for topic_id, keywords in analysis_result["topics_keywords"].items():
        topic_name = TOPIC_NAMES.get(topic_id, f"主题{topic_id}")
        top_words = ", ".join([k["word"] for k in keywords[:8]])
        report_lines.append(f"主题{topic_id} [{topic_name}]: {top_words}")
    
    report_lines.extend([
        "",
        "-" * 60,
    ])
    
    report = "\n".join(report_lines)
    report_file = LOG_DIR / "training_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"✅ 训练报告: {report_file.name}")
    
    logger.info(f"\n📁 输出目录: {OUTPUT_DIR}")
    logger.info(f"📁 模型目录: {MODEL_DIR}")
    logger.info(f"📁 日志目录: {LOG_DIR}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🚀 BERTopic 主题建模训练 (强制5主题)")
    logger.info("=" * 60)
    logger.info(f"工作目录: {WORK_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"日志目录: {LOG_DIR}")
    logger.info(f"目标主题数: {N_TOPICS}")
    logger.info("=" * 60)
    
    # 1. 加载数据
    note_ids, full_texts, seg_texts = load_and_validate_data(INPUT_FILE)
    
    # 2. 初始化模型
    embedding_model, umap_model, kmeans_model, vectorizer_model, ctfidf_model = initialize_models()
    
    # 3. 训练
    topic_model, topics, probs = train_topic_model(
        full_texts,
        embedding_model,
        umap_model,
        kmeans_model,
        vectorizer_model,
        ctfidf_model
    )
    
    # 4. 分析结果
    analysis_result = analyze_results(topic_model, topics, note_ids, seg_texts)
    
    # 5. 保存结果
    save_results(topic_model, topics, probs, note_ids, seg_texts, analysis_result)
    
    logger.info("=" * 60)
    logger.info("🎉 训练完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
