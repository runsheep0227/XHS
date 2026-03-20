"""
predict.py
交互式主题预测脚本
加载本地 BERTopic 模型，输入文本即可得到主题分类结果
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ==================== 环境配置 ====================
os.environ['NUMBA_CACHE_DIR'] = str(Path(os.environ.get('TEMP', '/tmp')) / 'numba_cache')
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 算法库 ====================
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==================== 路径配置 ====================
WORK_DIR = Path(r"E:\document\PG\studio\content")
MODEL_DIR = WORK_DIR / "bertopic_results_optimized" / "saved_model"

# ==================== 宏观锚点（与训练脚本保持一致） ====================
MACRO_ANCHORS = {
    "AI内容创作": {
        "definition": "核心词：绘画 插画 视频 建模 设计 生成 创作 写小说 网文 写作 音乐 短剧 脚本 IP 形象 Midjourney Stable Diffusion SD MJ ComfyUI 工作流 LoRA ControlNet。场景定义：利用人工智能生成图像、视频、音频、三维模型或文学作品的直接创作过程与作品展示。",
        "color": "\033[91m",  # 红色
        "keywords": ["绘画", "插画", "视频", "建模", "生成", "创作", "小说", "网文",
                     "写作", "音乐", "短剧", "Midjourney", "SD", "ComfyUI",
                     "工作流", "LoRA", "ControlNet", "出图", "生图", "画风"]
    },
    "AI应用与测评": {
        "definition": "核心词：测评 推荐 对比 红黑榜 体验 智能体 机器人 星野 猫箱 APP 硬件 大模型 GPT Claude Kimi 通义 文心 DeepSeek。场景定义：对具体的AI软件、应用、平台、硬件设备或虚拟角色伴侣进行功能测试、优缺点评价与盘点。",
        "color": "\033[94m",  # 蓝色
        "keywords": ["测评", "推荐", "对比", "体验", "智能体", "星野", "猫箱", "APP",
                     "硬件", "大模型", "GPT", "Claude", "Kimi", "通义", "文心", "DeepSeek"]
    },
    "AI学习教程": {
        "definition": "核心词：教程 学习 入门 提示词 指令 课程 培训 指南 考证 技巧 零基础 prompt 咒语 话术。场景定义：教授用户如何使用AI工具的实操指南、指令词编写方法、学习路线与教育培训内容。",
        "color": "\033[92m",  # 绿色
        "keywords": ["教程", "学习", "入门", "提示词", "指令", "课程", "培训", "指南",
                     "考证", "技巧", "零基础", "prompt", "怎么用", "手把手", "保姆级"]
    },
    "AI赋能工作生活": {
        "definition": "核心词：效率 办公 搞钱 副业 变现 论文 翻译 简历 英语 口语 旅游 算命 求职 PPT Excel 数据分析。场景定义：将AI作为辅助工具解决工作效率提升、学术研究、赚钱变现或日常生活具体场景的问题。",
        "color": "\033[93m",  # 黄色
        "keywords": ["效率", "办公", "副业", "变现", "论文", "翻译", "简历", "英语",
                     "口语", "赚钱", "搞钱", "PPT", "Excel", "数据分析", "代码", "编程"]
    },
    "AI社会反思": {
        "definition": "核心词：失业 焦虑 取代 版权 侵权 法律 伦理 诈骗 深度伪造 威胁 监管 维权 画师 设计师 失业潮。场景定义：反思AI带来的负面影响，包括画师维权、人类职业被取代的恐慌、数据隐私与科技伦理争议。",
        "color": "\033[95m",  # 紫色
        "keywords": ["失业", "焦虑", "取代", "版权", "侵权", "法律", "伦理", "诈骗",
                     "深度伪造", "维权", "画师", "设计师", "隐私", "数据安全", "歧视"]
    }
}
RESET = "\033[0m"
BOLD = "\033[1m"


# ==================== 模型加载 ====================
class TopicPredictor:
    """主题预测器：封装模型加载与预测逻辑"""

    def __init__(self):
        self.topic_model = None
        self.embedding_model = None
        self.macro_embeddings = None
        self.macro_names = list(MACRO_ANCHORS.keys())
        self._load()

    def _load(self):
        """加载本地模型与嵌入模型"""
        print(f"\n{'='*60}")
        print(f"  正在加载本地模型，请稍候...")
        print(f"{'='*60}\n")

        # 加载 BERTopic 模型
        if not MODEL_DIR.exists():
            print(f"❌ 模型目录不存在: {MODEL_DIR}")
            print(f"   请先运行 bertopic_train.py 完成训练")
            sys.exit(1)

        # ====== 先加载嵌入模型 ======
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  🤖 加载嵌入模型: BAAI/bge-large-zh-v1.5 (设备: {device})")
        self.embedding_model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device=device)

        # ====== 再加载 BERTopic 模型，并传入嵌入模型 ======
        print(f"  📂 加载 BERTopic 模型: {MODEL_DIR}")
        self.topic_model = BERTopic.load(
            str(MODEL_DIR),
            embedding_model=self.embedding_model  # ← 关键：显式传入
        )

        # 预计算宏观锚点向量
        macro_descriptions = [anchor["definition"] for anchor in MACRO_ANCHORS.values()]
        self.macro_embeddings = self.embedding_model.encode(
            macro_descriptions, normalize_embeddings=True
        )

        # 输出模型信息
        topic_info = self.topic_model.get_topic_info()
        n_topics = len(topic_info[topic_info['Topic'] != -1])
        print(f"\n  ✅ 加载完成！")
        print(f"     微观主题数: {n_topics}")
        print(f"     宏观类别数: {len(self.macro_names)}")
        print(f"{'='*60}\n")


    def predict(self, text: str) -> dict:
        """
        对单条文本进行主题预测

        返回:
        {
            "text": 原文,
            "micro_topic_id": 微观主题ID,
            "micro_topic_keywords": 微观主题关键词,
            "macro_topic_name": 宏观主题名称,
            "confidence": 映射置信度,
            "top3_macros": [(类别名, 置信度), ...]
        }
        """
        if not text or not text.strip():
            return {"error": "输入文本为空"}

        # 1. 微观主题预测
        micro_topics, micro_probs = self.topic_model.transform([text])
        micro_id = micro_topics[0]

        # 2. 获取微观主题关键词
        if micro_id == -1:
            micro_keywords = "无明确主题（离群点）"
        else:
            topic_words = self.topic_model.get_topic(micro_id)
            micro_keywords = ", ".join([w[0] for w in topic_words[:10]])

        # 3. 宏观映射（使用与训练脚本一致的双重相似度策略）
        if micro_id == -1:
            # 离群点：直接用原文嵌入与宏观锚点比较
            text_vector = self.embedding_model.encode([text], normalize_embeddings=True)
            similarities = cosine_similarity(text_vector, self.macro_embeddings)[0]
        else:
            # 有微观主题：用微观主题的关键词+代表性文档构建锚点
            top_words = " ".join([w[0] for w in self.topic_model.get_topic(micro_id)[:20]])
            rep_docs = self.topic_model.get_representative_docs(micro_id)
            rep_text = " ".join(rep_docs)[:400] if rep_docs else ""

            # 语义相似度
            micro_anchor = f"核心词：{top_words}。场景内容：{rep_text}"
            micro_vector = self.embedding_model.encode([micro_anchor], normalize_embeddings=True)
            semantic_sims = cosine_similarity(micro_vector, self.macro_embeddings)[0]

            # 关键词相似度
            macro_kw_texts = [" ".join(anchor["keywords"]) for anchor in MACRO_ANCHORS.values()]
            macro_kw_embeddings = self.embedding_model.encode(macro_kw_texts, normalize_embeddings=True)
            keyword_sims = cosine_similarity(micro_vector, macro_kw_embeddings)[0]

            # 加权融合
            similarities = 0.6 * semantic_sims + 0.4 * keyword_sims

        # 4. 排序输出 Top3
        ranked_indices = np.argsort(similarities)[::-1]
        top3 = [(self.macro_names[i], float(similarities[i])) for i in ranked_indices[:3]]

        best_idx = ranked_indices[0]
        best_macro = self.macro_names[best_idx]
        best_confidence = float(similarities[best_idx])

        return {
            "text": text,
            "micro_topic_id": micro_id,
            "micro_topic_keywords": micro_keywords,
            "macro_topic_name": best_macro,
            "confidence": best_confidence,
            "top3_macros": top3
        }

    def predict_batch(self, texts: list) -> list:
        """批量预测"""
        return [self.predict(t) for t in texts]


# ==================== 终端展示 ====================
def display_result(result: dict):
    """格式化输出预测结果"""

    if "error" in result:
        print(f"\n  ❌ {result['error']}\n")
        return

    macro = result["macro_topic_name"]
    conf = result["confidence"]
    color = MACRO_ANCHORS.get(macro, {}).get("color", "")

    # 置信度等级
    if conf >= 0.5:
        conf_level = "🟢 高置信"
    elif conf >= 0.35:
        conf_level = "🟡 中置信"
    else:
        conf_level = "🔴 低置信"

    print(f"\n{'─'*60}")
    print(f"  📝 输入文本: {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}")
    print(f"{'─'*60}")

    # 微观主题
    print(f"  🔬 微观主题: T{result['micro_topic_id']}")
    print(f"     关键词:   {result['micro_topic_keywords']}")

    # 宏观类别
    print(f"  🌟 宏观类别: {color}{BOLD}{macro}{RESET}")
    print(f"  📊 置信度:   {conf:.3f}  {conf_level}")

    # Top3 候选
    print(f"\n  📋 Top3 候选:")
    for rank, (name, score) in enumerate(result["top3_macros"], 1):
        c = MACRO_ANCHORS.get(name, {}).get("color", "")
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        marker = " ←" if rank == 1 else ""
        print(f"     {rank}. {c}{name}{RESET}  {bar} {score:.3f}{marker}")

    print(f"{'─'*60}\n")


def display_banner():
    """启动横幅"""
    print(f"""
{BOLD}╔════════════════════════════════════════════════════════════╗
║                                                            ║
║          🔮  AIGC 小红书笔记主题预测工具  🔮               ║
║                                                            ║
║  使用方法:                                                 ║
║    输入一段文本 → 自动判断所属主题类别                     ║
║                                                            ║
║  支持命令:                                                 ║
║    quit / q / exit    退出程序                             ║
║    batch              批量输入模式                         ║
║    info               查看模型信息                         ║
║    help               查看帮助                             ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
{RESET}""")


def display_model_info(predictor: TopicPredictor):
    """展示模型详细信息"""
    topic_info = predictor.topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]

    print(f"\n{'='*60}")
    print(f"  📊 模型信息")
    print(f"{'='*60}")
    print(f"  模型路径:     {MODEL_DIR}")
    print(f"  微观主题数:   {len(valid_topics)}")
    print(f"  总文档数:     {topic_info['Count'].sum()}")
    print(f"  嵌入模型:     BAAI/bge-large-zh-v1.5")
    print(f"  设备:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    print(f"\n  📋 各宏观类别包含的微观主题:")
    print(f"  {'─'*50}")

    # 用预测器做一次空查询来获取映射关系
    macro_topic_map = {}
    for _, row in valid_topics.iterrows():
        tid = row['Topic']
        words = " ".join([w[0] for w in predictor.topic_model.get_topic(tid)[:8]])
        # 用关键词做一次宏观映射
        test_result = predictor.predict(words)
        macro = test_result["macro_topic_name"]
        if macro not in macro_topic_map:
            macro_topic_map[macro] = []
        macro_topic_map[macro].append((tid, row['Count'], words))

    for macro in MACRO_ANCHORS:
        color = MACRO_ANCHORS[macro]["color"]
        topics = macro_topic_map.get(macro, [])
        total = sum(t[1] for t in topics)
        print(f"\n  {color}■ {macro} ({len(topics)}个子主题, {total}条数据){RESET}")
        for tid, count, kw in sorted(topics, key=lambda x: x[1], reverse=True):
            print(f"     T{tid:2d} ({count:4d}条) {kw}")

    print(f"\n{'='*60}\n")


# ==================== 批量模式 ====================
def batch_mode(predictor: TopicPredictor):
    """批量输入模式"""
    print(f"\n  📦 批量输入模式（每行一条，输入空行结束）:\n")
    texts = []
    while True:
        line = input(f"  [{len(texts)+1}] >>> ").strip()
        if not line:
            break
        texts.append(line)

    if not texts:
        print("  ⚠️ 未输入任何文本")
        return

    print(f"\n  正在预测 {len(texts)} 条文本...\n")
    results = predictor.predict_batch(texts)

    # 汇总表
    print(f"  {'序号':<5} {'微观主题':<12} {'宏观类别':<16} {'置信度':<8} {'文本摘要'}")
    print(f"  {'─'*75}")
    for i, r in enumerate(results, 1):
        if "error" in r:
            print(f"  {i:<5} {'ERROR':<12} {'-':<16} {'-':<8} {r['error']}")
            continue
        macro = r["macro_topic_name"]
        color = MACRO_ANCHORS.get(macro, {}).get("color", "")
        text_preview = r["text"][:30] + ("..." if len(r["text"]) > 30 else "")
        print(f"  {i:<5} T{r['micro_topic_id']:<10} {color}{macro}{RESET}{'':<{16-len(macro)}} {r['confidence']:<8.3f} {text_preview}")

    # 汇总统计
    print(f"\n  📊 批量预测汇总:")
    from collections import Counter
    macro_counts = Counter(r["macro_topic_name"] for r in results if "error" not in r)
    for macro, count in macro_counts.most_common():
        color = MACRO_ANCHORS.get(macro, {}).get("color", "")
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"     {color}■ {macro:<16}{RESET} {count:>4}条 ({pct:.1f}%) {bar}")
    print()


# ==================== 主循环 ====================
def main():
    display_banner()

    # 加载模型
    predictor = TopicPredictor()

    print(f"  💡 输入文本后按回车即可获得主题分类结果\n")

    while True:
        try:
            user_input = input(f"  >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  👋 再见！\n")
            break

        # 命令处理
        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "q", "exit"):
            print(f"\n  👋 再见！\n")
            break
        elif cmd == "help":
            display_banner()
            continue
        elif cmd == "info":
            display_model_info(predictor)
            continue
        elif cmd == "batch":
            batch_mode(predictor)
            continue

        # 正常预测
        result = predictor.predict(user_input)
        display_result(result)


if __name__ == "__main__":
    main()
