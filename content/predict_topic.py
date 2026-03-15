#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

# ==================== 1. 配置与种子词定义 ====================
MODEL_PATH = Path(r"E:\document\PG\studio\content\bertopic_results\bertopic_model\bertopic_model.pkl")

# 你的原始种子词（用来做自动匹配）
SEED_TOPICS_DEFINITION = {
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

class SmartTopicPredictor:
    def __init__(self, model_path):
        print(f"正在加载模型...")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        # 提取模型组件
        self.embedding_model = self.model.embedding_model
        self.topic_embeddings_ = None # 缓存主题中心向量
        
        # 自动建立映射
        print("正在自动匹配主题名称...")
        self.id_to_name_map = self._auto_match_topics()
        print("匹配完成！\n")

    def _auto_match_topics(self):
        """核心逻辑：将模型的Topic ID与用户定义的种子词进行匹配"""
        mapping = {}
        
        # 1. 获取模型的所有主题
        topic_info = self.model.get_topic_info()
        
        print("-" * 40)
        print(f"{'模型ID':<8} | {'匹配结果':<15} | {'模型关键词(Top5)'}")
        print("-" * 40)

        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1: continue # 跳过离群点
            
            # 获取该主题的模型关键词
            model_words = [w[0] for w in self.model.get_topic(topic_id)[:10]]
            
            # 2. 计算与每个种子词列表的重叠分数
            best_name = "未知主题"
            best_score = 0
            
            for name, seed_words in SEED_TOPICS_DEFINITION.items():
                # 简单的重叠计数 (也可以用Jaccard)
                score = len(set(model_words) & set(seed_words))
                # 稍微加权：如果完全匹配到核心词加分
                if score > best_score:
                    best_score = score
                    best_name = name
            
            # 如果完全没匹配上，留空或用模型自带的Name
            if best_score == 0:
                best_name = f"模型主题_{topic_id}"
            
            mapping[topic_id] = best_name
            print(f"{topic_id:<8} | {best_name:<15} | {', '.join(model_words[:5])}")
        
        print("-" * 40)
        return mapping

    def _calculate_similarity(self, text_embedding, topic_id):
        """计算文本与主题中心的余弦相似度（作为置信度）"""
        # 确保我们有主题的嵌入向量
        if self.topic_embeddings_ is None:
            # 利用 c-TF-IDF 或 聚类中心来获取代表向量
            # 这里为了简单，直接取该主题下所有文档的平均嵌入（如果有的话）
            # 或者使用 topic_embeddings_ (BERTopic 通常会计算好)
            if hasattr(self.model, 'topic_embeddings_'):
                self.topic_embeddings_ = self.model.topic_embeddings_
            else:
                return 0.5 #  fallback
        
        # 注意：topic_embeddings_ 的索引通常是按 -1, 0, 1, 2... 排列的
        # 这里做一个简单的索引对齐假设，或者直接用向量计算
        try:
            # 归一化
            text_norm = text_embedding / np.linalg.norm(text_embedding)
            
            # 找到对应ID的嵌入 (这里简化处理，假设topic_embeddings_是对齐的)
            # BERTopic 的 topic_embeddings_ 通常第一个是 -1，然后是 0,1,2...
            # 为了鲁棒性，我们重新计算该主题Top词的平均嵌入作为中心
            topic_words = [w[0] for w in self.model.get_topic(topic_id)[:5]]
            if not topic_words: return 0.0
            
            word_embeddings = self.embedding_model.encode(topic_words)
            centroid = np.mean(word_embeddings, axis=0)
            centroid_norm = centroid / np.linalg.norm(centroid)
            
            sim = np.dot(text_norm, centroid_norm)
            return max(0, float(sim)) # 保证非负
            
        except Exception as e:
            return 0.5

    def predict(self, text):
        # 1. 基础预测
        topics, _ = self.model.transform([text])
        topic_id = topics[0]
        
        # 2. 获取名称
        topic_name = self.id_to_name_map.get(topic_id, "未知")
        
        # 3. 计算嵌入与置信度
        text_embedding = self.embedding_model.encode([text])[0]
        confidence = self._calculate_similarity(text_embedding, topic_id)
        
        return {
            "id": topic_id,
            "name": topic_name,
            "confidence": confidence,
            "keywords": [w[0] for w in self.model.get_topic(topic_id)[:3]]
        }

# ==================== 主程序 ====================
if __name__ == "__main__":
    predictor = SmartTopicPredictor(MODEL_PATH)
    
    print("\n✅ 系统就绪！请输入文本进行测试 (输入 q 退出)\n")
    
    while True:
        user_input = input("请输入: ").strip()
        if user_input.lower() in ['q', 'quit']: break
        if not user_input: continue
        
        res = predictor.predict(user_input)
        
        # 彩色输出
        color_code = "\033[1;32m" if res['confidence'] > 0.7 else "\033[1;33m"
        reset_code = "\033[0m"
        
        print(f"  └─ 主题: {color_code}{res['name']}{reset_code} (ID: {res['id']})")
        print(f"  └─ 置信度: {res['confidence']:.2f}")
        print(f"  └─ 关键词: {', '.join(res['keywords'])}\n")