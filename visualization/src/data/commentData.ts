// 评论分析模拟数据
export interface CommentAnalysis {
  id: string;
  noteId: string;
  /** 与 content/rawdata 合并得到的 note_url，无则前端可按 noteId 拼 explore */
  noteUrl?: string;
  noteTitle: string;
  topicId: number;
  /** 与 content/BERTopic 关联的宏观主题；无匹配时为说明文案 */
  topicName: string;
  /** 是否在 content 主题数据中命中该 note_id */
  contentMatched?: boolean;
  /** content BERTopic 微观主题 ID */
  contentMicroTopicId?: number;
  /** 微观主题关键词 */
  contentMicroKeywords?: string;
  /** CSV 笔记关键词列 */
  contentTopicKeywords?: string;
  /** 主题映射置信度 */
  contentMappingConfidence?: number;
  /** content 合并：笔记正文 */
  noteContent?: string;
  /** content 合并：笔记简介 */
  noteDesc?: string;
  sentiment: 'positive' | 'neutral' | 'negative';
  sentimentScore: number; // 0-1, 越高越积极
  keywords: string[];
  commentCount: number;
  /** 本条情感统计所依据的预测评论条数（与 commentCount 可能不同） */
  sentimentSampleSize?: number;
  avgCommentLikes: number;
  topComments: TopComment[];
}

export interface TopComment {
  id: string;
  user: string;
  content: string;
  likes: number;
  sentiment: 'positive' | 'neutral' | 'negative';
  createdAt: string;
  /** 该评论所属笔记在 content/BERTopic 下的主题摘要，便于对照阅读 */
  noteTopicContext?: string;
}

/** 单条评论（与高评论笔记侧边项对齐，文本为本地预测/采集样本中的全文） */
export interface NoteCommentLine {
  id: string;
  user: string;
  content: string;
  sentiment: 'positive' | 'neutral' | 'negative';
}

export interface CommentTopic {
  id: number;
  /** 展示名：优先为笔记标题，否则为截断 note_id */
  name: string;
  /** 原始笔记 ID，与 content/rawdata、BERTopic 对齐 */
  noteId?: string;
  /** 笔记页链接（来自 content/rawdata 与 CSV 合并后的 note_url，可打开原帖/评论区） */
  noteUrl?: string;
  /** content 合并行：笔记正文（final_pro_topics.content / rawdata） */
  noteContent?: string;
  /** content 合并行：笔记简介 desc */
  noteDesc?: string;
  /** 该笔记在本地预测样本中的全部评论（完整 content，条数以样本为准） */
  noteComments?: NoteCommentLine[];
  /** content 侧宏观主题（BERTopic） */
  contentMacroTopic?: string;
  /** 是否在 content 主题表中命中 */
  contentMatched?: boolean;
  contentMicroTopicId?: number;
  contentMicroKeywords?: string;
  contentMappingConfidence?: number;
  /** 排名表/互动字段中的评论总数（可能大于已跑 MacBERT 的条数） */
  commentCount: number;
  /** 参与情感比例计算的预测评论条数；与 commentCount 一致时表示已覆盖该笔记全部预测行 */
  sentimentFromCount?: number;
  keywords: string[];
  positiveRatio: number;
  negativeRatio: number;
  neutralRatio: number;
  avgSentimentScore: number;
  trend: number[];
}

export interface SentimentTrend {
  date: string;
  positive: number;
  neutral: number;
  negative: number;
}

export interface CommentMetrics {
  totalComments: number;
  avgSentimentScore: number;
  positiveRatio: number;
  negativeRatio: number;
  neutralRatio: number;
  avgCommentLength: number;
  replyRate: number;
}

// 评论主题数据
export const mockCommentTopics: CommentTopic[] = [
  {
    id: 1,
    name: "工具推荐咨询",
    commentCount: 4521,
    keywords: ["推荐", "工具", "哪个好", "求分享", "链接"],
    positiveRatio: 0.65,
    negativeRatio: 0.1,
    neutralRatio: 0.25,
    avgSentimentScore: 0.72,
    trend: [120, 145, 168, 189, 210, 256, 298]
  },
  {
    id: 2,
    name: "使用教程询问",
    commentCount: 3287,
    keywords: ["怎么用", "教程", "步骤", "如何操作", "小白"],
    positiveRatio: 0.58,
    negativeRatio: 0.15,
    neutralRatio: 0.27,
    avgSentimentScore: 0.65,
    trend: [210, 234, 267, 289, 312, 345, 378]
  },
  {
    id: 3,
    name: "效果反馈评价",
    commentCount: 2890,
    keywords: ["真的", "好用", "效果", "不错", "赞"],
    positiveRatio: 0.78,
    negativeRatio: 0.08,
    neutralRatio: 0.14,
    avgSentimentScore: 0.82,
    trend: [145, 156, 178, 195, 212, 234, 256]
  },
  {
    id: 4,
    name: "问题与Bug反馈",
    commentCount: 2156,
    keywords: ["打不开", "闪退", "报错", "问题", "无效"],
    positiveRatio: 0.25,
    negativeRatio: 0.45,
    neutralRatio: 0.30,
    avgSentimentScore: 0.38,
    trend: [78, 89, 102, 115, 128, 142, 156]
  },
  {
    id: 5,
    name: "替代方案讨论",
    commentCount: 1876,
    keywords: ["替代", "平替", "类似的", "其他", "对比"],
    positiveRatio: 0.52,
    negativeRatio: 0.18,
    neutralRatio: 0.30,
    avgSentimentScore: 0.58,
    trend: [167, 182, 198, 215, 234, 256, 278]
  },
  {
    id: 6,
    name: "付费意愿表达",
    commentCount: 1654,
    keywords: ["付费", "花钱", "值不值", "免费", "会员"],
    positiveRatio: 0.42,
    negativeRatio: 0.28,
    neutralRatio: 0.30,
    avgSentimentScore: 0.48,
    trend: [189, 205, 228, 245, 267, 289, 312]
  },
  {
    id: 7,
    name: "学习经验分享",
    commentCount: 1432,
    keywords: ["学习", "经验", "心得", "分享", "技巧"],
    positiveRatio: 0.70,
    negativeRatio: 0.05,
    neutralRatio: 0.25,
    avgSentimentScore: 0.75,
    trend: [112, 125, 138, 152, 165, 178, 192]
  },
  {
    id: 8,
    name: "互动问答交流",
    commentCount: 1287,
    keywords: ["回复", "问答", "请教", "帮忙", "感谢"],
    positiveRatio: 0.68,
    negativeRatio: 0.07,
    neutralRatio: 0.25,
    avgSentimentScore: 0.71,
    trend: [134, 145, 162, 178, 195, 212, 228]
  }
];

// 情感趋势数据
export const mockSentimentTrends: SentimentTrend[] = Array.from({ length: 7 }, (_, i) => {
  const date = new Date();
  date.setDate(date.getDate() - 6 + i);
  const dateStr = date.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' });
  
  return {
    date: dateStr,
    positive: Math.floor(800 + Math.random() * 400),
    neutral: Math.floor(300 + Math.random() * 150),
    negative: Math.floor(100 + Math.random() * 80)
  };
});

// 评论指标汇总
export const mockCommentMetrics: CommentMetrics = {
  totalComments: 21543,
  avgSentimentScore: 0.65,
  positiveRatio: 0.62,
  negativeRatio: 0.15,
  neutralRatio: 0.23,
  avgCommentLength: 28,
  replyRate: 0.45
};

// 各主题的评论详细数据
export const mockCommentAnalysis: CommentAnalysis[] = Array.from({ length: 20 }, (_, i) => {
  const sentiments: ('positive' | 'neutral' | 'negative')[] = ['positive', 'neutral', 'negative'];
  const sentiment = sentiments[Math.floor(Math.random() * 3)];
  
  return {
    id: `comment_analysis_${i}`,
    noteId: `note_${Math.floor(i / 2) + 1}`,
    noteTitle: `AI相关笔记 ${i + 1}`,
    topicId: Math.floor(Math.random() * 10) + 1,
    topicName: ["AI工具推荐", "AI绘画教程", "AI办公自动化", "AI学习路线", "AI副业赚钱"][Math.floor(Math.random() * 5)],
    sentiment,
    sentimentScore: sentiment === 'positive' ? 0.6 + Math.random() * 0.4 : 
                   sentiment === 'neutral' ? 0.4 + Math.random() * 0.2 : 
                   Math.random() * 0.4,
    keywords: ["推荐", "教程", "好用", "问题", "分享"].slice(0, Math.floor(Math.random() * 3) + 2),
    commentCount: Math.floor(Math.random() * 500) + 50,
    avgCommentLikes: Math.floor(Math.random() * 20) + 5,
    topComments: Array.from({ length: 3 }, (_, j) => ({
      id: `top_comment_${i}_${j}`,
      user: `用户${Math.floor(Math.random() * 10000)}`,
      content: ["太棒了！", "谢谢分享", "学习了", "很有用", "支持一下"][j],
      likes: Math.floor(Math.random() * 100),
      sentiment: sentiments[Math.floor(Math.random() * 3)],
      createdAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
    }))
  };
});

// 情感分布数据（用于饼图）
export const mockSentimentDistribution = [
  { name: '积极', value: 62, color: '#22c55e' },
  { name: '中性', value: 23, color: '#a1a1aa' },
  { name: '消极', value: 15, color: '#ef4444' }
];

// 评论关键词云数据
export const mockCommentWordCloud = [
  { text: '推荐', weight: 156 },
  { text: '好用', weight: 134 },
  { text: '教程', weight: 123 },
  { text: '分享', weight: 118 },
  { text: '谢谢', weight: 98 },
  { text: '学习', weight: 89 },
  { text: '工具', weight: 87 },
  { text: '效果', weight: 76 },
  { text: '不错', weight: 72 },
  { text: '真的', weight: 68 },
  { text: '问题', weight: 65 },
  { text: '怎么用', weight: 58 },
  { text: '感谢', weight: 54 },
  { text: '付费', weight: 48 },
  { text: '免费', weight: 45 }
];

// ============================================================
// RoBERTa 模型训练结果数据
// ============================================================

// 模型配置信息
export interface RoBERTaConfig {
  /** 卡面常用中文/英文全称，便于读者对照论文与模型卡 */
  modelName: string;
  /** Hugging Face 预训练权重仓库 ID（与本仓库加载的基座一致） */
  baseModel: string;
  numLabels: number;
  maxLength: number;
  batchSize: number;
  learningRate: number;
  epochs: number;
  trainingDate: string;
  trainSize: number;
  valSize: number;
  testSize: number;
}

export const robertaConfig: RoBERTaConfig = {
  modelName:
    "Chinese MacBERT-large（与 comment/train_roberta.py 默认 DEFAULT_MODEL_NAME 一致）",
  baseModel: "hfl/chinese-macbert-large",
  numLabels: 3,
  maxLength: 256,
  batchSize: 16,
  learningRate: 2e-5,
  epochs: 4,
  trainingDate: "2024-01-15",
  trainSize: 4000,
  valSize: 500,
  testSize: 500
};

// 模型评估指标
export interface EvaluationMetrics {
  accuracy: number;
  macroF1: number;
  macroAuc: number;
  precision: number[];
  recall: number[];
  f1Score: number[];
}

export const evaluationMetrics: EvaluationMetrics = {
  accuracy: 0.8723,
  macroF1: 0.8546,
  macroAuc: 0.9234,
  precision: [0.89, 0.82, 0.85],
  recall: [0.87, 0.79, 0.88],
  f1Score: [0.88, 0.80, 0.86]
};

// 混淆矩阵数据
export const confusionMatrix = [
  [435, 35, 30],  // 正面预测
  [45, 395, 60],  // 中立预测
  [25, 45, 440]   // 负面预测
];

// 各类别详细指标
export const classMetrics = [
  { name: '正面/积极', precision: 0.89, recall: 0.87, f1: 0.88, support: 500 },
  { name: '中立/客观', precision: 0.82, recall: 0.79, f1: 0.80, support: 500 },
  { name: '负面/消极', precision: 0.85, recall: 0.88, f1: 0.86, support: 500 }
];

// 置信度分布数据
export const confidenceDistribution = [
  { range: '0.0-0.2', count: 12, accuracy: 0.25 },
  { range: '0.2-0.4', count: 45, accuracy: 0.48 },
  { range: '0.4-0.6', count: 128, accuracy: 0.72 },
  { range: '0.6-0.8', count: 312, accuracy: 0.85 },
  { range: '0.8-1.0', count: 503, accuracy: 0.94 }
];

// 训练过程数据（每个epoch的指标）
export const trainingHistory = [
  { epoch: 1, trainLoss: 0.85, valLoss: 0.72, valAcc: 0.72, valF1: 0.68 },
  { epoch: 2, trainLoss: 0.58, valLoss: 0.51, valAcc: 0.79, valF1: 0.76 },
  { epoch: 3, trainLoss: 0.42, valLoss: 0.43, valAcc: 0.84, valF1: 0.82 },
  { epoch: 4, trainLoss: 0.35, valLoss: 0.38, valAcc: 0.87, valF1: 0.85 }
];

// 预测样本示例
export const predictionSamples = [
  { id: 1, content: "太棒了！AI绘画真的太好用了，效率翻倍！", trueLabel: "正面", predLabel: "正面", confidence: 0.95, correct: true },
  { id: 2, content: "这个工具需要付费才能使用全部功能", trueLabel: "中立", predLabel: "中立", confidence: 0.78, correct: true },
  { id: 3, content: "打不开一直闪退，体验很差", trueLabel: "负面", predLabel: "负面", confidence: 0.92, correct: true },
  { id: 4, content: "有没有免费的替代方案？", trueLabel: "中立", predLabel: "正面", confidence: 0.65, correct: false },
  { id: 5, content: "ChatGPT太牛了，改变工作方式", trueLabel: "正面", predLabel: "正面", confidence: 0.89, correct: true },
  { id: 6, content: "感觉AI会取代很多工作，有点担忧", trueLabel: "负面", predLabel: "负面", confidence: 0.82, correct: true }
];
