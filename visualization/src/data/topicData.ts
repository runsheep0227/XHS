// 模拟小红书AI主题研究数据
export interface Note {
  id: string;
  title: string;
  content: string;
  likes: number;
  comments: number;
  shares: number;
  collects: number;
  discusses: number;
  topicId: number;
  topicName: string;
  keywords: string[];
  createdAt: string;
  commentList: Comment[];
}

export interface Comment {
  id: string;
  user: string;
  content: string;
  likes: number;
  createdAt: string;
}

export interface Topic {
  id: number;
  name: string;
  noteCount: number;
  keywords: string[];
  avgLikes: number;
  avgComments: number;
  avgCollects: number;
  avgShares: number;
  avgDiscusses: number;
  trend: number[]; // 最近7天的趋势
}

// 生成模拟主题数据
export const mockTopics: Topic[] = [
  {
    id: 1,
    name: "AI工具推荐",
    noteCount: 4521,
    keywords: ["ChatGPT", "AI工具", "效率", "Prompt", "神器"],
    avgLikes: 892,
    avgComments: 45,
    avgCollects: 234,
    avgShares: 67,
    avgDiscusses: 123,
    trend: [120, 145, 168, 189, 210, 256, 298]
  },
  {
    id: 2,
    name: "AI绘画教程",
    noteCount: 3287,
    keywords: ["Midjourney", "Stable Diffusion", "AI绘画", "提示词", "咒语"],
    avgLikes: 1245,
    avgComments: 78,
    avgCollects: 456,
    avgShares: 89,
    avgDiscusses: 234,
    trend: [210, 234, 267, 289, 312, 345, 378]
  },
  {
    id: 3,
    name: "AI办公自动化",
    noteCount: 2156,
    keywords: ["Excel", "PPT", "自动化", "工作效率", "职场"],
    avgLikes: 567,
    avgComments: 34,
    avgCollects: 189,
    avgShares: 45,
    avgDiscusses: 89,
    trend: [78, 89, 102, 115, 128, 142, 156]
  },
  {
    id: 4,
    name: "AI学习路线",
    noteCount: 1890,
    keywords: ["Python", "机器学习", "深度学习", "入门", "课程"],
    avgLikes: 723,
    avgComments: 56,
    avgCollects: 267,
    avgShares: 78,
    avgDiscusses: 145,
    trend: [145, 156, 178, 195, 212, 234, 256]
  },
  {
    id: 5,
    name: "AI副业赚钱",
    noteCount: 1654,
    keywords: ["变现", "副业", "接单", "赚钱", "自由职业"],
    avgLikes: 945,
    avgComments: 89,
    avgCollects: 312,
    avgShares: 102,
    avgDiscusses: 178,
    trend: [167, 182, 198, 215, 234, 256, 278]
  },
  {
    id: 6,
    name: "AI视频制作",
    noteCount: 1432,
    keywords: ["视频生成", "Runway", "Pika", "AI视频", "剪辑"],
    avgLikes: 1089,
    avgComments: 67,
    avgCollects: 389,
    avgShares: 95,
    avgDiscusses: 198,
    trend: [189, 205, 228, 245, 267, 289, 312]
  },
  {
    id: 7,
    name: "AI产品测评",
    noteCount: 1287,
    keywords: ["测评", "对比", "工具", "软件", "推荐"],
    avgLikes: 678,
    avgComments: 45,
    avgCollects: 234,
    avgShares: 56,
    avgDiscusses: 112,
    trend: [112, 125, 138, 152, 165, 178, 192]
  },
  {
    id: 8,
    name: "AI提示词技巧",
    noteCount: 1156,
    keywords: ["Prompt", "提示词", "技巧", "方法", "模板"],
    avgLikes: 834,
    avgComments: 52,
    avgCollects: 278,
    avgShares: 71,
    avgDiscusses: 134,
    trend: [134, 145, 162, 178, 195, 212, 228]
  },
  {
    id: 9,
    name: "AI行业资讯",
    noteCount: 987,
    keywords: ["新闻", "动态", "OpenAI", "谷歌", "微软"],
    avgLikes: 456,
    avgComments: 28,
    avgCollects: 156,
    avgShares: 34,
    avgDiscusses: 89,
    trend: [89, 95, 102, 112, 125, 138, 145]
  },
  {
    id: 10,
    name: "AI伦理与思考",
    noteCount: 654,
    keywords: ["伦理", "思考", "未来", "影响", "讨论"],
    avgLikes: 523,
    avgComments: 67,
    avgCollects: 189,
    avgShares: 45,
    avgDiscusses: 156,
    trend: [156, 162, 175, 188, 195, 205, 215]
  }
];

// 生成模拟笔记数据
const generateNotes = (): Note[] => {
  const notes: Note[] = [];
  const topics = mockTopics;
  
  topics.forEach(topic => {
    for (let i = 0; i < Math.min(topic.noteCount, 50); i++) {
      const noteId = `note_${topic.id}_${i}`;
      notes.push({
        id: noteId,
        title: `${topic.keywords[0]} ${topic.keywords[1]}经验分享`,
        content: `这是一篇关于${topic.name}的详细笔记。包含了${topic.keywords.slice(0, 3).join('、')}等方面的内容...`,
        likes: Math.floor(topic.avgLikes * (0.5 + Math.random())),
        comments: Math.floor(topic.avgComments * (0.5 + Math.random())),
        shares: Math.floor(topic.avgShares * (0.5 + Math.random())),
        collects: Math.floor(topic.avgCollects * (0.5 + Math.random())),
        discusses: Math.floor(topic.avgDiscusses * (0.5 + Math.random())),
        topicId: topic.id,
        topicName: topic.name,
        keywords: topic.keywords,
        createdAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        commentList: Array.from({ length: Math.floor(Math.random() * 10) + 1 }, (_, j) => ({
          id: `comment_${noteId}_${j}`,
          user: `用户${Math.floor(Math.random() * 10000)}`,
          content: "说得太对了！👍",
          likes: Math.floor(Math.random() * 50),
          createdAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
        }))
      });
    }
  });
  
  return notes;
};

export const mockNotes = generateNotes();

// 主题互动汇总数据（用于热力图）
export const interactionMatrix = mockTopics.map(topic => ({
  topic: topic.name,
  likes: topic.avgLikes,
  comments: topic.avgComments,
  collects: topic.avgCollects,
  shares: topic.avgShares,
  discusses: topic.avgDiscusses
}));

// 时间趋势数据
export const trendData = Array.from({ length: 7 }, (_, i) => {
  const date = new Date();
  date.setDate(date.getDate() - 6 + i);
  const dateStr = date.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' });
  
  return {
    date: dateStr,
    notes: Math.floor(500 + Math.random() * 200),
    likes: Math.floor(8000 + Math.random() * 3000),
    comments: Math.floor(1000 + Math.random() * 500),
    collects: Math.floor(3000 + Math.random() * 1000)
  };
});

// ============================================================
// BERTopic 模型相关数据
// ============================================================

// BERTopic 模型配置信息
export interface BERTopicConfig {
  embeddingModel: string;
  umapParams: {
    nNeighbors: number;
    nComponents: number;
    metric: string;
    minDist: number;
  };
  clustering: string;
  nTopics: number;
  trainingDate: string;
  nDocuments: number;
}

export const bertopicConfig: BERTopicConfig = {
  embeddingModel: "paraphrase-multilingual-MiniLM-L12-v2",
  umapParams: {
    nNeighbors: 15,
    nComponents: 2,
    metric: "cosine",
    minDist: 0.1
  },
  clustering: "K-Means",
  nTopics: 5,
  trainingDate: "2024-01-15",
  nDocuments: 11509
};

// 主题关键词及权重（用于词云和条形图）
export interface TopicKeyword {
  word: string;
  weight: number;
}

export interface TopicKeywords {
  topicId: number;
  topicName: string;
  keywords: TopicKeyword[];
}

export const bertopicKeywords: TopicKeywords[] = [
  {
    topicId: 0,
    topicName: "AI内容创作",
    keywords: [
      { word: "AI绘画", weight: 0.85 },
      { word: "Midjourney", weight: 0.78 },
      { word: "Stable Diffusion", weight: 0.72 },
      { word: "文生图", weight: 0.65 },
      { word: "AI视频", weight: 0.58 },
      { word: "AIGC", weight: 0.52 },
      { word: "AI建模", weight: 0.45 },
      { word: "AI动画", weight: 0.38 }
    ]
  },
  {
    topicId: 1,
    topicName: "AI工具测评",
    keywords: [
      { word: "ChatGPT", weight: 0.88 },
      { word: "测评", weight: 0.75 },
      { word: "对比", weight: 0.68 },
      { word: "推荐", weight: 0.62 },
      { word: "工具", weight: 0.55 },
      { word: "Claude", weight: 0.48 },
      { word: "使用体验", weight: 0.42 },
      { word: "神器", weight: 0.35 }
    ]
  },
  {
    topicId: 2,
    topicName: "AI学习教程",
    keywords: [
      { word: "教程", weight: 0.82 },
      { word: "学习", weight: 0.75 },
      { word: "入门", weight: 0.68 },
      { word: "Prompt", weight: 0.62 },
      { word: "课程", weight: 0.55 },
      { word: "技巧", weight: 0.48 },
      { word: "零基础", weight: 0.42 },
      { word: "实战", weight: 0.35 }
    ]
  },
  {
    topicId: 3,
    topicName: "AI生活融合",
    keywords: [
      { word: "效率", weight: 0.78 },
      { word: "AI办公", weight: 0.72 },
      { word: "工作", weight: 0.65 },
      { word: "副业", weight: 0.58 },
      { word: "变现", weight: 0.52 },
      { word: "AI写作", weight: 0.45 },
      { word: "职场", weight: 0.38 },
      { word: "打工", weight: 0.32 }
    ]
  },
  {
    topicId: 4,
    topicName: "AI社会反思",
    keywords: [
      { word: "AI伦理", weight: 0.72 },
      { word: "焦虑", weight: 0.65 },
      { word: "失业", weight: 0.58 },
      { word: "未来", weight: 0.52 },
      { word: "反思", weight: 0.45 },
      { word: "监管", weight: 0.38 },
      { word: "隐私", weight: 0.32 },
      { word: "安全", weight: 0.28 }
    ]
  }
];

// UMAP降维后的2D坐标数据（用于散点图）
export interface UmapPoint {
  x: number;
  y: number;
  topicId: number;
  topicName: string;
  noteCount: number;
}

export const umapPoints: UmapPoint[] = [
  // 主题0: AI内容创作 (聚集在左上区域)
  { x: 2.1, y: 8.2, topicId: 0, topicName: "AI内容创作", noteCount: 2341 },
  { x: 2.3, y: 8.5, topicId: 0, topicName: "AI内容创作", noteCount: 1856 },
  { x: 1.9, y: 7.9, topicId: 0, topicName: "AI内容创作", noteCount: 1523 },
  { x: 2.5, y: 8.1, topicId: 0, topicName: "AI内容创作", noteCount: 1234 },
  { x: 2.2, y: 7.7, topicId: 0, topicName: "AI内容创作", noteCount: 987 },
  
  // 主题1: AI工具测评 (聚集在右上区域)
  { x: 7.8, y: 7.5, topicId: 1, topicName: "AI工具测评", noteCount: 2156 },
  { x: 8.1, y: 7.8, topicId: 1, topicName: "AI工具测评", noteCount: 1789 },
  { x: 7.5, y: 7.2, topicId: 1, topicName: "AI工具测评", noteCount: 1456 },
  { x: 8.3, y: 7.6, topicId: 1, topicName: "AI工具测评", noteCount: 1123 },
  { x: 7.9, y: 7.9, topicId: 1, topicName: "AI工具测评", noteCount: 876 },
  
  // 主题2: AI学习教程 (聚集在中间上方)
  { x: 5.2, y: 5.8, topicId: 2, topicName: "AI学习教程", noteCount: 1987 },
  { x: 5.5, y: 5.5, topicId: 2, topicName: "AI学习教程", noteCount: 1654 },
  { x: 4.9, y: 6.1, topicId: 2, topicName: "AI学习教程", noteCount: 1345 },
  { x: 5.8, y: 5.2, topicId: 2, topicName: "AI学习教程", noteCount: 1023 },
  { x: 5.1, y: 5.9, topicId: 2, topicName: "AI学习教程", noteCount: 876 },
  
  // 主题3: AI生活融合 (聚集在右下区域)
  { x: 8.5, y: 2.8, topicId: 3, topicName: "AI生活融合", noteCount: 1876 },
  { x: 8.2, y: 2.5, topicId: 3, topicName: "AI生活融合", noteCount: 1543 },
  { x: 8.8, y: 3.1, topicId: 3, topicName: "AI生活融合", noteCount: 1234 },
  { x: 7.9, y: 2.2, topicId: 3, topicName: "AI生活融合", noteCount: 987 },
  { x: 8.1, y: 3.5, topicId: 3, topicName: "AI生活融合", noteCount: 765 },
  
  // 主题4: AI社会反思 (聚集在左下区域)
  { x: 1.5, y: 2.1, topicId: 4, topicName: "AI社会反思", noteCount: 1234 },
  { x: 1.8, y: 2.4, topicId: 4, topicName: "AI社会反思", noteCount: 987 },
  { x: 1.2, y: 1.8, topicId: 4, topicName: "AI社会反思", noteCount: 765 },
  { x: 2.1, y: 2.7, topicId: 4, topicName: "AI社会反思", noteCount: 543 },
  { x: 1.4, y: 3.0, topicId: 4, topicName: "AI社会反思", noteCount: 432 }
];

// 主题相似度矩阵
export const topicSimilarityMatrix = [
  [1.00, 0.45, 0.38, 0.28, 0.22], // AI内容创作
  [0.45, 1.00, 0.52, 0.35, 0.18], // AI工具测评
  [0.38, 0.52, 1.00, 0.42, 0.25], // AI学习教程
  [0.28, 0.35, 0.42, 1.00, 0.32], // AI生活融合
  [0.22, 0.18, 0.25, 0.32, 1.00]  // AI社会反思
];

// 主题分布统计
export const topicDistribution = [
  { name: "AI内容创作", count: 7941, ratio: 0.28 },
  { name: "AI工具测评", count: 7400, ratio: 0.26 },
  { name: "AI学习教程", count: 6885, ratio: 0.24 },
  { name: "AI生活融合", count: 6405, ratio: 0.22 },
  { name: "AI社会反思", count: 3961, ratio: 0.14 }
];
