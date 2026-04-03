import { useState } from 'react';
import { MessageCircle, TrendingUp, Heart, ThumbsDown, BarChart3, PieChart, Search, Filter, X, MessageSquare, Brain, Settings, BookOpen, Activity } from 'lucide-react';
import { 
  mockCommentTopics, 
  mockSentimentTrends, 
  mockCommentMetrics, 
  mockCommentAnalysis,
  mockSentimentDistribution,
  mockCommentWordCloud,
  CommentTopic,
  CommentAnalysis,
  robertaConfig,
  evaluationMetrics,
  confusionMatrix,
  classMetrics,
  confidenceDistribution,
  trainingHistory,
  predictionSamples,
  RoBERTaConfig,
  EvaluationMetrics
} from '../data/commentData';

type ViewMode = 'overview' | 'sentiment' | 'topics' | 'trends' | 'details' | 'model';

export default function CommentAnalysisPage() {
  const [viewMode, setViewMode] = useState<ViewMode>('overview');
  const [selectedTopic, setSelectedTopic] = useState<CommentTopic | null>(null);
  const [selectedNoteAnalysis, setSelectedNoteAnalysis] = useState<CommentAnalysis | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState<'all' | 'positive' | 'neutral' | 'negative'>('all');

  // 筛选评论分析数据
  const filteredAnalysis = mockCommentAnalysis.filter(a => {
    const matchSearch = a.noteTitle.includes(searchQuery) || 
                       a.topicName.includes(searchQuery) ||
                       a.keywords.some(k => k.includes(searchQuery));
    const matchSentiment = sentimentFilter === 'all' || a.sentiment === sentimentFilter;
    return matchSearch && matchSentiment;
  });

  // 情感筛选后的话题
  const filteredTopics = selectedTopic ? mockCommentTopics : 
    mockCommentTopics.filter(t => t.name.includes(searchQuery) || t.keywords.some(k => k.includes(searchQuery)));

  const metrics = mockCommentMetrics;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-cyan-50 to-blue-100">
      {/* 顶部导航栏 */}
      <header className="bg-white/80 backdrop-blur-md border-b border-blue-100 px-6 py-4 sticky top-0 z-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-lg">💬</span>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-500 to-cyan-600 bg-clip-text text-transparent">
                小红书AI主题评论分析
              </h1>
              <p className="text-xs text-gray-400">基于 BERTopic 模型的评论主题与情感分析</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* 搜索框 */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="搜索评论主题..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-300 focus:border-transparent w-64 transition-all"
              />
            </div>
            
            {/* 情感筛选 */}
            <select
              value={sentimentFilter}
              onChange={(e) => setSentimentFilter(e.target.value as any)}
              className="px-4 py-2 bg-gray-50 border border-gray-200 rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
            >
              <option value="all">全部情感</option>
              <option value="positive">积极</option>
              <option value="neutral">中性</option>
              <option value="negative">消极</option>
            </select>
          </div>
        </div>
        
        {/* 统计概览 */}
        <div className="flex items-center gap-6 mt-4 pt-4 border-t border-blue-50">
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-blue-500">{metrics.totalComments.toLocaleString()}</span>
            <span className="text-sm text-gray-500">总评论</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-green-500">{(metrics.positiveRatio * 100).toFixed(0)}%</span>
            <span className="text-sm text-gray-500">积极</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-gray-500">{(metrics.neutralRatio * 100).toFixed(0)}%</span>
            <span className="text-sm text-gray-500">中性</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-red-500">{(metrics.negativeRatio * 100).toFixed(0)}%</span>
            <span className="text-sm text-gray-500">消极</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-cyan-500">{metrics.avgSentimentScore.toFixed(2)}</span>
            <span className="text-sm text-gray-500">情感得分</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-blue-400">{metrics.replyRate * 100}%</span>
            <span className="text-sm text-gray-500">回复率</span>
          </div>
        </div>
      </header>

      <main className="flex h-[calc(100vh-180px)]">
        {/* 左侧评论主题列表 */}
        <aside className="w-72 bg-white/60 backdrop-blur-sm border-r border-blue-100 overflow-y-auto p-4">
          <h2 className="font-semibold text-gray-700 mb-4 flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-blue-500" />
            评论主题
          </h2>
          <div className="space-y-2">
            {mockCommentTopics.map((topic, index) => (
              <div
                key={topic.id}
                onClick={() => setSelectedTopic(topic)}
                className={`p-3 rounded-xl cursor-pointer transition-all duration-200 hover:shadow-md ${
                  selectedTopic?.id === topic.id
                    ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg'
                    : 'bg-white hover:bg-blue-50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-sm">{topic.name}</span>
                  <span className={`text-xs ${selectedTopic?.id === topic.id ? 'text-white/80' : 'text-gray-400'}`}>
                    {topic.commentCount.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {/* 情感条 */}
                  <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full bg-green-500" style={{ width: `${topic.positiveRatio * 100}%` }} />
                  </div>
                  <span className={`text-xs ${selectedTopic?.id === topic.id ? 'text-white/80' : 'text-green-500'}`}>
                    {(topic.positiveRatio * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* 中间主可视化区域 */}
        <section className="flex-1 p-6 overflow-y-auto">
          {/* 视图切换 */}
          <div className="flex items-center gap-2 mb-6">
            <button
              onClick={() => setViewMode('overview')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'overview'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-white text-gray-600 hover:bg-blue-50'
              }`}
            >
              <PieChart className="w-4 h-4 inline-block mr-2" />
              评论概览
            </button>
            <button
              onClick={() => setViewMode('sentiment')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'sentiment'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-white text-gray-600 hover:bg-blue-50'
              }`}
            >
              <Heart className="w-4 h-4 inline-block mr-2" />
              情感分布
            </button>
            <button
              onClick={() => setViewMode('topics')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'topics'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-white text-gray-600 hover:bg-blue-50'
              }`}
            >
              <MessageCircle className="w-4 h-4 inline-block mr-2" />
              评论主题
            </button>
            <button
              onClick={() => setViewMode('trends')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'trends'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-white text-gray-600 hover:bg-blue-50'
              }`}
            >
              <TrendingUp className="w-4 h-4 inline-block mr-2" />
              情感趋势
            </button>
            <button
              onClick={() => setViewMode('details')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'details'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-white text-gray-600 hover:bg-blue-50'
              }`}
            >
              <BarChart3 className="w-4 h-4 inline-block mr-2" />
              详细数据
            </button>
            <button
              onClick={() => setViewMode('model')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'model'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-white text-gray-600 hover:bg-blue-50'
              }`}
            >
              <Brain className="w-4 h-4 inline-block mr-2" />
              RoBERTa模型
            </button>
          </div>

          {/* 可视化内容 */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 min-h-[500px]">
            {viewMode === 'overview' && <OverviewView data={mockCommentAnalysis} />}
            {viewMode === 'sentiment' && <SentimentView />}
            {viewMode === 'topics' && <CommentTopicsView topics={mockCommentTopics} />}
            {viewMode === 'trends' && <TrendsView data={mockSentimentTrends} />}
            {viewMode === 'details' && <DetailsView analysis={filteredAnalysis} onNoteClick={setSelectedNoteAnalysis} />}
            {viewMode === 'model' && <RoBERTaModelView 
              config={robertaConfig} 
              metrics={evaluationMetrics}
              confMatrix={confusionMatrix}
              classData={classMetrics}
              confidenceData={confidenceDistribution}
              history={trainingHistory}
              samples={predictionSamples}
            />}
          </div>
        </section>

        {/* 右侧详情面板 */}
        <aside className="w-96 bg-white/60 backdrop-blur-sm border-l border-blue-100 overflow-y-auto p-4">
          {selectedTopic ? (
            <TopicDetailPanel topic={selectedTopic} />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-gray-400">
              <div className="text-6xl mb-4">👈</div>
              <p>选择一个评论主题查看详情</p>
            </div>
          )}
        </aside>
      </main>

      {/* 评论分析详情弹窗 */}
      {selectedNoteAnalysis && (
        <NoteCommentModal analysis={selectedNoteAnalysis} onClose={() => setSelectedNoteAnalysis(null)} />
      )}
    </div>
  );
}

// 评论概览组件
function OverviewView({ data }: { data: CommentAnalysis[] }) {
  const sentimentCounts = {
    positive: data.filter(d => d.sentiment === 'positive').length,
    neutral: data.filter(d => d.sentiment === 'neutral').length,
    negative: data.filter(d => d.sentiment === 'negative').length
  };

  return (
    <div className="h-full">
      <h3 className="text-lg font-semibold text-gray-700 mb-6">评论分析概览</h3>
      <div className="grid grid-cols-2 gap-6">
        {/* 情感分布饼图 */}
        <div className="flex flex-col items-center">
          <svg viewBox="0 0 300 300" className="w-64 h-64">
            {(() => {
              let currentAngle = 0;
              const colors = ['#22c55e', '#a1a1aa', '#ef4444'];
              const values = [sentimentCounts.positive, sentimentCounts.neutral, sentimentCounts.negative];
              const total = values.reduce((a, b) => a + b, 0);
              
              return values.map((val, i) => {
                const angle = (val / total) * 360;
                const startAngle = currentAngle;
                const endAngle = currentAngle + angle;
                currentAngle = endAngle;
                
                const startRad = (startAngle - 90) * Math.PI / 180;
                const endRad = (endAngle - 90) * Math.PI / 180;
                
                const x1 = 150 + 80 * Math.cos(startRad);
                const y1 = 150 + 80 * Math.sin(startRad);
                const x2 = 150 + 80 * Math.cos(endRad);
                const y2 = 150 + 80 * Math.sin(endRad);
                
                const largeArc = angle > 180 ? 1 : 0;
                
                return (
                  <path
                    key={i}
                    d={`M 150 150 L ${x1} ${y1} A 80 80 0 ${largeArc} 1 ${x2} ${y2} Z`}
                    fill={colors[i]}
                    className="cursor-pointer hover:opacity-80 transition-all"
                  />
                );
              });
            })()}
            <text x="150" y="145" textAnchor="middle" className="text-2xl font-bold fill-gray-700">
              {data.length}
            </text>
            <text x="150" y="170" textAnchor="middle" className="text-sm fill-gray-500">
              条笔记评论
            </text>
          </svg>
          <div className="flex gap-4 mt-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-600">积极 {sentimentCounts.positive}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
              <span className="text-sm text-gray-600">中性 {sentimentCounts.neutral}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span className="text-sm text-gray-600">消极 {sentimentCounts.negative}</span>
            </div>
          </div>
        </div>
        
        {/* 关键词云 */}
        <div>
          <h4 className="font-semibold text-gray-700 mb-4">高频评论关键词</h4>
          <div className="flex flex-wrap gap-2">
            {mockCommentWordCloud.map((item, i) => (
              <span
                key={i}
                className="px-3 py-1 rounded-full text-sm"
                style={{
                  fontSize: `${Math.max(12, Math.min(24, item.weight / 8))}px`,
                  backgroundColor: `rgba(59, 130, 246, ${0.1 + (item.weight / 200)})`,
                  color: '#1e40af'
                }}
              >
                {item.text}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// 情感分布组件
function SentimentView() {
  return (
    <div className="h-full">
      <h3 className="text-lg font-semibold text-gray-700 mb-6">情感分布详情</h3>
      <div className="grid grid-cols-3 gap-6">
        {/* 积极情感 */}
        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center">
              <Heart className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-3xl font-bold text-green-600">{(mockCommentMetrics.positiveRatio * 100).toFixed(0)}%</div>
              <div className="text-sm text-green-600/80">积极评论</div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-sm text-gray-600">典型特征：</div>
            <div className="flex flex-wrap gap-1">
              {["推荐", "好用", "赞", "感谢", "学习"].map(kw => (
                <span key={kw} className="px-2 py-0.5 bg-green-200 text-green-700 rounded text-xs">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        </div>
        
        {/* 中性情感 */}
        <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gray-400 rounded-xl flex items-center justify-center">
              <MessageCircle className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-3xl font-bold text-gray-600">{(mockCommentMetrics.neutralRatio * 100).toFixed(0)}%</div>
              <div className="text-sm text-gray-500">中性评论</div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-sm text-gray-600">典型特征：</div>
            <div className="flex flex-wrap gap-1">
              {["询问", "中立", "客观", "陈述"].map(kw => (
                <span key={kw} className="px-2 py-0.5 bg-gray-200 text-gray-600 rounded text-xs">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        </div>
        
        {/* 消极情感 */}
        <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-red-500 rounded-xl flex items-center justify-center">
              <ThumbsDown className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-3xl font-bold text-red-600">{(mockCommentMetrics.negativeRatio * 100).toFixed(0)}%</div>
              <div className="text-sm text-red-600/80">消极评论</div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-sm text-gray-600">典型特征：</div>
            <div className="flex flex-wrap gap-1">
              {["问题", "Bug", "无效", "报错"].map(kw => (
                <span key={kw} className="px-2 py-0.5 bg-red-200 text-red-700 rounded text-xs">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      {/* 情感得分 */}
      <div className="mt-8 bg-white rounded-2xl p-6">
        <h4 className="font-semibold text-gray-700 mb-4">整体情感得分</h4>
        <div className="flex items-center gap-4">
          <div className="text-4xl font-bold text-cyan-600">{mockCommentMetrics.avgSentimentScore.toFixed(2)}</div>
          <div className="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
              style={{ width: '100%' }}
            />
          </div>
          <div className="text-sm text-gray-500">满分 1.0</div>
        </div>
      </div>
    </div>
  );
}

// 评论主题组件
function CommentTopicsView({ topics }: { topics: CommentTopic[] }) {
  return (
    <div className="h-full overflow-auto">
      <h3 className="text-lg font-semibold text-gray-700 mb-6">评论主题分析</h3>
      <div className="space-y-4">
        {topics.map((topic, i) => (
          <div key={topic.id} className="bg-white rounded-xl p-4 hover:shadow-md transition-all">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 bg-blue-100 text-blue-600 rounded-lg flex items-center justify-center font-bold">
                  {i + 1}
                </span>
                <span className="font-semibold text-gray-700">{topic.name}</span>
              </div>
              <span className="text-sm text-gray-500">{topic.commentCount.toLocaleString()} 条评论</span>
            </div>
            
            {/* 情感比例条 */}
            <div className="flex h-2 rounded-full overflow-hidden mb-3">
              <div 
                className="bg-green-500" 
                style={{ width: `${topic.positiveRatio * 100}%` }}
              />
              <div 
                className="bg-gray-400" 
                style={{ width: `${topic.neutralRatio * 100}%` }}
              />
              <div 
                className="bg-red-500" 
                style={{ width: `${topic.negativeRatio * 100}%` }}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex flex-wrap gap-1">
                {topic.keywords.map(kw => (
                  <span key={kw} className="px-2 py-0.5 bg-blue-50 text-blue-600 rounded text-xs">
                    {kw}
                  </span>
                ))}
              </div>
              <div className="text-sm">
                <span className="text-green-500">积极 {(topic.positiveRatio * 100).toFixed(0)}%</span>
                <span className="text-gray-300 mx-2">|</span>
                <span className="text-gray-500">{(topic.avgSentimentScore).toFixed(2)} 分</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// 趋势分析组件
function TrendsView({ data }: { data: typeof mockSentimentTrends }) {
  const maxVal = Math.max(...data.map(d => d.positive + d.neutral + d.negative));
  
  return (
    <div className="h-full">
      <h3 className="text-lg font-semibold text-gray-700 mb-6">情感趋势分析 (7日)</h3>
      
      {/* 堆叠柱状图 */}
      <div className="flex items-end justify-between h-64 gap-2 mb-6">
        {data.map((day, i) => (
          <div key={i} className="flex-1 flex flex-col items-center">
            <div className="w-full flex flex-col-reverse h-56">
              <div 
                className="bg-red-400 rounded-t-sm transition-all hover:opacity-80"
                style={{ height: `${(day.negative / maxVal) * 100}%` }}
              />
              <div 
                className="bg-gray-400 transition-all hover:opacity-80"
                style={{ height: `${(day.neutral / maxVal) * 100}%` }}
              />
              <div 
                className="bg-green-500 rounded-t-sm transition-all hover:opacity-80"
                style={{ height: `${(day.positive / maxVal) * 100}%` }}
              />
            </div>
            <span className="text-xs text-gray-500 mt-2">{day.date}</span>
          </div>
        ))}
      </div>
      
      <div className="flex justify-center gap-6">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="text-sm text-gray-600">积极</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
          <span className="text-sm text-gray-600">中性</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-400 rounded-full"></div>
          <span className="text-sm text-gray-600">消极</span>
        </div>
      </div>
    </div>
  );
}

// 详细数据组件
function DetailsView({ analysis, onNoteClick }: { analysis: CommentAnalysis[]; onNoteClick: (a: CommentAnalysis) => void }) {
  return (
    <div className="h-full overflow-auto">
      <h3 className="text-lg font-semibold text-gray-700 mb-6">笔记评论详情</h3>
      <div className="space-y-3">
        {analysis.map(item => (
          <div 
            key={item.id}
            onClick={() => onNoteClick(item)}
            className="bg-white rounded-xl p-4 cursor-pointer hover:shadow-md transition-all"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-gray-700">{item.noteTitle}</span>
              <span className={`px-2 py-0.5 rounded text-xs ${
                item.sentiment === 'positive' ? 'bg-green-100 text-green-600' :
                item.sentiment === 'negative' ? 'bg-red-100 text-red-600' :
                'bg-gray-100 text-gray-600'
              }`}>
                {item.sentiment === 'positive' ? '积极' : item.sentiment === 'negative' ? '消极' : '中性'}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm text-gray-500">
              <span>{item.topicName}</span>
              <div className="flex items-center gap-3">
                <span>💬 {item.commentCount}</span>
                <span>❤️ {item.avgCommentLikes}</span>
                <span>📊 {(item.sentimentScore * 100).toFixed(0)}分</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// 主题详情面板
function TopicDetailPanel({ topic }: { topic: CommentTopic }) {
  return (
    <div>
      <div className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl p-4 text-white mb-4">
        <h2 className="text-xl font-bold mb-2">{topic.name}</h2>
        <div className="flex flex-wrap gap-2 mb-3">
          {topic.keywords.map(kw => (
            <span key={kw} className="px-2 py-1 bg-white/20 rounded-full text-xs">
              {kw}
            </span>
          ))}
        </div>
        <div className="text-sm text-white/80">
          {topic.commentCount.toLocaleString()} 条评论
        </div>
      </div>
      
      {/* 情感分布 */}
      <div className="bg-white rounded-xl p-4 mb-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">情感分布</h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">积极</span>
            <span className="text-sm text-green-600">{(topic.positiveRatio * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full bg-green-500" style={{ width: `${topic.positiveRatio * 100}%` }} />
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">中性</span>
            <span className="text-sm text-gray-500">{(topic.neutralRatio * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full bg-gray-400" style={{ width: `${topic.neutralRatio * 100}%` }} />
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">消极</span>
            <span className="text-sm text-red-500">{(topic.negativeRatio * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full bg-red-500" style={{ width: `${topic.negativeRatio * 100}%` }} />
          </div>
        </div>
      </div>
      
      {/* 情感得分 */}
      <div className="bg-white rounded-xl p-4 mb-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">平均情感得分</h3>
        <div className="text-3xl font-bold text-center text-cyan-600">
          {topic.avgSentimentScore.toFixed(2)}
        </div>
      </div>
      
      {/* 趋势 */}
      <div className="bg-white rounded-xl p-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">7日趋势</h3>
        <div className="flex items-end gap-1 h-20">
          {topic.trend.map((val, i) => (
            <div key={i} className="flex-1 bg-gradient-to-t from-blue-400 to-cyan-400 rounded-t transition-all hover:opacity-80"
                 style={{ height: `${(val / Math.max(...topic.trend)) * 100}%` }} />
          ))}
        </div>
      </div>
    </div>
  );
}

// 评论分析详情弹窗
function NoteCommentModal({ analysis, onClose }: { analysis: CommentAnalysis; onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
         onClick={onClose}>
      <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden"
           onClick={e => e.stopPropagation()}>
        {/* 弹窗头部 */}
        <div className="bg-gradient-to-r from-blue-500 to-cyan-500 px-6 py-4 flex items-center justify-between">
          <div>
            <h2 className="text-white font-semibold text-lg">{analysis.noteTitle}</h2>
            <p className="text-white/80 text-sm">{analysis.topicName} · 情感: {analysis.sentiment}</p>
          </div>
          <button onClick={onClose} className="text-white/80 hover:text-white">
            <X className="w-6 h-6" />
          </button>
        </div>
        
        {/* 内容 */}
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          {/* 统计 */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-blue-50 rounded-xl p-3 text-center">
              <div className="text-2xl font-bold text-blue-600">{analysis.commentCount}</div>
              <div className="text-xs text-gray-500">评论数</div>
            </div>
            <div className="bg-green-50 rounded-xl p-3 text-center">
              <div className="text-2xl font-bold text-green-600">{analysis.avgCommentLikes}</div>
              <div className="text-xs text-gray-500">平均点赞</div>
            </div>
            <div className="bg-cyan-50 rounded-xl p-3 text-center">
              <div className="text-2xl font-bold text-cyan-600">{(analysis.sentimentScore * 100).toFixed(0)}</div>
              <div className="text-xs text-gray-500">情感得分</div>
            </div>
            <div className="bg-purple-50 rounded-xl p-3 text-center">
              <div className="text-2xl font-bold text-purple-600">{analysis.keywords.length}</div>
              <div className="text-xs text-gray-500">关键词数</div>
            </div>
          </div>
          
          {/* 关键词 */}
          <div className="mb-6">
            <h3 className="font-semibold text-gray-700 mb-2">关键词</h3>
            <div className="flex flex-wrap gap-2">
              {analysis.keywords.map(kw => (
                <span key={kw} className="px-2 py-1 bg-blue-100 text-blue-600 rounded text-sm">
                  {kw}
                </span>
              ))}
            </div>
          </div>
          
          {/* 热门评论 */}
          <div>
            <h3 className="font-semibold text-gray-700 mb-3">热门评论</h3>
            <div className="space-y-3">
              {analysis.topComments.map(comment => (
                <div key={comment.id} className="bg-gray-50 rounded-xl p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm text-gray-700">{comment.user}</span>
                    <div className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${
                        comment.sentiment === 'positive' ? 'bg-green-100 text-green-600' :
                        comment.sentiment === 'negative' ? 'bg-red-100 text-red-600' :
                        'bg-gray-200 text-gray-600'
                      }`}>
                        {comment.sentiment === 'positive' ? '积极' : comment.sentiment === 'negative' ? '消极' : '中性'}
                      </span>
                      <span className="text-xs text-gray-400">{comment.createdAt}</span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600">{comment.content}</p>
                  <div className="mt-1 text-xs text-gray-400">👍 {comment.likes}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// RoBERTa 模型训练结果可视化组件
// ============================================================

function RoBERTaModelView({
  config,
  metrics,
  confMatrix,
  classData,
  confidenceData,
  history,
  samples
}: {
  config: RoBERTaConfig;
  metrics: EvaluationMetrics;
  confMatrix: number[][];
  classData: { name: string; precision: number; recall: number; f1: number; support: number }[];
  confidenceData: { range: string; count: number; accuracy: number }[];
  history: { epoch: number; trainLoss: number; valLoss: number; valAcc: number; valF1: number }[];
  samples: { id: number; content: string; trueLabel: string; predLabel: string; confidence: number; correct: boolean }[];
}) {
  const [subView, setSubView] = useState<'overview' | 'metrics' | 'training' | 'samples'>('overview');
  
  const sentimentColors = ['#22c55e', '#a1a1aa', '#ef4444'];
  
  return (
    <div className="h-full">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-700">RoBERTa 情感分析模型训练结果</h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSubView('overview')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              subView === 'overview' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-blue-50'
            }`}
          >
            📊 模型概览
          </button>
          <button
            onClick={() => setSubView('metrics')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              subView === 'metrics' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-blue-50'
            }`}
          >
            📈 评估指标
          </button>
          <button
            onClick={() => setSubView('training')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              subView === 'training' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-blue-50'
            }`}
          >
            🔥 训练过程
          </button>
          <button
            onClick={() => setSubView('samples')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              subView === 'samples' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-blue-50'
            }`}
          >
            📝 预测样例
          </button>
        </div>
      </div>
      
      <div className="h-[calc(100%-60px)] overflow-y-auto">
        {subView === 'overview' && <ModelOverviewView config={config} metrics={metrics} />}
        {subView === 'metrics' && <MetricsDetailView metrics={metrics} confMatrix={confMatrix} classData={classData} confidenceData={confidenceData} />}
        {subView === 'training' && <TrainingHistoryView history={history} />}
        {subView === 'samples' && <PredictionSamplesView samples={samples} />}
      </div>
    </div>
  );
}

// 模型概览视图
function ModelOverviewView({ config, metrics }: { config: RoBERTaConfig; metrics: EvaluationMetrics }) {
  return (
    <div className="space-y-6">
      {/* 核心指标卡片 */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 border border-blue-200">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-600">准确率</span>
          </div>
          <div className="text-3xl font-bold text-blue-700">{(metrics.accuracy * 100).toFixed(2)}%</div>
        </div>
        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-4 border border-green-200">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium text-green-600">Macro F1</span>
          </div>
          <div className="text-3xl font-bold text-green-700">{(metrics.macroF1 * 100).toFixed(2)}%</div>
        </div>
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-4 border border-purple-200">
          <div className="flex items-center gap-2 mb-2">
            <PieChart className="w-4 h-4 text-purple-600" />
            <span className="text-sm font-medium text-purple-600">Macro AUC</span>
          </div>
          <div className="text-3xl font-bold text-purple-700">{(metrics.macroAuc * 100).toFixed(2)}%</div>
        </div>
        <div className="bg-gradient-to-br from-cyan-50 to-cyan-100 rounded-xl p-4 border border-cyan-200">
          <div className="flex items-center gap-2 mb-2">
            <Brain className="w-4 h-4 text-cyan-600" />
            <span className="text-sm font-medium text-cyan-600">情感类别</span>
          </div>
          <div className="text-3xl font-bold text-cyan-700">{config.numLabels} 类</div>
        </div>
      </div>
      
      {/* 模型配置信息 */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <Settings className="w-4 h-4 text-blue-500" />
            模型配置
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">基础模型</span>
              <span className="text-gray-700 font-medium">{config.baseModel}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">序列长度</span>
              <span className="text-gray-700 font-medium">{config.maxLength}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">批次大小</span>
              <span className="text-gray-700 font-medium">{config.batchSize}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">学习率</span>
              <span className="text-gray-700 font-medium">{config.learningRate}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">训练轮次</span>
              <span className="text-gray-700 font-medium">{config.epochs}</span>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-green-500" />
            数据集规模
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">训练集</span>
              <span className="text-gray-700 font-medium">{config.trainSize} 条</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">验证集</span>
              <span className="text-gray-700 font-medium">{config.valSize} 条</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">测试集</span>
              <span className="text-gray-700 font-medium">{config.testSize} 条</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">训练日期</span>
              <span className="text-gray-700 font-medium">{config.trainingDate}</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* 简要说明 */}
      <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl p-4 border border-blue-100">
        <p className="text-sm text-gray-600">
          💡 <strong>模型说明：</strong>本模型基于哈工大预训练的中文 RoBERTa-wwm-ext 模型进行微调，<br/>
          针对小红书评论数据进行3分类情感分析（正面/中立/负面）。模型在测试集上达到了 87.23% 的准确率。
        </p>
      </div>
    </div>
  );
}

// 评估指标详细视图
function MetricsDetailView({
  metrics,
  confMatrix,
  classData,
  confidenceData
}: {
  metrics: EvaluationMetrics;
  confMatrix: number[][];
  classData: { name: string; precision: number; recall: number; f1: number; support: number }[];
  confidenceData: { range: string; count: number; accuracy: number }[];
}) {
  const sentimentNames = ['正面/积极', '中立/客观', '负面/消极'];
  const sentimentColors = ['#22c55e', '#a1a1aa', '#ef4444'];
  
  // 计算总样本数
  const total = confMatrix.flat().reduce((a, b) => a + b, 0);
  
  return (
    <div className="space-y-6">
      {/* 混淆矩阵 + 各类别指标 */}
      <div className="grid grid-cols-2 gap-6">
        {/* 混淆矩阵 */}
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-4">混淆矩阵</h4>
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-xs text-gray-500 p-1"></th>
                {sentimentNames.map(name => (
                  <th key={name} className="text-xs text-gray-500 p-1 text-center">{name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confMatrix.map((row, i) => (
                <tr key={i}>
                  <td className="text-xs text-gray-600 p-1 font-medium">{sentimentNames[i]}</td>
                  {row.map((val, j) => {
                    const percentage = (val / total) * 100;
                    const isCorrect = i === j;
                    return (
                      <td key={j} className="p-1">
                        <div 
                          className={`text-center text-xs font-medium py-2 rounded ${
                            isCorrect 
                              ? `bg-${sentimentColors[i].replace('#', '')}-100 text-${sentimentColors[i].replace('#', '')}-700` 
                              : 'bg-red-50 text-red-600'
                          }`}
                          style={{ backgroundColor: isCorrect ? `${sentimentColors[i]}20` : '#fef2f2', color: isCorrect ? sentimentColors[i] : '#dc2626' }}
                        >
                          {val}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {/* 各类别详细指标 */}
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-4">各类别评估指标</h4>
          <div className="space-y-3">
            {classData.map((cls, i) => (
              <div key={cls.name} className="p-3 rounded-lg" style={{ backgroundColor: `${sentimentColors[i]}10` }}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium" style={{ color: sentimentColors[i] }}>{cls.name}</span>
                  <span className="text-xs text-gray-500">样本数: {cls.support}</span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center text-sm">
                  <div>
                    <div className="text-gray-500 text-xs">精确率</div>
                    <div className="font-semibold">{(cls.precision * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs">召回率</div>
                    <div className="font-semibold">{(cls.recall * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs">F1分数</div>
                    <div className="font-semibold">{(cls.f1 * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* 置信度分布 */}
      <div className="bg-white rounded-xl p-4 border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">置信度分布与准确率</h4>
        <div className="space-y-3">
          {confidenceData.map((item, i) => (
            <div key={item.range} className="flex items-center gap-3">
              <span className="w-16 text-xs text-gray-500">{item.range}</span>
              <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${(item.count / 500) * 100}%`,
                    backgroundColor: sentimentColors[i % 3]
                  }}
                />
              </div>
              <span className="w-20 text-xs text-gray-600">{item.count} 条</span>
              <span className="w-16 text-xs text-right" style={{ color: item.accuracy > 0.8 ? '#22c55e' : item.accuracy > 0.5 ? '#f59e0b' : '#ef4444' }}>
                准确率 {(item.accuracy * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-3">💡 高置信度预测往往具有更高的准确率</p>
      </div>
    </div>
  );
}

// 训练过程视图
function TrainingHistoryView({ history }: { history: { epoch: number; trainLoss: number; valLoss: number; valAcc: number; valF1: number }[] }) {
  const maxLoss = Math.max(...history.map(h => Math.max(h.trainLoss, h.valLoss)));
  
  return (
    <div className="space-y-6">
      {/* 训练指标变化 */}
      <div className="bg-white rounded-xl p-4 border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">训练过程指标变化</h4>
        
        {/* Loss曲线 */}
        <div className="mb-6">
          <h5 className="text-xs font-medium text-gray-500 mb-2">Loss 变化曲线</h5>
          <div className="flex items-end gap-2 h-32">
            {history.map((h, i) => (
              <div key={i} className="flex-1 flex flex-col items-center">
                <div className="w-full flex flex-col-reverse h-28">
                  <div 
                    className="bg-red-400 rounded-t transition-all"
                    style={{ height: `${(h.valLoss / maxLoss) * 100}%` }}
                    title={`验证Loss: ${h.valLoss.toFixed(3)}`}
                  />
                </div>
                <span className="text-xs text-gray-500 mt-1">Epoch {h.epoch}</span>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-4 mt-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-1 bg-red-400 rounded"></div>
              <span className="text-xs text-gray-500">验证Loss</span>
            </div>
          </div>
        </div>
        
        {/* 准确率和F1曲线 */}
        <div>
          <h5 className="text-xs font-medium text-gray-500 mb-2">准确率与F1变化曲线</h5>
          <div className="flex items-end gap-2 h-32">
            {history.map((h, i) => (
              <div key={i} className="flex-1 flex flex-col items-center">
                <div className="w-full flex flex-col-reverse h-28 gap-1">
                  <div 
                    className="bg-blue-500 rounded-t transition-all"
                    style={{ height: `${h.valAcc * 100}%` }}
                    title={`验证准确率: ${(h.valAcc * 100).toFixed(1)}%`}
                  />
                  <div 
                    className="bg-green-500 rounded-t transition-all"
                    style={{ height: `${h.valF1 * 100}%` }}
                    title={`验证F1: ${(h.valF1 * 100).toFixed(1)}%`}
                  />
                </div>
                <span className="text-xs text-gray-500 mt-1">Epoch {h.epoch}</span>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-4 mt-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-1 bg-blue-500 rounded"></div>
              <span className="text-xs text-gray-500">验证准确率</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-1 bg-green-500 rounded"></div>
              <span className="text-xs text-gray-500">验证F1</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* 训练数据表格 */}
      <div className="bg-white rounded-xl p-4 border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">训练指标详情</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-2 text-gray-500">Epoch</th>
              <th className="text-right py-2 text-gray-500">训练Loss</th>
              <th className="text-right py-2 text-gray-500">验证Loss</th>
              <th className="text-right py-2 text-gray-500">验证准确率</th>
              <th className="text-right py-2 text-gray-500">验证F1</th>
            </tr>
          </thead>
          <tbody>
            {history.map((h, i) => (
              <tr key={i} className="border-b border-gray-100">
                <td className="py-2 font-medium">{h.epoch}</td>
                <td className="py-2 text-right">{h.trainLoss.toFixed(4)}</td>
                <td className="py-2 text-right">{h.valLoss.toFixed(4)}</td>
                <td className="py-2 text-right text-blue-600 font-medium">{(h.valAcc * 100).toFixed(2)}%</td>
                <td className="py-2 text-right text-green-600 font-medium">{(h.valF1 * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// 预测样例视图
function PredictionSamplesView({ samples }: { samples: { id: number; content: string; trueLabel: string; predLabel: string; confidence: number; correct: boolean }[] }) {
  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl p-4 border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">预测样例展示</h4>
        <div className="space-y-3">
          {samples.map(sample => (
            <div 
              key={sample.id} 
              className={`p-3 rounded-lg border ${
                sample.correct 
                  ? 'bg-green-50 border-green-200' 
                  : 'bg-red-50 border-red-200'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <p className="text-sm text-gray-700">{sample.content}</p>
                </div>
                <div className={`w-6 h-6 rounded-full flex items-center justify-center ml-2 ${
                  sample.correct ? 'bg-green-500' : 'bg-red-500'
                }`}>
                  {sample.correct ? (
                    <span className="text-white text-xs">✓</span>
                  ) : (
                    <span className="text-white text-xs">✗</span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-4 text-xs">
                <span className="text-gray-500">
                  真实: <span className="font-medium">{sample.trueLabel}</span>
                </span>
                <span className="text-gray-500">
                  预测: <span className={`font-medium ${sample.correct ? 'text-green-600' : 'text-red-600'}`}>{sample.predLabel}</span>
                </span>
                <span className="text-gray-500">
                  置信度: <span className="font-medium">{(sample.confidence * 100).toFixed(1)}%</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* 模型特点总结 */}
      <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl p-4 border border-blue-100">
        <h4 className="text-sm font-semibold text-blue-700 mb-2">模型特点总结</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• 整体准确率达到 87.23%，模型表现良好</li>
          <li>• 正面情感识别精确率最高(89%)，表明用户积极反馈容易被准确捕捉</li>
          <li>• 中立情感召回率相对较低(79%)，存在一定误判风险</li>
          <li>• 高置信度预测(&gt;0.8)准确率达94%，低置信度预测建议人工复核</li>
        </ul>
      </div>
    </div>
  );
}
