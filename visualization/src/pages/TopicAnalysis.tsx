import { useState } from 'react';
import { Search, BarChart3, PieChart, TrendingUp, X, Network, ArrowLeft } from 'lucide-react';
import { 
  mockTopics, mockNotes, trendData, interactionMatrix, Note, Topic,
  bertopicConfig, bertopicKeywords, umapPoints, topicSimilarityMatrix, topicDistribution,
  BERTopicConfig, TopicKeywords, UmapPoint
} from '../data/topicData';
import { EmptyState } from '../components/common/States';
import { formatNumber } from '../utils/responsive';

type ViewMode = 'overview' | 'trend' | 'heatmap' | 'ranking' | 'bertopic';

export default function TopicAnalysis() {
  const [viewMode, setViewMode] = useState<ViewMode>('overview');
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [selectedNote, setSelectedNote] = useState<Note | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'noteCount' | 'avgLikes' | 'avgComments'>('noteCount');
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // 筛选和排序主题
  const filteredTopics = mockTopics
    .filter(t => t.name.includes(searchQuery) || t.keywords.some(k => k.includes(searchQuery)))
    .sort((a, b) => b[sortBy] - a[sortBy]);

  // 选中主题的笔记
  const topicNotes = selectedTopic 
    ? mockNotes.filter(n => n.topicId === selectedTopic.id).sort((a, b) => b.likes - a.likes)
    : [];

  // 总计数据
  const totalNotes = mockTopics.reduce((sum, t) => sum + t.noteCount, 0);
  const totalLikes = mockTopics.reduce((sum, t) => sum + t.avgLikes * t.noteCount, 0);
  const totalComments = mockTopics.reduce((sum, t) => sum + t.avgComments * t.noteCount, 0);
  const totalCollects = mockTopics.reduce((sum, t) => sum + t.avgCollects * t.noteCount, 0);

  // 视图切换按钮配置
  const viewButtons = [
    { id: 'overview', label: '主题概览', icon: <PieChart className="w-4 h-4" /> },
    { id: 'trend', label: '趋势分析', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'heatmap', label: '互动热力', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'ranking', label: '排行榜', icon: <span>🏆</span> },
    { id: 'bertopic', label: 'BERTopic', icon: <Network className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-50 via-pink-50 to-rose-100">
      {/* 顶部导航栏 - 响应式 */}
      <header className="bg-white/80 backdrop-blur-md border-b border-rose-100 px-3 sm:px-4 lg:px-6 py-3 sm:py-4 sticky top-14 sm:top-16 z-40">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 lg:gap-4">
          {/* 标题区 */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-rose-400 to-pink-500 rounded-xl flex items-center justify-center shadow-lg shrink-0">
              <span className="text-white font-bold text-sm sm:text-lg">📕</span>
            </div>
            <div>
              <h1 className="text-lg sm:text-xl font-bold bg-gradient-to-r from-rose-500 to-pink-600 bg-clip-text text-transparent">
                小红书AI主题研究分析
              </h1>
              <p className="text-xs text-gray-400 hidden sm:block">基于 BERTopic 模型的主题分析</p>
            </div>
          </div>
          
          {/* 搜索和排序 - 响应式 */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3">
            {/* 移动端主题切换按钮 */}
            <button 
              onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
              className="lg:hidden flex items-center justify-center gap-2 px-3 py-2 bg-rose-100 text-rose-600 rounded-lg text-sm font-medium"
            >
              <BarChart3 className="w-4 h-4" />
              {selectedTopic ? selectedTopic.name : '选择主题'}
            </button>
            
            {/* 搜索框 */}
            <div className="relative flex-1 sm:flex-none">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="搜索主题或关键词..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full sm:w-48 lg:w-64 pl-9 pr-3 py-2 bg-gray-50 border border-gray-200 rounded-lg sm:rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-rose-300 focus:border-transparent transition-all"
              />
            </div>
            
            {/* 排序选择 */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-3 sm:px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg sm:rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-rose-300"
            >
              <option value="noteCount">按笔记数量</option>
              <option value="avgLikes">按平均点赞</option>
              <option value="avgComments">按平均评论</option>
            </select>
          </div>
        </div>
        
        {/* 统计概览 - 响应式 */}
        <div className="flex flex-wrap items-center gap-3 sm:gap-6 mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-rose-50">
          <div className="flex items-center gap-2">
            <span className="text-xl sm:text-2xl font-bold text-rose-500">{mockTopics.length}</span>
            <span className="text-xs sm:text-sm text-gray-500">主题</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl sm:text-2xl font-bold text-pink-500">{formatNumber(totalNotes)}</span>
            <span className="text-xs sm:text-sm text-gray-500">笔记</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl sm:text-2xl font-bold text-rose-400">{formatNumber(totalLikes, { decimals: 1 })}</span>
            <span className="text-xs sm:text-sm text-gray-500">总点赞</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl sm:text-2xl font-bold text-pink-400">{formatNumber(totalComments, { decimals: 1 })}</span>
            <span className="text-xs sm:text-sm text-gray-500">总评论</span>
          </div>
          <div className="hidden sm:flex items-center gap-2">
            <span className="text-xl font-bold text-rose-300">{formatNumber(totalCollects, { decimals: 1 })}</span>
            <span className="text-sm text-gray-500">总收藏</span>
          </div>
        </div>
      </header>

      {/* 主内容区 - 响应式 */}
      <main className="flex flex-col lg:flex-row min-h-[calc(100vh-200px)]">
        {/* 左侧主题列表 - 移动端可折叠 */}
        <aside className={`
          lg:w-64 xl:w-72 bg-white/60 backdrop-blur-sm border-r border-rose-100 
          overflow-y-auto p-3 sm:p-4
          ${isMobileSidebarOpen ? 'block' : 'hidden'} 
          lg:block
          fixed lg:sticky top-36 lg:top-auto left-0 right-0 z-30 lg:z-auto
          max-h-[50vh] lg:max-h-none
          shadow-xl lg:shadow-none
        `}>
          <div className="flex items-center justify-between lg:hidden mb-3">
            <h2 className="font-semibold text-gray-700 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-rose-500" />
              主题列表
            </h2>
            <button onClick={() => setIsMobileSidebarOpen(false)} className="p-1">
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>
          
          {filteredTopics.length === 0 ? (
            <EmptyState 
              type="search" 
              title="未找到匹配主题"
              description="请尝试其他搜索关键词"
            />
          ) : (
            <div className="space-y-2">
              {filteredTopics.map((topic, index) => (
                <div
                  key={topic.id}
                  onClick={() => {
                    setSelectedTopic(topic);
                    setIsMobileSidebarOpen(false);
                  }}
                  className={`p-2.5 sm:p-3 rounded-xl cursor-pointer transition-all duration-200 hover:shadow-md ${
                    selectedTopic?.id === topic.id
                      ? 'bg-gradient-to-r from-rose-500 to-pink-500 text-white shadow-lg'
                      : 'bg-white hover:bg-rose-50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm truncate">{topic.name}</span>
                    <span className={`text-xs shrink-0 ${selectedTopic?.id === topic.id ? 'text-white/80' : 'text-gray-400'}`}>
                      #{index + 1}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <span className={selectedTopic?.id === topic.id ? 'text-white/80' : 'text-gray-500'}>
                      {formatNumber(topic.noteCount)} 篇
                    </span>
                    <span className={selectedTopic?.id === topic.id ? 'text-white/80' : 'text-pink-500'}>
                      ❤️ {formatNumber(topic.avgLikes)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </aside>

        {/* 中间主可视化区域 */}
        <section className="flex-1 p-3 sm:p-4 lg:p-6 overflow-y-auto">
          {/* 视图切换 - 响应式 */}
          <div className="flex flex-wrap items-center gap-2 mb-4 sm:mb-6">
            {viewButtons.map(btn => (
              <button
                key={btn.id}
                onClick={() => setViewMode(btn.id as ViewMode)}
                className={`flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  viewMode === btn.id
                    ? 'bg-rose-500 text-white shadow-lg'
                    : 'bg-white text-gray-600 hover:bg-rose-50'
                }`}
              >
                {btn.icon}
                <span className="hidden xs:inline">{btn.label}</span>
              </button>
            ))}
          </div>

          {/* 可视化内容 */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-4 sm:p-6 min-h-[400px]">
            {filteredTopics.length === 0 ? (
              <EmptyState type="search" />
            ) : (
              <>
                {viewMode === 'overview' && <OverviewView topics={filteredTopics} />}
                {viewMode === 'trend' && <TrendView data={trendData} />}
                {viewMode === 'heatmap' && <HeatmapView data={interactionMatrix} />}
                {viewMode === 'ranking' && <RankingView topics={filteredTopics} />}
                {viewMode === 'bertopic' && <BERTopicView 
                  config={bertopicConfig} 
                  keywords={bertopicKeywords}
                  umapPoints={umapPoints}
                  similarityMatrix={topicSimilarityMatrix}
                  distribution={topicDistribution}
                />}
              </>
            )}
          </div>
        </section>

        {/* 右侧详情面板 - 响应式 */}
        <aside className={`
          lg:w-80 xl:w-96 bg-white/60 backdrop-blur-sm border-l border-rose-100 
          overflow-y-auto p-3 sm:p-4
          ${selectedTopic ? 'block' : 'hidden'}
          lg:block
        `}>
          {selectedTopic ? (
            <TopicDetailPanel 
              topic={selectedTopic} 
              notes={topicNotes}
              onNoteClick={setSelectedNote}
              onBack={() => setSelectedTopic(null)}
            />
          ) : (
            <div className="hidden lg:flex flex-col items-center justify-center h-full text-gray-400 py-12">
              <div className="text-5xl mb-4">👈</div>
              <p className="text-center">选择一个主题查看详情</p>
            </div>
          )}
        </aside>
      </main>

      {/* 笔记评论弹窗 */}
      {selectedNote && (
        <NoteModal note={selectedNote} onClose={() => setSelectedNote(null)} />
      )}
    </div>
  );
}

// ============ 子组件 ============

function OverviewView({ topics }: { topics: Topic[] }) {
  const colors = ['#f43f5e', '#ec4899', '#d946ef', '#a855f7', '#8b5cf6', '#6366f1', '#3b82f6', '#06b6d4', '#14b8a6', '#22c55e'];
  
  return (
    <div>
      <h3 className="text-base sm:text-lg font-semibold text-gray-700 mb-4 sm:mb-6">主题分布与关键词</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
        <div className="relative">
          <svg viewBox="0 0 400 400" className="w-full h-auto max-w-[300px] mx-auto">
            {topics.reduce((acc, topic, i) => {
              const startAngle = acc.angle;
              const angle = (topic.noteCount / topics.reduce((s, t) => s + t.noteCount, 0)) * 360;
              const endAngle = startAngle + angle;
              const startRad = (startAngle - 90) * Math.PI / 180;
              const endRad = (endAngle - 90) * Math.PI / 180;
              const x1 = 200 + 120 * Math.cos(startRad);
              const y1 = 200 + 120 * Math.sin(startRad);
              const x2 = 200 + 120 * Math.cos(endRad);
              const y2 = 200 + 120 * Math.sin(endRad);
              const largeArc = angle > 180 ? 1 : 0;
              acc.elements.push(
                <path
                  key={topic.id}
                  d={`M 200 200 L ${x1} ${y1} A 120 120 0 ${largeArc} 1 ${x2} ${y2} Z`}
                  fill={colors[i % colors.length]}
                  className="cursor-pointer transition-all hover:opacity-80"
                />
              );
              acc.angle = endAngle;
              return acc;
            }, { angle: 0, elements: [] as JSX.Element[] }).elements}
            <text x="200" y="190" textAnchor="middle" className="text-3xl font-bold fill-gray-700">
              {topics.length}
            </text>
            <text x="200" y="220" textAnchor="middle" className="text-sm fill-gray-500">个主题</text>
          </svg>
        </div>
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {topics.map((topic, i) => (
            <div key={topic.id} className="flex items-center gap-3 p-2 sm:p-3 rounded-xl bg-gradient-to-r from-gray-50 to-white hover:shadow-md transition-all">
              <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: colors[i % colors.length] }} />
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm text-gray-700 truncate">{topic.name}</div>
                <div className="text-xs text-gray-400">{formatNumber(topic.noteCount)} 篇</div>
              </div>
              <div className="flex flex-wrap gap-1 max-w-[120px] shrink-0">
                {topic.keywords.slice(0, 3).map(kw => (
                  <span key={kw} className="px-1.5 py-0.5 bg-rose-100 text-rose-600 text-xs rounded">{kw}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TrendView({ data }: { data: typeof trendData }) {
  const maxNotes = Math.max(...data.map(d => d.notes));
  return (
    <div>
      <h3 className="text-base sm:text-lg font-semibold text-gray-700 mb-4 sm:mb-6">7日趋势分析</h3>
      <div className="flex items-end justify-between h-48 sm:h-64 gap-2 sm:gap-4">
        {data.map((day, i) => (
          <div key={i} className="flex-1 flex flex-col items-center">
            <div className="w-full bg-gradient-to-t from-rose-400 to-pink-400 rounded-t-lg" style={{ height: `${Math.max((day.notes / maxNotes) * 150, 10)}px` }} />
            <span className="text-xs text-gray-500 mt-2">{day.date}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HeatmapView({ data }: { data: typeof interactionMatrix }) {
  const maxVal = Math.max(...data.map(d => Math.max(d.likes, d.comments, d.collects, d.shares, d.discusses)));
  const getColor = (val: number) => {
    const intensity = val / maxVal;
    if (intensity > 0.8) return 'bg-rose-500';
    if (intensity > 0.6) return 'bg-rose-400';
    if (intensity > 0.4) return 'bg-pink-400';
    if (intensity > 0.2) return 'bg-pink-300';
    return 'bg-pink-200';
  };
  return (
    <div className="overflow-x-auto">
      <h3 className="text-base sm:text-lg font-semibold text-gray-700 mb-4 sm:mb-6">主题互动热力图</h3>
      <table className="w-full min-w-[500px]">
        <thead>
          <tr>
            <th className="text-left text-xs sm:text-sm font-medium text-gray-500 p-2">主题</th>
            <th className="text-center text-xs sm:text-sm font-medium text-gray-500 p-2">👍</th>
            <th className="text-center text-xs sm:text-sm font-medium text-gray-500 p-2">💬</th>
            <th className="text-center text-xs sm:text-sm font-medium text-gray-500 p-2">⭐</th>
            <th className="text-center text-xs sm:text-sm font-medium text-gray-500 p-2">📤</th>
            <th className="text-center text-xs sm:text-sm font-medium text-gray-500 p-2">💭</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="hover:bg-rose-50">
              <td className="text-xs sm:text-sm text-gray-700 p-2 font-medium truncate max-w-[100px]">{row.topic}</td>
              {['likes', 'comments', 'collects', 'shares', 'discusses'].map(key => (
                <td key={key} className={`p-1 ${getColor(row[key as keyof typeof row] as number)}`}>
                  <div className="text-center text-white text-xs sm:text-sm font-medium">{formatNumber(row[key as keyof typeof row] as number)}</div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RankingView({ topics }: { topics: Topic[] }) {
  const sortedByLikes = [...topics].sort((a, b) => b.avgLikes - a.avgLikes);
  const sortedByComments = [...topics].sort((a, b) => b.avgComments - a.avgComments);
  const sortedByCollects = [...topics].sort((a, b) => b.avgCollects - a.avgCollects);
  
  const RankItem = ({ items, type }: { items: Topic[]; type: string }) => (
    <div className="bg-gradient-to-br from-rose-50 to-rose-100 rounded-xl p-3 sm:p-4">
      <h4 className="font-semibold text-rose-600 mb-3 text-sm sm:text-base">
        {type === 'likes' ? '❤️' : type === 'comments' ? '💬' : '⭐'} TOP5
      </h4>
      {items.slice(0, 5).map((t, i) => (
        <div key={t.id} className="flex items-center gap-2 mb-2">
          <span className={`w-5 h-5 sm:w-6 sm:h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${i === 0 ? 'bg-yellow-400 text-white' : i === 1 ? 'bg-gray-300 text-white' : i === 2 ? 'bg-amber-600 text-white' : 'bg-rose-200 text-rose-600'}`}>{i + 1}</span>
          <span className="text-xs sm:text-sm text-gray-700 flex-1 truncate">{t.name}</span>
          <span className="text-xs sm:text-sm text-rose-500 font-medium">{type === 'likes' ? t.avgLikes : type === 'comments' ? t.avgComments : t.avgCollects}</span>
        </div>
      ))}
    </div>
  );
  return (
    <div>
      <h3 className="text-base sm:text-lg font-semibold text-gray-700 mb-4 sm:mb-6">互动排行榜</h3>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-6">
        <RankItem items={sortedByLikes} type="likes" />
        <RankItem items={sortedByComments} type="comments" />
        <RankItem items={sortedByCollects} type="collects" />
      </div>
    </div>
  );
}

function BERTopicView({ config, keywords, distribution }: { config: BERTopicConfig; keywords: TopicKeywords[]; umapPoints: UmapPoint[]; similarityMatrix: number[][]; distribution: { name: string; count: number; ratio: number }[] }) {
  const [subView, setSubView] = useState<'scatter' | 'keywords' | 'config'>('scatter');
  const topicColors = ['#f43f5e', '#ec4899', '#8b5cf6', '#06b6d4', '#22c55e'];
  
  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4 sm:mb-6">
        <h3 className="text-base sm:text-lg font-semibold text-gray-700">BERTopic 模型可视化</h3>
        <div className="flex flex-wrap gap-2">
          {[{ id: 'scatter', label: 'UMAP散点' }, { id: 'keywords', label: '关键词' }, { id: 'config', label: '配置' }].map(btn => (
            <button key={btn.id} onClick={() => setSubView(btn.id as typeof subView)} className={`px-2 sm:px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium ${subView === btn.id ? 'bg-rose-500 text-white' : 'bg-gray-100 text-gray-600'}`}>{btn.label}</button>
          ))}
        </div>
      </div>
      {subView === 'scatter' && (
        <div className="space-y-4">
          <div className="bg-slate-50 rounded-xl p-4"><h4 className="text-sm font-semibold text-gray-600 mb-3">📍 UMAP 2D 降维可视化</h4><div className="relative w-full h-48 sm:h-64 border border-slate-200 rounded-lg bg-white p-4 flex items-center justify-center text-gray-400 text-sm">UMAP散点图区域</div></div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {distribution.map((d, i) => (<div key={d.name} className="text-center p-2 bg-white rounded-lg shadow-sm"><div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ backgroundColor: topicColors[i] }} /><div className="text-xs text-gray-600 truncate">{d.name}</div><div className="text-sm font-medium">{formatNumber(d.count)}</div></div>))}
          </div>
        </div>
      )}
      {subView === 'keywords' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {keywords.map(topic => (<div key={topic.topicId} className="bg-white rounded-xl p-4 border border-slate-200"><h4 className="font-semibold text-sm mb-2" style={{ color: topicColors[topic.topicId] }}>{topic.topicName}</h4><div className="flex flex-wrap gap-1">{topic.keywords.slice(0, 5).map(kw => (<span key={kw.word} className="px-2 py-0.5 bg-slate-100 text-slate-600 text-xs rounded">{kw.word}</span>))}</div></div>))}
        </div>
      )}
      {subView === 'config' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-white rounded-xl p-4 border border-slate-200"><h4 className="font-semibold text-rose-600 mb-3 text-sm">模型配置</h4><div className="space-y-2 text-xs sm:text-sm"><div className="flex justify-between"><span className="text-gray-500">嵌入模型</span><span className="text-gray-700">{config.embeddingModel}</span></div><div className="flex justify-between"><span className="text-gray-500">聚类算法</span><span className="text-gray-700">{config.clustering}</span></div><div className="flex justify-between"><span className="text-gray-500">主题数量</span><span className="text-gray-700">{config.nTopics} 个</span></div></div></div>
          <div className="bg-white rounded-xl p-4 border border-slate-200"><h4 className="font-semibold text-pink-600 mb-3 text-sm">UMAP参数</h4><div className="space-y-2 text-xs sm:text-sm"><div className="flex justify-between"><span className="text-gray-500">近邻数</span><span className="text-gray-700">{config.umapParams.nNeighbors}</span></div><div className="flex justify-between"><span className="text-gray-500">降维维度</span><span className="text-gray-700">{config.umapParams.nComponents}</span></div><div className="flex justify-between"><span className="text-gray-500">距离度量</span><span className="text-gray-700">{config.umapParams.metric}</span></div></div></div>
        </div>
      )}
    </div>
  );
}

function TopicDetailPanel({ topic, notes, onNoteClick, onBack }: { topic: Topic; notes: Note[]; onNoteClick: (n: Note) => void; onBack?: () => void }) {
  return (
    <div>
      {onBack && (<button onClick={onBack} className="lg:hidden flex items-center gap-1 text-sm text-rose-500 mb-3"><ArrowLeft className="w-4 h-4" />返回列表</button>)}
      <div className="bg-gradient-to-r from-rose-500 to-pink-500 rounded-2xl p-3 sm:p-4 text-white mb-3 sm:mb-4">
        <h2 className="text-lg sm:text-xl font-bold mb-2">{topic.name}</h2>
        <div className="flex flex-wrap gap-1.5 sm:gap-2 mb-2 sm:mb-3">{topic.keywords.map(kw => (<span key={kw} className="px-2 py-0.5 sm:py-1 bg-white/20 rounded-full text-xs">{kw}</span>))}</div>
        <div className="text-xs sm:text-sm text-white/80">{formatNumber(topic.noteCount)} 篇笔记</div>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:gap-3 mb-3 sm:mb-4">
        {[{ label: '平均点赞', value: topic.avgLikes }, { label: '平均评论', value: topic.avgComments }, { label: '平均收藏', value: topic.avgCollects }, { label: '平均分享', value: topic.avgShares }].map(stat => (<div key={stat.label} className="bg-white rounded-xl p-2 sm:p-3 text-center"><div className="text-lg sm:text-2xl font-bold text-rose-500">{formatNumber(stat.value)}</div><div className="text-xs text-gray-500">{stat.label}</div></div>))}
      </div>
      <div><h3 className="text-sm font-semibold text-gray-700 mb-2 sm:mb-3">热门笔记</h3><div className="space-y-2">{notes.slice(0, 5).map(note => (<div key={note.id} onClick={() => onNoteClick(note)} className="bg-white rounded-xl p-2 sm:p-3 cursor-pointer hover:shadow-md"><div className="text-xs sm:text-sm text-gray-700 mb-1.5 sm:mb-2 line-clamp-2">{note.title}</div><div className="flex items-center gap-2 sm:gap-3 text-xs text-gray-400"><span>❤️ {formatNumber(note.likes)}</span><span>💬 {formatNumber(note.comments)}</span><span>⭐ {formatNumber(note.collects)}</span></div></div>))}</div></div>
    </div>
  );
}

function NoteModal({ note, onClose }: { note: Note; onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-2 sm:p-4" onClick={onClose}>
      <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col" onClick={e => e.stopPropagation()}>
        <div className="bg-gradient-to-r from-rose-500 to-pink-500 px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between shrink-0">
          <div><h2 className="text-white font-semibold text-sm sm:text-lg truncate">{note.title}</h2><p className="text-white/80 text-xs sm:text-sm">{note.topicName} · {note.createdAt}</p></div>
          <button onClick={onClose} className="text-white/80 hover:text-white shrink-0"><X className="w-5 h-5 sm:w-6 sm:h-6" /></button>
        </div>
        <div className="p-4 sm:p-6 overflow-y-auto">
          <div className="mb-4"><span className="px-2 py-1 bg-rose-100 text-rose-600 rounded text-xs">{note.topicName}</span></div>
          <p className="text-gray-700 leading-relaxed mb-4">{note.content}</p>
          <div className="flex items-center gap-4 py-3 border-t border-b border-gray-100">
            {[{ icon: '❤️', value: note.likes }, { icon: '💬', value: note.comments }, { icon: '⭐', value: note.collects }, { icon: '📤', value: note.shares }].map(item => (<div key={item.icon} className="flex items-center gap-1"><span className="text-xl">{item.icon}</span><span className="text-gray-600">{formatNumber(item.value)}</span></div>))}
          </div>
        </div>
      </div>
    </div>
  );
}
