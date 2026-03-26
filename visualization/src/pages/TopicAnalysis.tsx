import React, { useState, useEffect, useMemo } from 'react';
import { Search, BarChart3, PieChart as LucidePieChart, TrendingUp, X, Network, ArrowLeft, Loader2 } from 'lucide-react';
import PieChart from '../components/common/PieChart';
import { computePieSlices } from '../utils/pie';
import {
  loadRawNotes,
  loadTopicCSV,
  aggregateTopics,
  buildInteractionMatrix,
  buildTrendData,
  toNotes,
  Topic,
  TopicRecord,
} from '../data/topicData';
import { EmptyState } from '../components/common/States';
import { formatNumber } from '../utils/responsive';

type ViewMode = 'overview' | 'trend' | 'heatmap' | 'ranking' | 'bertopic' | 'confidence';

export default function TopicAnalysis() {
  const [viewMode, setViewMode] = useState<ViewMode>('overview');
  const [topics, setTopics] = useState<Topic[]>([]);
  const [allRecords, setAllRecords] = useState<TopicRecord[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'noteCount' | 'avgLikes' | 'avgComments' | 'avgCollects'>('noteCount');
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // ─── 数据加载（使用内存模拟数据）───
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setLoadError(null);
      try {
        const [rawNotes, topicRecords] = await Promise.all([
          loadRawNotes(),
          loadTopicCSV(),
        ]);

        // 关联合并：把 raw JSON 的互动数据注入到 topicRecords
        const rawMap = new Map(rawNotes.map((n) => [n.note_id, n]));
        const merged = topicRecords.map((record) => {
          const raw = rawMap.get(record.note_id);
          if (!raw) return record;
          return {
            ...record,
            liked_count: parseInt(String(raw.liked_count || '0'), 10),
            collected_count: parseInt(String(raw.collected_count || '0'), 10),
            comment_count: parseInt(String(raw.comment_count || '0'), 10),
            share_count: parseInt(String(raw.share_count || '0'), 10),
            title: raw.title,
            desc: raw.desc,
            time: raw.time,
            ip_location: raw.ip_location,
            tag_list: raw.tag_list,
            note_url: raw.note_url,
            nickname: raw.nickname,
          } as TopicRecord;
        });

        setAllRecords(merged);
        const aggregated = aggregateTopics(merged);
        setTopics(aggregated);
      } catch (e) {
        console.error('[TopicAnalysis] 数据加载失败:', e);
        setLoadError('数据加载失败，请检查控制台错误信息。');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // ─── 筛选 + 排序 ───
  const filteredTopics = useMemo(() => {
    return topics
      .filter(
        (t) =>
          t.name.includes(searchQuery) ||
          t.keywords.some((k) => k.includes(searchQuery))
      )
      .sort((a, b) => {
        if (sortBy === 'avgLikes') return b.avgLikes - a.avgLikes;
        if (sortBy === 'avgComments') return b.avgComments - a.avgComments;
        if (sortBy === 'avgCollects') return b.avgCollects - a.avgCollects;
        return b.noteCount - a.noteCount;
      });
  }, [topics, searchQuery, sortBy]);

  // ─── 选中主题的笔记 ───
  const topicNotes = useMemo(() => {
    if (!selectedTopic) return [];
    return toNotes(selectedTopic.rawRecords).sort((a, b) => b.likes - a.likes);
  }, [selectedTopic]);

  // ─── 全局统计 ───
  const totalNotes = topics.reduce((s, t) => s + t.noteCount, 0);
  const totalLikes = topics.reduce((s, t) => s + t.avgLikes * t.noteCount, 0);
  const totalComments = topics.reduce((s, t) => s + t.avgComments * t.noteCount, 0);
  const totalCollects = topics.reduce((s, t) => s + t.avgCollects * t.noteCount, 0);
  const totalRecords = allRecords.length;

  // ─── 视图按钮配置 ───
  const viewButtons = [
    { id: 'overview', label: '主题概览', icon: <LucidePieChart className="w-4 h-4" /> },
    { id: 'trend', label: '趋势分析', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'heatmap', label: '互动热力', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'ranking', label: '排行榜', icon: <span>🏆</span> },
    { id: 'bertopic', label: 'BERTopic', icon: <Network className="w-4 h-4" /> },
    { id: 'confidence', label: '置信度', icon: <span>🎯</span> },
  ];

  // ─── 加载状态 ───
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-rose-50 via-pink-50 to-rose-100 flex flex-col items-center justify-center gap-4">
        <Loader2 className="w-12 h-12 text-rose-400 animate-spin" />
        <div className="text-center">
          <p className="text-rose-500 font-semibold">正在加载数据...</p>
          <p className="text-gray-400 text-sm mt-1">正在读取主题分类与笔记数据</p>
        </div>
      </div>
    );
  }

  // ─── 错误状态 ───
  if (loadError) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-rose-50 via-pink-50 to-rose-100 flex items-center justify-center p-6">
        <div className="bg-white rounded-2xl shadow-xl p-8 max-w-lg w-full text-center">
          <div className="text-5xl mb-4">⚠️</div>
          <h2 className="text-xl font-bold text-rose-500 mb-3">数据加载失败</h2>
          <p className="text-sm text-gray-600 bg-gray-50 rounded-xl p-4">
            {loadError}
          </p>
          <p className="mt-4 text-sm text-gray-400">
            请检查控制台错误信息，或重新启动开发服务器（pnpm run dev）
          </p>
        </div>
      </div>
    );
  }

  const interactionMatrix = buildInteractionMatrix(topics);
  const trendData = buildTrendData(allRecords);

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-50 via-pink-50 to-rose-100">
      {/* ─── 顶部导航 ─── */}
      <header className="bg-white/80 backdrop-blur-md border-b border-rose-100 px-3 sm:px-4 lg:px-6 py-3 sm:py-4 sticky top-14 sm:top-16 z-40">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 lg:gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-rose-400 to-pink-500 rounded-xl flex items-center justify-center shadow-lg shrink-0">
              <span className="text-white font-bold text-sm sm:text-lg">📕</span>
            </div>
            <div>
              <h1 className="text-lg sm:text-xl font-bold bg-gradient-to-r from-rose-500 to-pink-600 bg-clip-text text-transparent">
                小红书AI主题研究分析
              </h1>
              <p className="text-xs text-gray-400 hidden sm:block">
                {totalRecords > 0
                  ? `基于 BERTopic · ${formatNumber(totalRecords)} 条笔记 · ${topics.length} 个宏观主题`
                  : '基于 BERTopic 模型的主题分析'}
              </p>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3">
            <button
              onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
              className="lg:hidden flex items-center justify-center gap-2 px-3 py-2 bg-rose-100 text-rose-600 rounded-lg text-sm font-medium"
            >
              <BarChart3 className="w-4 h-4" />
              {selectedTopic ? selectedTopic.name : '选择主题'}
            </button>

            <div className="relative flex-1 sm:flex-none">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="搜索主题或关键词..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full sm:w-48 lg:w-64 pl-9 pr-3 py-2 bg-gray-50 border border-gray-200 rounded-lg sm:rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-rose-300 transition-all"
              />
            </div>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-3 sm:px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg sm:rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-rose-300"
            >
              <option value="noteCount">按笔记数量</option>
              <option value="avgLikes">按平均点赞</option>
              <option value="avgComments">按平均评论</option>
              <option value="avgCollects">按平均收藏</option>
            </select>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 sm:gap-6 mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-rose-50">
          <div className="flex items-center gap-2">
            <span className="text-xl sm:text-2xl font-bold text-rose-500">{topics.length}</span>
            <span className="text-xs sm:text-sm text-gray-500">主题</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl sm:text-2xl font-bold text-pink-500">{formatNumber(totalRecords)}</span>
            <span className="text-xs sm:text-sm text-gray-500">笔记</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl sm:text-2xl font-bold text-rose-400">{formatNumber(totalLikes)}</span>
            <span className="text-xs sm:text-sm text-gray-500">总点赞</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl sm:text-2xl font-bold text-pink-400">{formatNumber(totalComments)}</span>
            <span className="text-xs sm:text-sm text-gray-500">总评论</span>
          </div>
          <div className="hidden sm:flex items-center gap-2">
            <span className="text-xl font-bold text-rose-300">{formatNumber(totalCollects)}</span>
            <span className="text-sm text-gray-500">总收藏</span>
          </div>
        </div>
      </header>

      {/* ─── 主内容 ─── */}
      <main className="flex flex-col lg:flex-row min-h-[calc(100vh-220px)]">
        {/* 左侧主题列表 */}
        <aside
          className={`
            lg:w-64 xl:w-72 bg-white/60 backdrop-blur-sm border-r border-rose-100
            overflow-y-auto p-3 sm:p-4
            ${isMobileSidebarOpen ? 'block' : 'hidden'}
            lg:block
            fixed lg:sticky top-36 lg:top-auto left-0 right-0 z-30 lg:z-auto
            max-h-[50vh] lg:max-h-none
            shadow-xl lg:shadow-none
          `}
        >
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
            <EmptyState type="search" title="未找到匹配主题" description="请尝试其他搜索关键词" />
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

        {/* 中间主可视化 */}
        <section className="flex-1 p-3 sm:p-4 lg:p-6 overflow-y-auto">
          <div className="flex flex-wrap items-center gap-2 mb-4 sm:mb-6">
            {viewButtons.map((btn) => (
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

          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-4 sm:p-6 min-h-[400px]">
            {filteredTopics.length === 0 ? (
              <EmptyState type="search" />
            ) : (
              <>
                {viewMode === 'overview' && <OverviewView topics={filteredTopics} />}
                {viewMode === 'trend' && <TrendView data={trendData} />}
                {viewMode === 'heatmap' && <HeatmapView data={interactionMatrix} />}
                {viewMode === 'ranking' && <RankingView topics={filteredTopics} />}
                {viewMode === 'bertopic' && <BERTopicView topics={filteredTopics} allRecords={allRecords} />}
                {viewMode === 'confidence' && <ConfidenceView topics={filteredTopics} allRecords={allRecords} />}
              </>
            )}
          </div>
        </section>

        {/* 右侧详情面板 */}
        <aside
          className={`
            lg:w-80 xl:w-96 bg-white/60 backdrop-blur-sm border-l border-rose-100
            overflow-y-auto p-3 sm:p-4
            ${selectedTopic ? 'block' : 'hidden'}
            lg:block
          `}
        >
          {selectedTopic ? (
            <TopicDetailPanel
              topic={selectedTopic}
              notes={topicNotes}
              onNoteClick={(n) => setSelectedNoteId(n.id)}
              onBack={() => setSelectedTopic(null)}
              allRecords={allRecords}
            />
          ) : (
            <div className="hidden lg:flex flex-col items-center justify-center h-full text-gray-400 py-12">
              <div className="text-5xl mb-4">👈</div>
              <p className="text-center">选择一个主题查看详情</p>
            </div>
          )}
        </aside>
      </main>

      {/* 笔记详情弹窗 */}
      {selectedNoteId && (
        <NoteModal noteId={selectedNoteId} records={allRecords} onClose={() => setSelectedNoteId(null)} />
      )}
    </div>
  );
}

// ================================================================
// 子组件
// ================================================================

const OVERVIEW_COLORS = [
  '#f43f5e', '#ec4899', '#d946ef', '#a855f7',
  '#8b5cf6', '#6366f1', '#3b82f6', '#06b6d4',
  '#14b8a6', '#22c55e',
];

function OverviewView({ topics }: { topics: Topic[] }) {
  const total = topics.reduce((s, t) => s + t.noteCount, 0);

  const pieData = useMemo(() => {
    return topics.map((t, i) => ({
      id: t.id,
      name: t.name,
      value: t.noteCount,
      color: OVERVIEW_COLORS[i % OVERVIEW_COLORS.length],
    }));
  }, [topics]);

  const pieSlices = useMemo(() => computePieSlices(pieData), [pieData]);

  return (
    <div>
      <h3 className="text-base sm:text-lg font-semibold text-gray-700 mb-4 sm:mb-6">主题分布与关键词</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
        {/* 环形饼图 */}
        <div className="flex flex-col items-center">
          <PieChart slices={pieSlices} />
          <div className="text-center mt-2">
            <span className="text-3xl font-bold text-gray-700">{topics.length}</span>
            <span className="text-sm text-gray-500 block">个主题</span>
          </div>
          <div className="flex flex-wrap justify-center gap-2 mt-3">
            {topics.slice(0, 8).map((t, i) => (
              <div key={t.id} className="flex items-center gap-1.5 text-xs">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: OVERVIEW_COLORS[i % OVERVIEW_COLORS.length] }} />
                <span className="text-gray-600">{t.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* 主题列表 */}
        <div className="space-y-2 max-h-[420px] overflow-y-auto">
          {topics.map((topic, i) => (
            <div
              key={topic.id}
              className="flex items-center gap-3 p-2 sm:p-3 rounded-xl bg-gradient-to-r from-gray-50 to-white hover:shadow-md transition-all"
            >
              <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: OVERVIEW_COLORS[i % OVERVIEW_COLORS.length] }} />
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm text-gray-700 truncate">{topic.name}</div>
                <div className="text-xs text-gray-400">
                  {formatNumber(topic.noteCount)} 篇 · 置信度 {formatNumber(topic.avgConfidence * 100)}%
                </div>
              </div>
              <div className="flex flex-wrap gap-1 max-w-[120px] shrink-0">
                {topic.keywords.slice(0, 3).map((kw) => (
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

function TrendView({ data }: { data: ReturnType<typeof buildTrendData> }) {
  if (!data || data.length === 0) return <EmptyState type="data" title="暂无趋势数据" />;
  const maxNotes = Math.max(...data.map((d) => d.notes), 1);

  return (
    <div>
      <h3 className="text-base sm:text-lg font-semibold text-gray-700 mb-4 sm:mb-6">
        发布趋势分析（{data[0]?.date || ''} ~ {data[data.length - 1]?.date || ''}）
      </h3>
      <div className="flex items-end justify-between h-48 sm:h-64 gap-2 sm:gap-4">
        {data.map((day, i) => (
          <div key={i} className="flex-1 flex flex-col items-center">
            <div
              className="w-full bg-gradient-to-t from-rose-400 to-pink-400 rounded-t-lg transition-all hover:opacity-80 cursor-pointer"
              style={{ height: `${Math.max((day.notes / maxNotes) * 150, 8)}px` }}
              title={`${day.notes} 篇笔记`}
            />
            <span className="text-xs text-gray-500 mt-2">{day.date}</span>
            <span className="text-xs text-rose-400 font-medium">{day.notes}篇</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HeatmapView({ data }: { data: ReturnType<typeof buildInteractionMatrix> }) {
  if (!data || data.length === 0) return <EmptyState type="data" title="暂无互动数据" />;
  const maxVal = Math.max(...data.map((d) => Math.max(d.likes, d.comments, d.collects, d.shares, d.discusses)), 1);

  const getColor = (val: number) => {
    const intensity = val / maxVal;
    if (intensity > 0.8) return 'bg-rose-500 text-white';
    if (intensity > 0.6) return 'bg-rose-400 text-white';
    if (intensity > 0.4) return 'bg-pink-400 text-white';
    if (intensity > 0.2) return 'bg-pink-300 text-rose-700';
    return 'bg-pink-100 text-rose-600';
  };

  return (
    <div className="overflow-x-auto">
      <h3 className="text-base sm:text-lg font-semibold text-gray-700 mb-4 sm:mb-6">主题互动热力图</h3>
      <table className="w-full min-w-[500px]">
        <thead>
          <tr>
            <th className="text-left text-xs sm:text-sm font-medium text-gray-500 p-2">主题</th>
            <th className="text-center text-xs sm:text-sm font-medium text-gray-500 p-2">👍 均赞</th>
            <th className="text-center text-xs sm:text-sm font-medium text-gray-500 p-2">💬 均评</th>
            <th className="text-center text-xs sm:text-sm font-medium text-gray-500 p-2">⭐ 均藏</th>
            <th className="text-center text-xs sm:text-sm font-medium text-gray-500 p-2">📤 均分</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="hover:bg-rose-50 transition-colors">
              <td className="text-xs sm:text-sm text-gray-700 p-2 font-medium truncate max-w-[120px]" title={row.topic}>
                {row.topic}
              </td>
              {(['likes', 'comments', 'collects', 'shares'] as const).map((key) => (
                <td key={key} className={`p-1 ${getColor(row[key])}`}>
                  <div className="text-center text-xs sm:text-sm font-medium">
                    {formatNumber(row[key])}
                  </div>
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

  const RankItem = ({
    items,
    type,
    getValue,
  }: {
    items: Topic[];
    type: string;
    getValue: (t: Topic) => number;
  }) => (
    <div className="bg-gradient-to-br from-rose-50 to-rose-100 rounded-xl p-3 sm:p-4">
      <h4 className="font-semibold text-rose-600 mb-3 text-sm sm:text-base">{type} TOP5</h4>
      {items.slice(0, 5).map((t, i) => (
        <div key={t.id} className="flex items-center gap-2 mb-2">
          <span
            className={`w-5 h-5 sm:w-6 sm:h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
              i === 0 ? 'bg-yellow-400 text-white'
                : i === 1 ? 'bg-gray-300 text-white'
                : i === 2 ? 'bg-amber-600 text-white'
                : 'bg-rose-200 text-rose-600'
            }`}
          >
            {i + 1}
          </span>
          <span className="text-xs sm:text-sm text-gray-700 flex-1 truncate" title={t.name}>{t.name}</span>
          <span className="text-xs sm:text-sm text-rose-500 font-medium">{formatNumber(getValue(t))}</span>
        </div>
      ))}
    </div>
  );

  return (
    <div>
      <h3 className="text-base sm:text-lg font-semibold text-gray-700 mb-4 sm:mb-6">互动排行榜</h3>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-6">
        <RankItem items={sortedByLikes} type="❤️ 平均点赞" getValue={(t) => t.avgLikes} />
        <RankItem items={sortedByComments} type="💬 平均评论" getValue={(t) => t.avgComments} />
        <RankItem items={sortedByCollects} type="⭐ 平均收藏" getValue={(t) => t.avgCollects} />
      </div>
    </div>
  );
}

function BERTopicView({ topics, allRecords }: { topics: Topic[]; allRecords: TopicRecord[] }) {
  const [subView, setSubView] = useState<'keywords' | 'micro' | 'data'>('keywords');
  const CHART_COLORS = ['#f43f5e', '#ec4899', '#8b5cf6', '#06b6d4', '#22c55e', '#14b8a6', '#f59e0b', '#ef4444', '#6366f1', '#a855f7'];
  const totalRecords = allRecords.length;
  const totalMicroTopics = topics.reduce((s, t) => s + t.microTopics.length, 0);
  const avgConf = totalRecords > 0
    ? parseFloat((allRecords.reduce((s, r) => s + r.confidence, 0) / totalRecords).toFixed(3))
    : 0;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4 sm:mb-6">
        <h3 className="text-base sm:text-lg font-semibold text-gray-700">BERTopic 模型可视化</h3>
        <div className="flex flex-wrap gap-2">
          {[
            { id: 'keywords', label: '关键词视图' },
            { id: 'micro', label: '微观主题' },
            { id: 'data', label: '数据概览' },
          ].map((btn) => (
            <button
              key={btn.id}
              onClick={() => setSubView(btn.id as typeof subView)}
              className={`px-2 sm:px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium ${
                subView === btn.id ? 'bg-rose-500 text-white' : 'bg-gray-100 text-gray-600'
              }`}
            >
              {btn.label}
            </button>
          ))}
        </div>
      </div>

      {subView === 'keywords' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {topics.map((topic, i) => (
            <div key={topic.id} className="bg-white rounded-xl p-4 border border-gray-100 hover:shadow-md transition-all">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: OVERVIEW_COLORS[i % OVERVIEW_COLORS.length] }} />
                <h4 className="font-semibold text-sm text-gray-700 truncate flex-1">{topic.name}</h4>
                <span className="text-xs text-gray-400 shrink-0">{topic.noteCount}篇</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {topic.keywords.slice(0, 6).map((kw) => (
                  <span
                    key={kw}
                    className="px-2 py-0.5 text-xs rounded-full"
                    style={{ backgroundColor: `${OVERVIEW_COLORS[i % OVERVIEW_COLORS.length]}18`, color: OVERVIEW_COLORS[i % OVERVIEW_COLORS.length] }}
                  >
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {subView === 'micro' && (
        <div className="space-y-4">
          {topics.map((topic, i) => (
            <div key={topic.id} className="bg-white rounded-xl p-4 border border-gray-100">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: OVERVIEW_COLORS[i % OVERVIEW_COLORS.length] }} />
                <h4 className="font-semibold text-sm text-gray-700">{topic.name}</h4>
                <span className="text-xs text-gray-400 ml-auto">{topic.microTopics.length} 个微观主题</span>
              </div>
              <div className="space-y-2">
                {topic.microTopics.slice(0, 4).map((mt) => (
                  <div key={mt.id} className="flex items-start gap-2">
                    <span className="text-xs text-gray-400 w-5 shrink-0 mt-0.5">#{mt.id}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex flex-wrap gap-1">
                        {mt.keywords.slice(0, 4).map((kw) => (
                          <span key={kw} className="px-1.5 py-0.5 bg-gray-50 text-gray-500 text-xs rounded">{kw}</span>
                        ))}
                      </div>
                    </div>
                    <span className="text-xs text-gray-400 shrink-0 mt-0.5">{mt.noteCount}篇</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {subView === 'data' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-white rounded-xl p-4 border border-gray-200">
            <h4 className="font-semibold text-rose-600 mb-3 text-sm">数据规模</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">宏观主题数</span>
                <span className="text-gray-700 font-medium">{topics.length} 个</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">笔记总数</span>
                <span className="text-gray-700 font-medium">{formatNumber(topics.reduce((s, t) => s + t.noteCount, 0))} 篇</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">微观主题数</span>
                <span className="text-gray-700 font-medium">{totalMicroTopics} 个</span>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 border border-gray-200">
            <h4 className="font-semibold text-pink-600 mb-3 text-sm">模型质量</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">平均置信度</span>
                <span className="text-gray-700 font-medium">{(avgConf * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">高置信度(&gt;0.7)笔记</span>
                <span className="text-gray-700 font-medium">
                  {formatNumber(topics.reduce((s, t) => s + t.rawRecords.filter(r => r.confidence > 0.7).length, 0))} 篇
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── 置信度分布视图（新增）───
function ConfidenceView({ topics, allRecords }: { topics: Topic[]; allRecords: TopicRecord[] }) {
  if (!allRecords || allRecords.length === 0) return <EmptyState type="data" title="暂无置信度数据" />;

  const buckets = [
    { label: '0.0–0.2', min: 0, max: 0.2, color: '#ef4444' },
    { label: '0.2–0.4', min: 0.2, max: 0.4, color: '#f97316' },
    { label: '0.4–0.6', min: 0.4, max: 0.6, color: '#eab308' },
    { label: '0.6–0.8', min: 0.6, max: 0.8, color: '#22c55e' },
    { label: '0.8–1.0', min: 0.8, max: 1.0, color: '#06b6d4' },
  ];

  const bucketData = buckets.map((b) => {
    const count = allRecords.filter((r) => r.confidence >= b.min && r.confidence < b.max).length;
    return { ...b, count };
  });

  const maxCount = Math.max(...bucketData.map((b) => b.count), 1);
  const avgConf = parseFloat((allRecords.reduce((s, r) => s + r.confidence, 0) / allRecords.length).toFixed(3));
  const highConfCount = allRecords.filter((r) => r.confidence > 0.7).length;
  const highConfRate = (highConfCount / allRecords.length * 100).toFixed(1);

  return (
    <div>
      <h3 className="text-base sm:text-lg font-semibold text-gray-700 mb-4 sm:mb-6">归类置信度分布</h3>

      {/* 概览卡片 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 text-center border border-blue-100">
          <div className="text-2xl sm:text-3xl font-bold text-blue-600">{(avgConf * 100).toFixed(1)}%</div>
          <div className="text-xs sm:text-sm text-blue-500 mt-1">平均置信度</div>
        </div>
        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-4 text-center border border-green-100">
          <div className="text-2xl sm:text-3xl font-bold text-green-600">{highConfRate}%</div>
          <div className="text-xs sm:text-sm text-green-500 mt-1">高置信度(&gt;0.7)</div>
        </div>
        <div className="bg-gradient-to-br from-violet-50 to-violet-100 rounded-xl p-4 text-center border border-violet-100">
          <div className="text-2xl sm:text-3xl font-bold text-violet-600">{formatNumber(allRecords.length)}</div>
          <div className="text-xs sm:text-sm text-violet-500 mt-1">总笔记数</div>
        </div>
      </div>

      {/* 置信度柱状图 */}
      <div className="bg-white rounded-xl p-4 border border-gray-100 mb-4">
        <h4 className="text-sm font-semibold text-gray-600 mb-4">置信度分桶分布</h4>
        <div className="space-y-3">
          {bucketData.map((b) => (
            <div key={b.label} className="flex items-center gap-3">
              <span className="text-xs text-gray-500 w-12 shrink-0">{b.label}</span>
              <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all flex items-center justify-end pr-2"
                  style={{ width: `${(b.count / maxCount) * 100}%`, backgroundColor: b.color }}
                >
                  {b.count > 0 && (
                    <span className="text-white text-xs font-medium">{b.count}</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 各主题置信度 */}
      <div className="bg-white rounded-xl p-4 border border-gray-100">
        <h4 className="text-sm font-semibold text-gray-600 mb-3">各主题平均置信度</h4>
        <div className="space-y-3">
          {topics.map((topic, i) => {
            const confPct = (topic.avgConfidence * 100).toFixed(1);
            const confColor = topic.avgConfidence > 0.7 ? '#22c55e' : topic.avgConfidence > 0.5 ? '#eab308' : '#ef4444';
            return (
              <div key={topic.id} className="flex items-center gap-3">
                <span className="text-xs text-gray-500 w-5 shrink-0">{i + 1}</span>
                <span className="text-xs text-gray-700 flex-1 truncate max-w-[120px]">{topic.name}</span>
                <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${confPct}%`, backgroundColor: confColor }}
                  />
                </div>
                <span className="text-xs font-medium shrink-0" style={{ color: confColor }}>{confPct}%</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── 主题详情面板 ───
function TopicDetailPanel({
  topic,
  notes,
  onNoteClick,
  onBack,
  allRecords,
}: {
  topic: Topic;
  notes: ReturnType<typeof toNotes>;
  onNoteClick: (n: ReturnType<typeof toNotes>[number]) => void;
  onBack?: () => void;
  allRecords: TopicRecord[];
}) {
  const topicRecord = allRecords.find((r) => r.macro_topic === topic.name) || allRecords[0];

  return (
    <div>
      {onBack && (
        <button
          onClick={onBack}
          className="lg:hidden flex items-center gap-1 text-sm text-rose-500 mb-3"
        >
          <ArrowLeft className="w-4 h-4" />
          返回列表
        </button>
      )}

      {/* 主题头 */}
      <div className="bg-gradient-to-r from-rose-500 to-pink-500 rounded-2xl p-3 sm:p-4 text-white mb-3 sm:mb-4">
        <h2 className="text-lg sm:text-xl font-bold mb-2">{topic.name}</h2>
        <div className="flex flex-wrap gap-1.5 sm:gap-2 mb-2 sm:mb-3">
          {topic.keywords.slice(0, 6).map((kw) => (
            <span key={kw} className="px-2 py-0.5 sm:py-1 bg-white/20 rounded-full text-xs">{kw}</span>
          ))}
        </div>
        <div className="text-xs sm:text-sm text-white/80">
          {formatNumber(topic.noteCount)} 篇笔记 · {topic.microTopics.length} 个微观主题
        </div>
      </div>

      {/* 互动数据 */}
      <div className="grid grid-cols-2 gap-2 sm:gap-3 mb-3 sm:mb-4">
        {[
          { label: '平均点赞', value: topic.avgLikes },
          { label: '平均评论', value: topic.avgComments },
          { label: '平均收藏', value: topic.avgCollects },
          { label: '平均分享', value: topic.avgShares },
        ].map((stat) => (
          <div key={stat.label} className="bg-white rounded-xl p-2 sm:p-3 text-center">
            <div className="text-lg sm:text-2xl font-bold text-rose-500">{formatNumber(stat.value)}</div>
            <div className="text-xs text-gray-500">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* 置信度 */}
      <div className="bg-white rounded-xl p-3 sm:p-4 mb-3 sm:mb-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">归类置信度</h3>
        <div className="flex items-center gap-3">
          <span className="text-2xl font-bold text-rose-500">{(topic.avgConfidence * 100).toFixed(1)}%</span>
          <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-rose-400 to-pink-400 rounded-full"
              style={{ width: `${topic.avgConfidence * 100}%` }}
            />
          </div>
        </div>
        <div className="text-xs text-gray-400 mt-1">
          高置信度笔记占 {(topic.confidenceRate * 100).toFixed(1)}%
        </div>
      </div>

      {/* 7日趋势 */}
      <div className="bg-white rounded-xl p-3 sm:p-4 mb-3 sm:mb-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">7日发布趋势</h3>
        <div className="flex items-end gap-1 h-16">
          {topic.trend.map((val, i) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-1">
              <div
                className="w-full bg-gradient-to-t from-rose-400 to-pink-400 rounded-t transition-all"
                style={{ height: `${Math.max((val / Math.max(...topic.trend, 1)) * 48, 4)}px` }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* 热门笔记 */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2 sm:mb-3">热门笔记</h3>
        <div className="space-y-2">
          {notes.slice(0, 8).map((note) => (
            <div
              key={note.id}
              onClick={() => onNoteClick(note)}
              className="bg-white rounded-xl p-2 sm:p-3 cursor-pointer hover:shadow-md transition-all"
            >
              <div className="text-xs sm:text-sm text-gray-700 mb-1.5 sm:mb-2 line-clamp-2">{note.title}</div>
              <div className="flex items-center gap-2 sm:gap-3 text-xs text-gray-400">
                <span>❤️ {formatNumber(note.likes)}</span>
                <span>💬 {formatNumber(note.comments)}</span>
                <span>⭐ {formatNumber(note.collects)}</span>
                {note.confidence > 0 && (
                  <span className="ml-auto text-cyan-500">{(note.confidence * 100).toFixed(0)}%</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── 笔记详情弹窗 ───
function NoteModal({
  noteId,
  records,
  onClose,
}: {
  noteId: string;
  records: TopicRecord[];
  onClose: () => void;
}) {
  const record = records.find((r) => r.note_id === noteId);

  if (!record) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-2 sm:p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 头部 */}
        <div className="bg-gradient-to-r from-rose-500 to-pink-500 px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between shrink-0">
          <div className="min-w-0 mr-4">
            <h2 className="text-white font-semibold text-sm sm:text-lg truncate">
              {record.title || '无标题'}
            </h2>
            <p className="text-white/80 text-xs sm:text-sm truncate">
              {record.macro_topic}
              {record.nickname ? ` · @${record.nickname}` : ''}
              {record.ip_location ? ` · ${record.ip_location}` : ''}
            </p>
          </div>
          <button onClick={onClose} className="text-white/80 hover:text-white shrink-0">
            <X className="w-5 h-5 sm:w-6 sm:h-6" />
          </button>
        </div>

        {/* 内容 */}
        <div className="p-4 sm:p-6 overflow-y-auto">
          {/* 标签 */}
          <div className="flex flex-wrap gap-2 mb-4">
            <span className="px-2 py-1 bg-rose-100 text-rose-600 rounded text-xs font-medium">
              {record.macro_topic}
            </span>
            <span className="px-2 py-1 bg-violet-100 text-violet-600 rounded text-xs">
              微观主题 #{record.micro_topic_id}
            </span>
            {record.tag_list && record.tag_list.split(',').slice(0, 4).map((tag) => (
              <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-500 rounded text-xs">{tag.trim()}</span>
            ))}
          </div>

          {/* 互动数据 */}
          <div className="grid grid-cols-4 gap-3 mb-4">
            {[
              { icon: '❤️', label: '点赞', value: record.liked_count },
              { icon: '💬', label: '评论', value: record.comment_count },
              { icon: '⭐', label: '收藏', value: record.collected_count },
              { icon: '📤', label: '分享', value: record.share_count },
            ].map((item) => (
              <div key={item.label} className="bg-gray-50 rounded-xl p-2 sm:p-3 text-center">
                <div className="text-lg sm:text-xl">{item.icon}</div>
                <div className="text-sm sm:text-base font-bold text-gray-700">{formatNumber(item.value || 0)}</div>
                <div className="text-xs text-gray-400">{item.label}</div>
              </div>
            ))}
          </div>

          {/* 置信度 */}
          <div className="bg-cyan-50 rounded-xl p-3 mb-4 flex items-center gap-3">
            <span className="text-sm text-cyan-700">归类置信度</span>
            <div className="flex-1 h-2.5 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full"
                style={{ width: `${record.confidence * 100}%` }}
              />
            </div>
            <span className="text-sm font-bold text-cyan-700">
              {(record.confidence * 100).toFixed(1)}%
            </span>
          </div>

          {/* 正文摘要 */}
          {record.content && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">正文摘要</h3>
              <p className="text-sm text-gray-600 leading-relaxed bg-gray-50 rounded-xl p-3 max-h-40 overflow-y-auto">
                {record.content.slice(0, 300)}
                {record.content.length > 300 && '...'}
              </p>
            </div>
          )}

          {/* 分词文本 */}
          {record.segmented_text && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">分词文本</h3>
              <p className="text-xs text-gray-500 leading-relaxed bg-gray-50 rounded-xl p-3 max-h-32 overflow-y-auto">
                {record.segmented_text.slice(0, 200)}
                {record.segmented_text.length > 200 && '...'}
              </p>
            </div>
          )}

          {/* 关键词 */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">微观主题关键词</h3>
            <div className="flex flex-wrap gap-1.5">
              {record.micro_topic_full_keywords
                ? record.micro_topic_full_keywords.split(',').map((kw) => (
                    <span key={kw} className="px-2 py-1 bg-violet-50 text-violet-600 text-xs rounded-full">
                      {kw.trim()}
                    </span>
                  ))
                : record.micro_topic_keywords.split(',').map((kw) => (
                    <span key={kw} className="px-2 py-1 bg-violet-50 text-violet-600 text-xs rounded-full">
                      {kw.trim()}
                    </span>
                  ))}
            </div>
          </div>

          {/* 链接 */}
          {record.note_url && (
            <div className="pt-3 border-t border-gray-100">
              <a
                href={record.note_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-rose-500 hover:text-rose-600 underline"
              >
                查看原笔记 →
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
