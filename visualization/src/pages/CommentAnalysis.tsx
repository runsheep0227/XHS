import { useState, useEffect, useMemo, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  MessageCircle,
  Heart,
  ThumbsDown,
  BarChart3,
  PieChart,
  Search,
  X,
  MessageSquare,
  Brain,
  Settings,
  BookOpen,
  Activity,
  type LucideIcon,
} from 'lucide-react';
import {
  ResponsiveContainer,
  PieChart as RPie,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  LineChart,
  Line,
  ReferenceLine,
} from 'recharts';
import type { CommentTopic, CommentAnalysis, RoBERTaConfig, EvaluationMetrics } from '../data/commentData';
import { loadLiveCommentBundle, type LiveCommentBundle, type EvalVizPayload } from '../data/commentLiveData';
import evalVizDemoPayload from '../data/evalVizDemoPayload.json';
import { MacroSentimentEvalCharts, ConfidenceHistogramChart, RocCurvesEcharts } from '../components/MacroSentimentEvalCharts';
import { CommentGeoInsightSections } from '../components/CommentGeoInsightSections';
import { macroTopicColorWithAlpha, macroTopicDisplayColor } from '../theme/macroTopicColors';
import topicCorrelationSnapshot from '../data/topicCorrelationSnapshot.json';

const TOPIC_CORRELATION_LS_KEY = 'viz_topic_correlation_snapshot_v1';
/** Recharts 类组件与 @types/react 的 JSX 推断不兼容，与 InteractiveCharts.tsx 一致用 any 断言 */
const ResponsiveContainerAny = ResponsiveContainer as any;
const PieChartRecharts = RPie as any;
const PieAny = Pie as any;
const CellAny = Cell as any;
const TooltipAny = Tooltip as any;
const LegendAny = Legend as any;
const LineChartAny = LineChart as any;
const LineAny = Line as any;
const BarChartAny = BarChart as any;
const BarAny = Bar as any;
const XAxisAny = XAxis as any;
const YAxisAny = YAxis as any;
const CartesianGridAny = CartesianGrid as any;
const ReferenceLineAny = ReferenceLine as any;

/** 基于 MacBERT 的评论反馈（与笔记主题通过 note_id 对齐） */
type FeedbackView = 'sentiment' | 'notes';
type CommentModule = 'feedback' | 'model';
type ModelSubView = 'overview' | 'metrics' | 'training';

const SENT_COLORS = { pos: '#10b981', neu: '#94a3b8', neg: '#f43f5e' };

const CHART_TOOLTIP_STYLE = {
  borderRadius: 12,
  border: '1px solid rgb(226 232 240)',
  boxShadow: '0 10px 40px -12px rgb(0 0 0 / 0.18)',
};

/** 无原始 note_url 时的 explore 兜底（不含 xsec_token，可能被站方限流） */
function xiaohongshuExploreFallback(noteId: string) {
  const id = noteId.trim();
  if (!id) return '';
  return `https://www.xiaohongshu.com/explore/${encodeURIComponent(id)}`;
}

/** 优先 content/rawdata 合并字段 note_url，否则按 note_id 拼 explore */
function resolveNotePageHref(noteUrl: string | undefined, noteId: string | undefined): string | null {
  const u = noteUrl?.trim();
  if (u) return u;
  const id = noteId?.trim();
  if (!id) return null;
  const fb = xiaohongshuExploreFallback(id);
  return fb || null;
}

const FEEDBACK_TABS: { id: FeedbackView; label: string; short: string; icon: LucideIcon }[] = [
  { id: 'sentiment', label: '情感分布', short: '情感', icon: Heart },
  { id: 'notes', label: '主题关联', short: '关联', icon: MessageCircle },
];

const MODEL_TABS: { id: ModelSubView; label: string; icon: LucideIcon }[] = [
  { id: 'overview', label: '概览', icon: PieChart },
  { id: 'metrics', label: '评估', icon: BarChart3 },
  { id: 'training', label: '训练', icon: Activity },
];

const COMMENT_MAIN_MODULES = [
  {
    id: 'feedback' as const,
    label: '基于 MacBERT 的评论反馈',
    hint: '极性统计、情感结构与 note_id 主题联动',
    icon: BookOpen,
  },
  {
    id: 'model' as const,
    label: '模型效果',
    hint: 'MacBERT-large 指标、训练与误判样例',
    icon: Brain,
  },
] as const;

export default function CommentAnalysisPage() {
  const [commentModule, setCommentModule] = useState<CommentModule>('feedback');
  const [feedbackView, setFeedbackView] = useState<FeedbackView>('sentiment');
  const [modelSubView, setModelSubView] = useState<ModelSubView>('overview');
  const [selectedTopic, setSelectedTopic] = useState<CommentTopic | null>(null);
  const [selectedNoteAnalysis, setSelectedNoteAnalysis] = useState<CommentAnalysis | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState<'all' | 'positive' | 'neutral' | 'negative'>('all');
  const [bundle, setBundle] = useState<LiveCommentBundle | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    loadLiveCommentBundle()
      .then(b => {
        if (!cancelled) {
          setBundle(b);
          setLoadErr(null);
        }
      })
      .catch(e => {
        if (!cancelled)
          setLoadErr(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const metrics = bundle?.metrics ?? {
    totalComments: 0,
    avgSentimentScore: 0,
    positiveRatio: 0,
    negativeRatio: 0,
    neutralRatio: 0,
    avgCommentLength: 0,
    replyRate: NaN,
  };

  const topics = bundle?.topics ?? [];
  const noteDetails = bundle?.noteDetails ?? [];

  const filteredAnalysis = useMemo(
    () =>
      noteDetails.filter(a => {
        const matchSearch =
          !searchQuery ||
          a.noteTitle.includes(searchQuery) ||
          a.topicName.includes(searchQuery) ||
          a.keywords.some(k => k.includes(searchQuery)) ||
          a.noteId.includes(searchQuery) ||
          (a.contentMicroKeywords?.includes(searchQuery) ?? false) ||
          (a.contentTopicKeywords?.includes(searchQuery) ?? false);
        const matchSentiment =
          sentimentFilter === 'all' || a.sentiment === sentimentFilter;
        return matchSearch && matchSentiment;
      }),
    [noteDetails, searchQuery, sentimentFilter]
  );

  const switchCommentModule = (id: CommentModule) => {
    if (id === commentModule) return;
    setCommentModule(id);
    if (id === 'feedback') {
      setFeedbackView('sentiment');
    } else {
      setModelSubView('overview');
    }
  };

  const filteredTopics = useMemo(() => {
    if (selectedTopic) return topics;
    if (!searchQuery) return topics;
    return topics.filter(
      t =>
        t.name.includes(searchQuery) ||
        t.keywords.some(k => k.includes(searchQuery)) ||
        (t.noteId?.includes(searchQuery) ?? false) ||
        (t.contentMacroTopic?.includes(searchQuery) ?? false)
    );
  }, [topics, selectedTopic, searchQuery]);

  const commentTopicScrollRef = useRef<HTMLDivElement>(null);
  const commentTopicVirtualizer = useVirtualizer({
    count: filteredTopics.length,
    getScrollElement: () => commentTopicScrollRef.current,
    estimateSize: () => 120,
    overscan: 8,
    getItemKey: (index) => String(filteredTopics[index]?.id ?? index),
  });

  return (
    <div className="min-h-screen text-slate-900 pb-10 md:pb-14">
      {/* 顶部导航栏 */}
      <header className="bg-white/80 backdrop-blur-md border-b border-blue-100 px-6 py-4 sticky top-[65px] z-30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-lg">💬</span>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-500 to-cyan-600 bg-clip-text text-transparent">
                小红书AI主题评论分析
              </h1>
              <p className="text-xs text-gray-400 max-w-xl">
                基于 MacBERT 的评论情感反馈
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* 搜索框 */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="搜索笔记 ID、关键词…"
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
        <div className="flex flex-wrap items-center gap-6 mt-4 pt-4 border-t border-blue-50">
          {loading && (
            <span className="text-sm text-blue-600">正在加载 comment 数据…</span>
          )}
          {loadErr && (
            <span className="text-sm text-red-600 max-w-xl" title={loadErr}>
              加载失败：{loadErr}（请确认在项目根下执行 npm run dev，且 visualization 与 comment 目录相对位置未改）
            </span>
          )}
          {bundle &&
            bundle.predictionsMeta.sampleSize > 0 &&
            bundle.predictionsMeta.sampleSize < bundle.predictionsMeta.total && (
              <span
                className="text-xs text-amber-800 bg-amber-50 border border-amber-200/80 rounded-lg px-2.5 py-1 max-w-xl leading-snug"
                title="情感与笔记卡片仅基于已拉取的预测子集；可在环境变量中提高 VITE_COMMENT_PRED_SAMPLE_LIMIT 后重启 dev"
              >
                当前仅加载预测评论 {bundle.predictionsMeta.sampleSize.toLocaleString()} /{' '}
                {bundle.predictionsMeta.total.toLocaleString()} 条：全库极性饼图仍完整，单笔记情感以已加载子集为准。
              </span>
            )}
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-blue-500">{metrics.totalComments.toLocaleString()}</span>
            <span className="text-sm text-gray-500">总评论（极性表）</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-green-500">{(metrics.positiveRatio * 100).toFixed(1)}%</span>
            <span className="text-sm text-gray-500">正向</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-gray-500">{(metrics.neutralRatio * 100).toFixed(1)}%</span>
            <span className="text-sm text-gray-500">中性</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-red-500">{(metrics.negativeRatio * 100).toFixed(1)}%</span>
            <span className="text-sm text-gray-500">负向</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-cyan-500">
              {metrics.avgSentimentScore.toFixed(3)}
            </span>
            <span className="text-sm text-gray-500">极性均值 (−1～1)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-indigo-500">{metrics.avgCommentLength || '—'}</span>
            <span className="text-sm text-gray-500">均长（字）</span>
          </div>
        </div>
      </header>

      <main className="flex h-[calc(100vh-180px)] min-h-0">
        {/* 左侧评论主题列表（虚拟滚动，数据仍为全量） */}
        <aside className="w-72 shrink-0 bg-white/60 backdrop-blur-sm border-r border-blue-100 flex flex-col min-h-0 overflow-hidden p-4">
          <h2 className="font-semibold text-gray-700 mb-4 flex items-center gap-2 shrink-0">
            <MessageSquare className="w-4 h-4 text-blue-500" />
            笔记列表（按评论数排序）
          </h2>
          <div ref={commentTopicScrollRef} className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden -mx-1 px-1">
            <div
              className="relative w-full"
              style={{ height: commentTopicVirtualizer.getTotalSize() }}
            >
              {commentTopicVirtualizer.getVirtualItems().map((vRow) => {
                const topic = filteredTopics[vRow.index];
                return (
                  <div
                    key={vRow.key}
                    className="absolute left-0 top-0 w-full pb-2"
                    style={{ transform: `translateY(${vRow.start}px)` }}
                  >
                    <div
                      onClick={() => setSelectedTopic(topic)}
                      className={`p-3 rounded-xl cursor-pointer transition-all duration-200 hover:shadow-md ${
                        selectedTopic?.id === topic.id
                          ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg'
                          : 'bg-white hover:bg-blue-50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-sm">{topic.name}</span>
                        <span
                          className={`text-xs ${selectedTopic?.id === topic.id ? 'text-white/80' : 'text-gray-400'}`}
                          title={
                            topic.sentimentFromCount != null && topic.sentimentFromCount < topic.commentCount
                              ? `排名表评论数 ${topic.commentCount.toLocaleString()}，已加载预测 ${topic.sentimentFromCount.toLocaleString()} 条（情感条柱仅基于后者）`
                              : `评论数 ${topic.commentCount.toLocaleString()}（与预测条数一致）`
                          }
                        >
                          {topic.sentimentFromCount != null && topic.sentimentFromCount < topic.commentCount
                            ? `${topic.sentimentFromCount.toLocaleString()}/${topic.commentCount.toLocaleString()}`
                            : topic.commentCount.toLocaleString()}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 mt-0.5">
                        <div
                          className={`flex-1 h-2 min-w-0 rounded-full overflow-hidden flex ${
                            selectedTopic?.id === topic.id
                              ? 'bg-white/14 ring-1 ring-white/35 shadow-inner shadow-black/5'
                              : 'ring-1 ring-slate-200/90 bg-slate-100'
                          }`}
                          title={`积极 ${(topic.positiveRatio * 100).toFixed(0)}% · 中性 ${(topic.neutralRatio * 100).toFixed(0)}% · 消极 ${(topic.negativeRatio * 100).toFixed(0)}%`}
                        >
                          {topic.positiveRatio > 0 && (
                            <div
                              className="h-full shrink-0 transition-[width] duration-300 ease-out"
                              style={{
                                width: `${topic.positiveRatio * 100}%`,
                                minWidth: topic.positiveRatio >= 0.03 ? undefined : 3,
                                backgroundColor:
                                  selectedTopic?.id === topic.id ? '#6ee7b7' : SENT_COLORS.pos,
                              }}
                            />
                          )}
                          {topic.neutralRatio > 0 && (
                            <div
                              className="h-full shrink-0 transition-[width] duration-300 ease-out"
                              style={{
                                width: `${topic.neutralRatio * 100}%`,
                                minWidth: topic.neutralRatio >= 0.03 ? undefined : 3,
                                backgroundColor:
                                  selectedTopic?.id === topic.id ? 'rgba(255,255,255,0.42)' : SENT_COLORS.neu,
                              }}
                            />
                          )}
                          {topic.negativeRatio > 0 && (
                            <div
                              className="h-full shrink-0 transition-[width] duration-300 ease-out"
                              style={{
                                width: `${topic.negativeRatio * 100}%`,
                                minWidth: topic.negativeRatio >= 0.03 ? undefined : 3,
                                backgroundColor:
                                  selectedTopic?.id === topic.id ? '#fda4af' : SENT_COLORS.neg,
                              }}
                            />
                          )}
                        </div>
                        <span
                          className={`text-[10px] tabular-nums shrink-0 font-medium leading-none ${
                            selectedTopic?.id === topic.id ? 'text-white/90' : 'text-slate-500'
                          }`}
                        >
                          {(topic.positiveRatio * 100).toFixed(0)}%
                          <span
                            className={`ml-0.5 font-normal ${selectedTopic?.id === topic.id ? 'text-white/65' : 'text-slate-400'}`}
                          >
                            积极
                          </span>
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>

        {/* 中间主可视化区域 */}
        <section className="flex-1 flex flex-col min-w-0 p-3 sm:p-4 md:p-6 overflow-hidden">
          <div className="flex flex-col gap-3 mb-4 sm:mb-6 shrink-0">
            {/* 一级：大卡片模块切换，选中态与顶栏「评论分析」NavLink 同色 */}
            <div
              className="flex flex-wrap gap-2 p-1.5 rounded-2xl bg-white/70 border border-slate-200/80 shadow-sm backdrop-blur-sm"
              role="tablist"
              aria-label="评论分析模块"
            >
              {COMMENT_MAIN_MODULES.map((mod) => {
                const ModIcon = mod.icon;
                const isOn = mod.id === commentModule;
                return (
                  <button
                    key={mod.id}
                    type="button"
                    role="tab"
                    aria-selected={isOn}
                    onClick={() => switchCommentModule(mod.id)}
                    className={`flex-1 min-w-[140px] sm:min-w-[200px] inline-flex items-center gap-2.5 px-3 sm:px-4 py-2.5 rounded-xl text-left transition-all border ${
                      isOn
                        ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white border-transparent shadow-lg'
                        : 'bg-transparent text-gray-700 border-transparent hover:bg-gray-100'
                    }`}
                  >
                    <ModIcon
                      className={`w-5 h-5 shrink-0 ${isOn ? 'text-white' : 'text-gray-600'}`}
                      aria-hidden
                    />
                    <span className="min-w-0">
                      <span
                        className={`block text-sm font-semibold leading-tight ${isOn ? 'text-white' : 'text-gray-800'}`}
                      >
                        {mod.label}
                      </span>
                      <span
                        className={`block text-[11px] mt-0.5 leading-snug ${isOn ? 'text-white/85' : 'text-gray-500'}`}
                      >
                        {mod.hint}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>

            {/* 二级：当前模块下的视图（结构与笔记主题页一致） */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wide mr-1 w-full sm:w-auto sm:mr-0">
                {commentModule === 'feedback' ? '评论视图' : '模型视图'}
              </span>
              {commentModule === 'feedback' &&
                FEEDBACK_TABS.map((tab) => {
                  const TabIcon = tab.icon;
                  const isOn = feedbackView === tab.id;
                  return (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setFeedbackView(tab.id)}
                      className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs sm:text-sm font-medium transition-all shrink-0 border ${
                        isOn
                          ? 'bg-blue-500 text-white border-blue-500 shadow-md'
                          : 'bg-white text-gray-700 border-gray-100 hover:bg-blue-50 hover:border-blue-100'
                      }`}
                    >
                      <TabIcon className="w-4 h-4 shrink-0 opacity-90" aria-hidden />
                      <span className="whitespace-nowrap hidden sm:inline">{tab.label}</span>
                      <span className="whitespace-nowrap sm:hidden">{tab.short}</span>
                    </button>
                  );
                })}
              {commentModule === 'model' &&
                MODEL_TABS.map((tab) => {
                  const TabIcon = tab.icon;
                  const isOn = modelSubView === tab.id;
                  return (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setModelSubView(tab.id)}
                      className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs sm:text-sm font-medium transition-all shrink-0 border ${
                        isOn
                          ? 'bg-blue-500 text-white border-blue-500 shadow-md'
                          : 'bg-white text-gray-700 border-gray-100 hover:bg-blue-50 hover:border-blue-100'
                      }`}
                    >
                      <TabIcon className="w-4 h-4 shrink-0 opacity-90" aria-hidden />
                      <span className="whitespace-nowrap">{tab.label}</span>
                    </button>
                  );
                })}
            </div>
          </div>

          <div className="relative flex-1 min-h-[min(400px,70vh)] rounded-2xl border border-slate-200/90 bg-white/80 backdrop-blur-sm shadow-xl overflow-hidden flex flex-col">
            {loading && (
              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-white/75 backdrop-blur-sm">
                <div className="h-10 w-10 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin" />
                <p className="text-sm text-slate-600">加载 comment 真实数据…</p>
              </div>
            )}
            <div className="flex-1 overflow-y-auto overscroll-contain p-5 md:p-8">
              <div
                key={
                  commentModule === 'feedback'
                    ? feedbackView
                    : `model-${modelSubView}`
                }
                className="comment-view-enter"
              >
                {commentModule === 'feedback' && feedbackView === 'sentiment' && bundle && (
                  <SentimentView bundle={bundle} />
                )}
                {commentModule === 'feedback' && feedbackView === 'notes' && bundle && (
                  <NotesTopicsUnifiedView
                    analysis={filteredAnalysis}
                    topics={topics}
                    onNoteClick={setSelectedNoteAnalysis}
                  />
                )}
                {commentModule === 'model' && bundle?.evalParse && (
                  <RoBERTaModelView
                    config={bundle.robertaConfig}
                    metrics={bundle.evalParse.metrics}
                    classData={bundle.evalParse.classData}
                    history={bundle.trainingHistory}
                    trainingHistoryMeta={bundle.trainingHistoryMeta}
                    evalViz={bundle.evalViz}
                    subView={modelSubView}
                  />
                )}
                {commentModule === 'model' && !bundle?.evalParse && !loading && (
                  <div className="rounded-xl border border-amber-100 bg-amber-50/60 p-6 text-amber-900 text-sm">
                    无法读取 evaluation_report.txt。请确认已启动开发服务器，且 comment/results/ 可被访问。
                  </div>
                )}
                {!bundle && !loading && <p className="text-slate-500 text-sm">暂无数据</p>}
              </div>
            </div>
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

const POLARITY_RAD = Math.PI / 180;

type PolarityPieLabelPayload = {
  cx: number;
  cy: number;
  midAngle: number;
  innerRadius: number;
  outerRadius: number;
  name: string;
  value: number;
  percent: number;
  fill: string;
};

/** 圆环外围：分类名 + 占比 + 条数 */
function PolarityOuterLabel(props: PolarityPieLabelPayload) {
  const { cx, cy, midAngle, outerRadius, name, value, percent, fill } = props;
  const r = outerRadius + 36;
  const x = cx + r * Math.cos(-midAngle * POLARITY_RAD);
  const y = cy + r * Math.sin(-midAngle * POLARITY_RAD);
  const anchor = x > cx + 8 ? 'start' : x < cx - 8 ? 'end' : 'middle';
  const pct = (percent * 100).toFixed(1);

  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      dominantBaseline="middle"
      className="select-none"
      fill="#334155"
    >
      <tspan x={x} dy="-0.55em" fontSize={12} fontWeight={600} fill={fill}>
        {name}
      </tspan>
      <tspan x={x} dy="1.35em" fontSize={11} fill="#64748b">
        {pct}% · {value.toLocaleString()} 条
      </tspan>
    </text>
  );
}

/** 全库极性（prediction_stats_polarity.csv）；展示在「情感分布」页首部 */
function GlobalPolaritySection({ bundle }: { bundle: LiveCommentBundle }) {
  const pieData = bundle.polarity
    ? [
        { name: '正向', value: bundle.polarity.positive, color: SENT_COLORS.pos },
        { name: '中性', value: bundle.polarity.neutral, color: SENT_COLORS.neu },
        { name: '负向', value: bundle.polarity.negative, color: SENT_COLORS.neg },
      ]
    : [];
  const totalPolarity = bundle.polarity?.total ?? 0;

  return (
    <div className="space-y-6 pb-2 border-b border-slate-100">
      <header>
        <h3 className="text-base font-semibold tracking-tight text-slate-800">全库情感极性</h3>
      </header>

      <div className="relative overflow-hidden rounded-2xl border border-slate-200/70 bg-gradient-to-b from-white via-slate-50/40 to-emerald-50/15 p-5 md:p-6 shadow-[0_20px_50px_-24px_rgba(15,23,42,0.14)] ring-1 ring-white/80">
        <div
          className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-gradient-to-br from-emerald-200/25 to-cyan-200/15 blur-3xl"
          aria-hidden
        />

        <div className="relative flex flex-col items-center gap-5">
          <div className="flex w-full flex-col items-center gap-1 text-center sm:flex-row sm:justify-center sm:gap-3 sm:text-left">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-md shadow-emerald-500/20">
              <PieChart className="h-[18px] w-[18px]" aria-hidden />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-800">三分类占比</h4>
              <p className="text-[11px] text-slate-500">
                全量预测汇总 · 共{' '}
                <span className="font-medium tabular-nums text-slate-700">{totalPolarity.toLocaleString()}</span> 条
              </p>
            </div>
          </div>

          {totalPolarity > 0 ? (
            <div className="relative w-full max-w-lg">
              <div className="h-[min(380px,52vh)] w-full min-h-[280px] sm:min-h-[320px]">
                <ResponsiveContainerAny width="100%" height="100%">
                  <PieChartRecharts margin={{ top: 28, right: 52, bottom: 28, left: 52 }}>
                    <PieAny
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius="36%"
                      outerRadius="50%"
                      paddingAngle={2}
                      stroke="#fff"
                      strokeWidth={2}
                      labelLine={{ stroke: '#cbd5e1', strokeWidth: 1 }}
                      label={(p: PolarityPieLabelPayload) => <PolarityOuterLabel {...p} />}
                    >
                      {pieData.map((e, i) => (
                        <CellAny key={i} fill={e.color} />
                      ))}
                    </PieAny>
                    <TooltipAny
                      formatter={(v: number) => [v.toLocaleString(), '条']}
                      contentStyle={CHART_TOOLTIP_STYLE}
                    />
                  </PieChartRecharts>
                </ResponsiveContainerAny>
              </div>
              <div className="pointer-events-none absolute left-1/2 top-1/2 z-10 flex max-w-[7.5rem] -translate-x-1/2 -translate-y-1/2 flex-col items-center justify-center px-2 text-center">
                <span className="text-[10px] font-medium uppercase tracking-wider text-slate-400">合计</span>
                <span className="text-xl font-semibold tabular-nums tracking-tight text-slate-800 sm:text-2xl">
                  {totalPolarity.toLocaleString()}
                </span>
                <span className="text-[11px] text-slate-500">条评论</span>
              </div>
            </div>
          ) : (
            <p className="py-8 text-sm text-slate-500">暂无极性汇总数据。</p>
          )}
        </div>
      </div>
    </div>
  );
}

// 情感分布组件
function SentimentView({ bundle }: { bundle: LiveCommentBundle }) {
  const m = bundle.metrics;
  const stackData = bundle.lengthBySentiment;

  return (
    <div className="h-full space-y-8">
      <GlobalPolaritySection bundle={bundle} />
      <div>
        <h3 className="text-lg font-semibold tracking-tight text-slate-800">情感分布详情</h3>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-emerald-50 to-green-100 rounded-2xl p-6 border border-emerald-100/60 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center shadow-md">
              <Heart className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-3xl font-bold text-green-700">
                {(m.positiveRatio * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-green-800/80">正向（全库）</div>
            </div>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            label=1：赞扬、认同与正面情绪表达。
          </p>
        </div>

        <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-2xl p-6 border border-slate-200/80 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-slate-400 rounded-xl flex items-center justify-center shadow-md">
              <MessageCircle className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-3xl font-bold text-slate-700">
                {(m.neutralRatio * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-slate-600">中性（全库）</div>
            </div>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">label=0：陈述、询问等中性语气。</p>
        </div>

        <div className="bg-gradient-to-br from-red-50 to-rose-100 rounded-2xl p-6 border border-rose-100/70 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-red-500 rounded-xl flex items-center justify-center shadow-md">
              <ThumbsDown className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-3xl font-bold text-red-700">
                {(m.negativeRatio * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-red-800/80">负向（全库）</div>
            </div>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">label=−1：抱怨、批评与负面体验。</p>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
        <h4 className="font-semibold text-slate-700 mb-2">评论字数 × 情感</h4>
        <p className="text-xs text-slate-500 mb-4">按字符长度分桶，堆叠展示全量预测评论的极性构成</p>
        <div className="h-[300px]">
          <ResponsiveContainerAny width="100%" height="100%">
            <BarChartAny data={stackData}>
              <CartesianGridAny strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxisAny dataKey="bucket" tick={{ fontSize: 11 }} />
              <YAxisAny tick={{ fontSize: 11 }} />
              <TooltipAny contentStyle={CHART_TOOLTIP_STYLE} />
              <LegendAny wrapperStyle={{ fontSize: 12 }} />
              <BarAny dataKey="positive" stackId="a" fill={SENT_COLORS.pos} name="正向" radius={[0, 0, 0, 0]} />
              <BarAny dataKey="neutral" stackId="a" fill={SENT_COLORS.neu} name="中性" />
              <BarAny dataKey="negative" stackId="a" fill={SENT_COLORS.neg} name="负向" radius={[6, 6, 0, 0]} />
            </BarChartAny>
          </ResponsiveContainerAny>
        </div>
      </div>

      <CommentGeoInsightSections bundle={bundle} embedded />

      <div className="rounded-2xl bg-gradient-to-r from-cyan-50/90 to-indigo-50/50 border border-cyan-100/80 p-6 shadow-sm">
        <h4 className="font-semibold text-slate-700 mb-3">整体极性均值</h4>
        <div className="flex flex-wrap items-center gap-4">
          <div className="text-4xl font-bold text-cyan-700 tabular-nums">
            {m.avgSentimentScore.toFixed(3)}
          </div>
          <div className="flex-1 min-w-[200px] h-3 rounded-full bg-slate-200 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-rose-400 via-slate-300 to-emerald-500"
              style={{
                width: `${Math.min(100, Math.max(0, ((m.avgSentimentScore + 1) / 2) * 100))}%`,
              }}
            />
          </div>
          <span className="text-sm text-slate-600">范围 −1（偏负）～ +1（偏正）</span>
        </div>
      </div>
    </div>
  );
}

/** 离线汇总的宏观情感表 + 全量/联表对比；首次挂载写入 localStorage，避免重复解析 */
function TopicCorrelationSnapshotSection() {
  useEffect(() => {
    try {
      localStorage.setItem(TOPIC_CORRELATION_LS_KEY, JSON.stringify(topicCorrelationSnapshot));
    } catch {
      /* 存储配额或隐私模式 */
    }
  }, []);

  const { macroSentiment, linkedVsAll } = topicCorrelationSnapshot;

  const stackBarData = macroSentiment.map(r => {
    const tail = r.macro.replace(/^AI/, '').trim();
    const short = tail.length > 6 ? `${tail.slice(0, 5)}…` : tail;
    return {
      short,
      full: r.macro,
      正向: r.positive,
      中性: r.neutral,
      负向: r.negative,
      total: r.total,
      positivePct: r.positivePct,
      neutralPct: r.neutralPct,
      negativePct: r.negativePct,
      pnRatio: r.pnRatio,
    };
  });

  const comparePctData = [
    {
      name: '正向占比',
      全量: linkedVsAll.allComments.positivePct,
      联表: linkedVsAll.linkedMatched.positivePct,
      diffPp: linkedVsAll.diffPp.positivePct,
    },
    {
      name: '中性占比',
      全量: linkedVsAll.allComments.neutralPct,
      联表: linkedVsAll.linkedMatched.neutralPct,
      diffPp: linkedVsAll.diffPp.neutralPct,
    },
    {
      name: '负向占比',
      全量: linkedVsAll.allComments.negativePct,
      联表: linkedVsAll.linkedMatched.negativePct,
      diffPp: linkedVsAll.diffPp.negativePct,
    },
  ];

  return (
    <div className="space-y-6 mb-8 pb-8 border-b border-slate-100">
      <header>
        <h4 className="text-sm font-semibold text-slate-800">主题关联 · 情感结构快照</h4>
        <p className="text-[11px] text-slate-500 mt-1 max-w-3xl leading-relaxed">
          仅保留图表展示；悬停柱条可查看条数、占比与 P/N。数据随前端打包，首次进入会写入{' '}
          <code className="text-[10px] bg-slate-100 px-1 rounded">localStorage</code>（{TOPIC_CORRELATION_LS_KEY}）以便本地快速复用。
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        <div className="rounded-2xl border border-slate-200/80 bg-white shadow-sm overflow-hidden flex flex-col min-h-0">
          <div className="px-4 py-2.5 border-b border-slate-100 bg-slate-50/70 shrink-0">
            <span className="text-xs font-semibold text-slate-700">各宏观主题 · 情感条数（堆叠）</span>
            <p className="text-[10px] text-slate-500 mt-0.5">鼠标悬停查看占比与 P/N</p>
          </div>
          <div className="h-[min(300px,42vh)] min-h-[220px] p-1 flex-1">
            <ResponsiveContainerAny width="100%" height="100%">
              <BarChartAny layout="vertical" data={stackBarData} margin={{ top: 4, right: 8, left: 4, bottom: 4 }}>
                <CartesianGridAny strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                <XAxisAny type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <YAxisAny type="category" dataKey="short" width={76} tick={{ fontSize: 10, fill: '#475569' }} />
                <TooltipAny
                  content={({ payload }: { payload?: Array<{ payload: (typeof stackBarData)[0] }> }) => {
                    const p = payload?.[0]?.payload;
                    if (!p) return null;
                    return (
                      <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg max-w-[260px]">
                        <div className="font-semibold text-slate-800">{p.full}</div>
                        <div className="mt-1.5 space-y-1 tabular-nums text-slate-600 leading-relaxed">
                          <div>总评论 {p.total.toLocaleString()} 条</div>
                          <div>
                            正 {p.正向.toLocaleString()} · 中 {p.中性.toLocaleString()} · 负 {p.负向.toLocaleString()}
                          </div>
                          <div className="text-[11px] text-slate-500">
                            正 {p.positivePct.toFixed(2)}% · 中 {p.neutralPct.toFixed(2)}% · 负 {p.negativePct.toFixed(2)}%
                          </div>
                          <div className="text-slate-700">P/N 比 {p.pnRatio.toFixed(2)}</div>
                        </div>
                      </div>
                    );
                  }}
                  contentStyle={CHART_TOOLTIP_STYLE}
                />
                <LegendAny wrapperStyle={{ fontSize: 11 }} />
                <BarAny dataKey="正向" stackId="a" fill={SENT_COLORS.pos} name="正向" />
                <BarAny dataKey="中性" stackId="a" fill={SENT_COLORS.neu} name="中性" />
                <BarAny dataKey="负向" stackId="a" fill={SENT_COLORS.neg} name="负向" radius={[0, 3, 3, 0]} />
              </BarChartAny>
            </ResponsiveContainerAny>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200/80 bg-white shadow-sm overflow-hidden flex flex-col min-h-0">
          <div className="px-4 py-2.5 border-b border-slate-100 bg-slate-50/70 shrink-0">
            <span className="text-xs font-semibold text-slate-700">全量评论 vs 联表后（可匹配）</span>
            <p className="text-[10px] text-slate-500 mt-0.5">悬停查看差值（pp）</p>
          </div>
          <div className="h-[min(300px,42vh)] min-h-[220px] p-2 flex-1">
            <ResponsiveContainerAny width="100%" height="100%">
              <BarChartAny data={comparePctData} margin={{ top: 8, right: 8, left: 4, bottom: 4 }}>
                <CartesianGridAny strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxisAny dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} interval={0} />
                <YAxisAny
                  tick={{ fontSize: 10, fill: '#94a3b8' }}
                  domain={[0, 45]}
                  tickFormatter={(v: number) => `${v}%`}
                />
                <TooltipAny
                  content={({ payload }: { payload?: Array<{ payload: (typeof comparePctData)[0] }> }) => {
                    const p = payload?.[0]?.payload;
                    if (!p) return null;
                    const d = p.diffPp;
                    return (
                      <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg">
                        <div className="font-semibold text-slate-800">{p.name}</div>
                        <div className="mt-1.5 space-y-0.5 tabular-nums text-slate-600">
                          <div>全量 {p.全量.toFixed(2)}%</div>
                          <div>联表 {p.联表.toFixed(2)}%</div>
                          <div className={d >= 0 ? 'text-emerald-700' : 'text-rose-700'}>
                            差异 {d >= 0 ? '+' : ''}
                            {d.toFixed(2)}pp
                          </div>
                        </div>
                      </div>
                    );
                  }}
                  contentStyle={CHART_TOOLTIP_STYLE}
                />
                <LegendAny wrapperStyle={{ fontSize: 11 }} />
                <BarAny dataKey="全量" fill="#6366f1" name="全量评论" radius={[4, 4, 0, 0]} />
                <BarAny dataKey="联表" fill="#14b8a6" name="联表后(可匹配)" radius={[4, 4, 0, 0]} />
              </BarChartAny>
            </ResponsiveContainerAny>
          </div>
          <div className="px-4 py-2 border-t border-slate-100 bg-slate-50/40 text-[10px] text-slate-500 tabular-nums flex flex-wrap gap-x-4 gap-y-1">
            <span>
              正/负比：全量 {linkedVsAll.allComments.pnRatio.toFixed(2)} · 联表 {linkedVsAll.linkedMatched.pnRatio.toFixed(2)} · 差异{' '}
              {linkedVsAll.diffPnRatio >= 0 ? '+' : ''}
              {linkedVsAll.diffPnRatio.toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/** 笔记 × 主题 × 评论：主题关联视图，数据源一致（note_id 对齐） */
function NotesTopicsUnifiedView({
  analysis,
  topics,
  onNoteClick,
}: {
  analysis: CommentAnalysis[];
  topics: CommentTopic[];
  onNoteClick: (a: CommentAnalysis) => void;
}) {
  const [selectedMacroTopic, setSelectedMacroTopic] = useState<string | null>(null);

  const topicByNoteId = useMemo(() => {
    const m = new Map<string, CommentTopic>();
    for (const t of topics) {
      if (t.noteId) m.set(t.noteId, t);
    }
    return m;
  }, [topics]);

  const macroTopicAgg = useMemo(() => {
    type Agg = {
      macro: string;
      noteIds: Set<string>;
      commentCount: number;
      pos: number;
      neu: number;
      neg: number;
      avgScoreWeightedSum: number;
      kwFreq: Map<string, number>;
    };

    const aggByMacro = new Map<string, Agg>();
    for (const t of topics) {
      const macro = t.contentMacroTopic?.trim();
      if (!macro) continue;

      const a =
        aggByMacro.get(macro) ??
        ({
          macro,
          noteIds: new Set<string>(),
          commentCount: 0,
          pos: 0,
          neu: 0,
          neg: 0,
          avgScoreWeightedSum: 0,
          kwFreq: new Map<string, number>(),
        } satisfies Agg);

      if (t.noteId) a.noteIds.add(t.noteId.trim());

      // 情感比例来自「已预测评论」子集，聚合权重用 sentimentFromCount，避免用全量评论数放大偏差
      const w =
        t.sentimentFromCount != null && t.sentimentFromCount > 0
          ? t.sentimentFromCount
          : Math.max(0, Number.isFinite(t.commentCount) ? t.commentCount : 0);
      a.commentCount += w;
      a.pos += w * (t.positiveRatio ?? 0);
      a.neu += w * (t.neutralRatio ?? 0);
      a.neg += w * (t.negativeRatio ?? 0);
      a.avgScoreWeightedSum += w * (t.avgSentimentScore ?? 0);

      for (const kw of t.keywords ?? []) {
        const k = kw?.trim();
        if (!k) continue;
        a.kwFreq.set(k, (a.kwFreq.get(k) ?? 0) + 1);
      }

      aggByMacro.set(macro, a);
    }

    const rows = Array.from(aggByMacro.values()).map(a => {
      const denom = a.pos + a.neu + a.neg || a.commentCount || 1;
      const positiveRatio = a.pos / denom;
      const neutralRatio = a.neu / denom;
      const negativeRatio = a.neg / denom;
      const avgSentimentScore = a.commentCount ? a.avgScoreWeightedSum / a.commentCount : 0;
      const topKeywords = Array.from(a.kwFreq.entries())
        .sort((x, y) => y[1] - x[1])
        .slice(0, 8)
        .map(([k]) => k);

      return {
        macro: a.macro,
        noteCount: a.noteIds.size,
        commentCount: Math.round(a.commentCount),
        positiveRatio,
        neutralRatio,
        negativeRatio,
        avgSentimentScore,
        topKeywords,
      };
    });

    rows.sort((x, y) => y.commentCount - x.commentCount);
    return rows;
  }, [topics]);

  const filteredAnalysisByMacro = useMemo(() => {
    if (!selectedMacroTopic) return analysis;
    return analysis.filter(item => {
      const t = item.noteId ? topicByNoteId.get(item.noteId) : undefined;
      const macro = t?.contentMacroTopic?.trim();
      return macro === selectedMacroTopic;
    });
  }, [analysis, selectedMacroTopic, topicByNoteId]);

  /** 列表最多展示前 20 条（按匹配评论数降序），减轻渲染 */
  const NOTES_LIST_LIMIT = 20;
  const displayedAnalysis = useMemo(() => {
    const list = [...filteredAnalysisByMacro].sort((a, b) => {
      const ta = a.noteId ? topicByNoteId.get(a.noteId) : undefined;
      const tb = b.noteId ? topicByNoteId.get(b.noteId) : undefined;
      const ca = ta?.commentCount ?? a.commentCount ?? 0;
      const cb = tb?.commentCount ?? b.commentCount ?? 0;
      return cb - ca;
    });
    return list.slice(0, NOTES_LIST_LIMIT);
  }, [filteredAnalysisByMacro, topicByNoteId]);

  return (
    <div className="h-full overflow-auto">
      <div className="mb-6">
        <h3 className="text-lg font-semibold tracking-tight text-slate-800">主题关联（note_id 对齐）</h3>
        <p className="text-xs text-slate-500 mt-1 max-w-3xl leading-relaxed">
          每条对应一篇高评论笔记：主题侧与 content BERTopic 同 note_id；下列为评论侧情感结构与关键词。
          <span className="text-slate-600">点击卡片</span>可打开完整字段与 content 关联摘要（原「详细列表」行为）。
        </p>
      </div>

      <TopicCorrelationSnapshotSection />

      <div className="mb-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h4 className="text-sm font-semibold text-slate-800">主题情感分布（按 BERTopic 宏观主题聚合）</h4>
            <p className="text-xs text-slate-500 mt-1">
              将评论侧情感（MacBERT）按笔记 <span className="font-medium text-slate-600">note_id</span> 与 content/BERTopic 同键对齐后汇总到宏观主题；卡片中的条数为
              <span className="font-medium text-slate-600">已加载预测 JSON 内的评论条数</span>（与全站评论总数可能不同）。
              点击某个主题可筛选下方笔记列表。
            </p>
          </div>
          <button
            type="button"
            onClick={() => setSelectedMacroTopic(null)}
            className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-colors ${
              selectedMacroTopic
                ? 'border-slate-200 bg-white hover:bg-slate-50 text-slate-700'
                : 'border-slate-100 bg-slate-50 text-slate-400 cursor-default'
            }`}
            disabled={!selectedMacroTopic}
            title={selectedMacroTopic ? '清除主题筛选' : '未选择主题'}
          >
            清除筛选
          </button>
        </div>

        {macroTopicAgg.length === 0 ? (
          <div className="mt-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-xs text-slate-500">
            当前 topics 中缺少 <span className="font-mono">contentMacroTopic</span>，无法生成“按主题聚合”的情感分布。
          </div>
        ) : (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {macroTopicAgg.map(row => {
              const isOn = selectedMacroTopic === row.macro;
              const tc = macroTopicDisplayColor(row.macro);
              return (
                <button
                  key={row.macro}
                  type="button"
                  onClick={() => setSelectedMacroTopic(isOn ? null : row.macro)}
                  className="text-left rounded-2xl border border-slate-200/80 p-4 transition-all shadow-sm hover:shadow-md bg-white"
                  style={{
                    borderLeftWidth: 4,
                    borderLeftColor: tc,
                    ...(isOn
                      ? {
                          backgroundColor: macroTopicColorWithAlpha(row.macro, 0.12),
                          boxShadow: `0 0 0 2px ${macroTopicColorWithAlpha(row.macro, 0.42)}`,
                        }
                      : {}),
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="font-semibold truncate" style={{ color: tc }}>
                        {row.macro}
                      </div>
                      <div className="mt-1 text-xs text-slate-500 tabular-nums">
                        覆盖 {row.noteCount.toLocaleString()} 篇笔记 · {row.commentCount.toLocaleString()} 条预测评论
                      </div>
                    </div>
                    <div className="shrink-0 text-right">
                      <div className="text-xs font-semibold text-slate-600 tabular-nums">
                        均值 {row.avgSentimentScore.toFixed(2)}
                      </div>
                      <div className="mt-1 text-[11px] text-slate-400 tabular-nums">
                        正 {(row.positiveRatio * 100).toFixed(0)}% · 中 {(row.neutralRatio * 100).toFixed(0)}% · 负{' '}
                        {(row.negativeRatio * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 flex h-2.5 rounded-full overflow-hidden ring-1 ring-slate-900/[0.06]">
                    <div style={{ width: `${row.positiveRatio * 100}%`, backgroundColor: SENT_COLORS.pos }} />
                    <div style={{ width: `${row.neutralRatio * 100}%`, backgroundColor: SENT_COLORS.neu }} />
                    <div style={{ width: `${row.negativeRatio * 100}%`, backgroundColor: SENT_COLORS.neg }} />
                  </div>

                  {row.topKeywords.length > 0 ? (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {row.topKeywords.map(kw => (
                        <span
                          key={kw}
                          className={`px-2 py-0.5 rounded-lg text-[11px] font-medium border ${
                            isOn ? 'text-slate-800' : 'bg-slate-50 text-slate-700 border-slate-200/80'
                          }`}
                          style={
                            isOn
                              ? {
                                  backgroundColor: macroTopicColorWithAlpha(row.macro, 0.18),
                                  borderColor: macroTopicColorWithAlpha(row.macro, 0.45),
                                }
                              : undefined
                          }
                        >
                          {kw}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </button>
              );
            })}
          </div>
        )}
      </div>
      {selectedMacroTopic ? (
        <p className="text-[11px] text-amber-900/90 bg-amber-50/90 border border-amber-200/70 rounded-xl px-3 py-2 mb-3 leading-relaxed">
          已选宏观主题「<span className="font-semibold">{selectedMacroTopic}</span>
          」：下方列表仅展示该主题下<strong className="mx-0.5">匹配评论数（排名表 commentCount）最高</strong>的 {NOTES_LIST_LIMIT}{' '}
          篇笔记，避免长列表卡顿。
        </p>
      ) : filteredAnalysisByMacro.length > NOTES_LIST_LIMIT ? (
        <p className="text-[11px] text-slate-600 bg-slate-50/90 border border-slate-200/80 rounded-xl px-3 py-2 mb-3 leading-relaxed">
          当前筛选共 {filteredAnalysisByMacro.length.toLocaleString()} 篇笔记，列表仅展示匹配评论数最高的 {NOTES_LIST_LIMIT}{' '}
          篇。
        </p>
      ) : null}

      {displayedAnalysis.length === 0 ? (
        <p className="text-sm text-slate-500 py-10 text-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/50">
          当前筛选下暂无笔记，请调整顶部搜索或情感筛选。
        </p>
      ) : null}
      <div className="space-y-3">
        {displayedAnalysis.map((item, i) => {
          const topic = item.noteId ? topicByNoteId.get(item.noteId) : undefined;
          const displayTitle = topic?.name ?? item.noteTitle;
          const macroLine = topic?.contentMacroTopic ?? item.topicName;
          const microId = topic?.contentMicroTopicId ?? item.contentMicroTopicId;
          const microKw = topic?.contentMicroKeywords ?? item.contentMicroKeywords;

          return (
            <div
              key={item.id}
              role="button"
              tabIndex={0}
              onClick={() => onNoteClick(item)}
              onKeyDown={e => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onNoteClick(item);
                }
              }}
              className="rounded-2xl border border-slate-200/80 bg-white p-4 md:p-5 cursor-pointer shadow-sm hover:shadow-md hover:border-indigo-200/70 transition-all duration-200 text-left"
            >
              <div className="flex items-center justify-between gap-3 mb-3">
                <div className="flex items-center gap-3 min-w-0">
                  <span
                    className={`w-9 h-9 shrink-0 rounded-xl border flex items-center justify-center text-sm font-bold tabular-nums ${
                      !macroLine ? 'bg-gradient-to-br from-cyan-500/15 to-indigo-600/15 text-indigo-700 border-indigo-100/80' : ''
                    }`}
                    style={
                      macroLine
                        ? {
                            backgroundColor: macroTopicColorWithAlpha(macroLine, 0.16),
                            color: macroTopicDisplayColor(macroLine),
                            borderColor: macroTopicColorWithAlpha(macroLine, 0.4),
                          }
                        : undefined
                    }
                  >
                    {i + 1}
                  </span>
                  <div className="min-w-0">
                    <span className="font-semibold text-slate-800 block truncate">{displayTitle}</span>
                    {macroLine && (
                      <span
                        className="text-xs mt-0.5 block leading-snug truncate font-medium"
                        style={{ color: macroTopicDisplayColor(macroLine) }}
                      >
                        BERTopic 宏观 · {macroLine}
                        {microId != null ? ` · 微观#${microId}` : ''}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1 shrink-0">
                  <span
                    className="text-sm text-slate-500 tabular-nums"
                    title={
                      topic &&
                      topic.sentimentFromCount != null &&
                      topic.sentimentFromCount < topic.commentCount
                        ? `排名表 ${topic.commentCount} 条 · 已预测 ${topic.sentimentFromCount} 条`
                        : undefined
                    }
                  >
                    {topic &&
                    topic.sentimentFromCount != null &&
                    topic.sentimentFromCount < topic.commentCount
                      ? `${topic.sentimentFromCount.toLocaleString()}/${topic.commentCount.toLocaleString()} 预测/排名`
                      : `${(topic?.commentCount ?? item.commentCount).toLocaleString()} 条评论`}
                  </span>
                  <span
                    className={`px-2.5 py-0.5 rounded-lg text-xs font-medium ${
                      item.sentiment === 'positive'
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-100'
                        : item.sentiment === 'negative'
                          ? 'bg-rose-50 text-rose-700 border border-rose-100'
                          : 'bg-slate-100 text-slate-600 border border-slate-200/80'
                    }`}
                  >
                    {item.sentiment === 'positive' ? '积极' : item.sentiment === 'negative' ? '消极' : '中性'}
                  </span>
                </div>
              </div>

              {topic ? (
                <div className="flex h-2.5 rounded-full overflow-hidden mb-3 ring-1 ring-slate-900/[0.06]">
                  <div
                    className="transition-all"
                    style={{ width: `${topic.positiveRatio * 100}%`, backgroundColor: SENT_COLORS.pos }}
                  />
                  <div
                    className="transition-all"
                    style={{ width: `${topic.neutralRatio * 100}%`, backgroundColor: SENT_COLORS.neu }}
                  />
                  <div
                    className="transition-all"
                    style={{ width: `${topic.negativeRatio * 100}%`, backgroundColor: SENT_COLORS.neg }}
                  />
                </div>
              ) : null}

              {!topic && microKw && (
                <div className="text-xs text-indigo-600 mb-3 leading-relaxed">
                  {microKw.length > 96 ? `${microKw.slice(0, 96)}…` : microKw}
                </div>
              )}

              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="flex flex-wrap gap-1.5">
                  {(topic?.keywords ?? item.keywords).map(kw => (
                    <span
                      key={kw}
                      className="px-2 py-0.5 rounded-lg text-xs font-medium bg-slate-50 text-slate-700 border border-slate-200/80"
                    >
                      {kw}
                    </span>
                  ))}
                </div>
                <div className="text-sm flex flex-wrap items-center gap-x-4 gap-y-1 shrink-0 tabular-nums text-slate-600">
                  {topic && (
                    <>
                      <span style={{ color: SENT_COLORS.pos }}>积极 {(topic.positiveRatio * 100).toFixed(0)}%</span>
                      <span className="text-slate-300 hidden sm:inline">·</span>
                      <span className="text-slate-500">均值 {topic.avgSentimentScore.toFixed(2)}</span>
                      <span className="text-slate-300 hidden sm:inline">·</span>
                    </>
                  )}
                  <span title="平均点赞">均赞 {item.avgCommentLikes}</span>
                  <span title="情感分">得分 {(item.sentimentScore * 100).toFixed(0)}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// 主题详情面板
function TopicDetailPanel({ topic }: { topic: CommentTopic }) {
  const href = resolveNotePageHref(topic.noteUrl, topic.noteId);
  const hasMicro =
    topic.contentMicroTopicId != null || Boolean(topic.contentMicroKeywords && topic.contentMicroKeywords.length > 0);

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-slate-200/90 bg-white shadow-md shadow-slate-200/30 overflow-hidden">
        <div className="h-1 w-full bg-gradient-to-r from-blue-500 via-cyan-500 to-sky-400" aria-hidden />
        <div className="p-4 sm:p-5 space-y-4">
          <div>
            <h2 className="text-[15px] sm:text-base font-semibold text-slate-900 leading-snug tracking-tight">
              {topic.name}
            </h2>
            {href && (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-cyan-200/90 bg-cyan-50/90 px-3 py-1.5 text-xs font-semibold text-cyan-800 hover:bg-cyan-100/90 transition-colors"
              >
                查看原帖 / 评论区
                <span aria-hidden className="text-cyan-600">
                  →
                </span>
              </a>
            )}
          </div>

          {!topic.contentMatched && topic.noteId && (
            <div className="rounded-lg border border-amber-200/80 bg-amber-50/90 px-3 py-2 text-[11px] text-amber-950 leading-relaxed">
              未在 content/final_pro_topics 中匹配到该笔记
            </div>
          )}

          {topic.contentMatched &&
            topic.sentimentFromCount != null &&
            topic.sentimentFromCount < topic.commentCount && (
              <div className="rounded-lg border border-sky-200/80 bg-sky-50/90 px-3 py-2 text-[11px] text-sky-950 leading-relaxed">
                左侧条柱与概览情感基于 <strong>已预测 {topic.sentimentFromCount.toLocaleString()} 条</strong>
                评论；排名表显示该笔记共 <strong>{topic.commentCount.toLocaleString()} 条</strong>
                评论。二者不一致时，请以预测子集为准理解情感分布。
              </div>
            )}

          {topic.contentMacroTopic && (
            <div className="rounded-xl border border-slate-100 bg-slate-50/80 px-3 py-2.5">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1">宏观主题</div>
              <p className="text-sm font-medium text-slate-800 leading-snug">{topic.contentMacroTopic}</p>
            </div>
          )}

          {hasMicro && (
            <div className="rounded-xl border border-indigo-100 bg-indigo-50/40 px-3 py-2.5 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-600/85">
                  BERTopic 微观
                </span>
                {topic.contentMicroTopicId != null && (
                  <span className="shrink-0 rounded-md border border-indigo-200/80 bg-white px-2 py-0.5 font-mono text-[11px] font-medium text-indigo-900 tabular-nums">
                    #{topic.contentMicroTopicId}
                  </span>
                )}
              </div>
              {topic.contentMicroKeywords && (
                <p className="text-xs text-slate-700 leading-relaxed max-h-24 overflow-y-auto pr-0.5">
                  {topic.contentMicroKeywords}
                </p>
              )}
              {topic.contentMappingConfidence != null && (
                <div className="flex items-center justify-between gap-2 pt-1 border-t border-indigo-100/80">
                  <span className="text-[11px] text-indigo-600/90">映射置信度</span>
                  <span className="text-xs font-bold tabular-nums text-indigo-900">
                    {(topic.contentMappingConfidence * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          )}

          {topic.keywords.length > 0 && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-2">评论关键词</div>
              <div className="flex flex-wrap gap-1.5">
                {topic.keywords.map((kw) => (
                  <span
                    key={kw}
                    className="rounded-md border border-slate-200/90 bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-700"
                  >
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-end justify-between gap-3 border-t border-slate-100 pt-3">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">评论量</div>
              <div className="text-lg font-bold tabular-nums text-slate-900 leading-none mt-1">
                {topic.commentCount.toLocaleString()}
              </div>
              <div className="text-[10px] text-slate-500 mt-1">rankings 全量</div>
            </div>
          </div>

          {topic.noteId && (
            <div className="rounded-lg bg-slate-100/80 px-2.5 py-2 border border-slate-100">
              <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400 mb-1">note_id</div>
              <p className="text-[10px] font-mono text-slate-600 break-all leading-snug">{topic.noteId}</p>
            </div>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-cyan-100/80 bg-cyan-50/40 p-3 text-xs text-cyan-950 leading-relaxed">
        <span className="font-semibold text-cyan-800">提示：</span>
        选中笔记后，右侧详情面板展示本篇摘要；全库极性统计见
        <strong className="mx-0.5">情感分布</strong>。
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
            {resolveNotePageHref(analysis.noteUrl, analysis.noteId) && (
              <a
                href={resolveNotePageHref(analysis.noteUrl, analysis.noteId)!}
                target="_blank"
                rel="noopener noreferrer"
                className="text-white/95 text-xs font-medium mt-1 underline underline-offset-2 hover:text-white inline-flex items-center gap-0.5"
              >
                查看原评论→
              </a>
            )}
            <p className="text-white/80 text-sm mt-1">
              {analysis.topicName} · 情感: {analysis.sentiment}
              {!analysis.contentMatched && ' · 未命中 content 主题表'}
              {analysis.sentimentSampleSize != null &&
                analysis.commentCount > 0 &&
                analysis.sentimentSampleSize < analysis.commentCount &&
                ` · 预测 ${analysis.sentimentSampleSize}/${analysis.commentCount} 条`}
            </p>
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
          
          {(analysis.noteDesc || analysis.noteContent || analysis.noteId) && (
            <div className="mb-6 rounded-xl border border-slate-200 bg-slate-50/80 p-4">
              <h3 className="font-semibold text-slate-800 mb-2 text-sm">笔记正文（content）</h3>
              {analysis.noteId && (
                <p className="text-[11px] text-slate-500 font-mono mb-2 break-all">note_id · {analysis.noteId}</p>
              )}
              {analysis.noteDesc && (
                <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap mb-3">{analysis.noteDesc}</p>
              )}
              {analysis.noteContent ? (
                <div className="max-h-48 overflow-y-auto rounded-lg bg-white p-3 text-sm text-slate-800 whitespace-pre-wrap border border-slate-100 leading-relaxed">
                  {analysis.noteContent}
                </div>
              ) : (
                <p className="text-xs text-amber-800">暂无合并正文。</p>
              )}
            </div>
          )}

          {/* 关键词 */}
          <div className="mb-6">
            <h3 className="font-semibold text-gray-700 mb-2">关键词（评论挖掘）</h3>
            <div className="flex flex-wrap gap-2">
              {analysis.keywords.map(kw => (
                <span key={kw} className="px-2 py-1 bg-blue-100 text-blue-600 rounded text-sm">
                  {kw}
                </span>
              ))}
            </div>
          </div>

          {analysis.contentMatched && (
            <div className="mb-6 rounded-xl border border-indigo-100 bg-indigo-50/70 p-4">
              <h3 className="font-semibold text-indigo-950 mb-2 text-sm">笔记主题（content / BERTopic）</h3>
              <p className="text-xs text-indigo-800/80 mb-3">
                以下「热门评论」均来自该笔记；主题信息用于理解评论所讨论的内容语境。
              </p>
              <dl className="text-sm text-gray-800 space-y-1.5">
                <div>
                  <span className="text-indigo-600/90">宏观主题 · </span>
                  <span className="font-medium">{analysis.topicName}</span>
                </div>
                {analysis.contentMicroTopicId != null && (
                  <div>
                    <span className="text-indigo-600/90">微观主题 ID · </span>
                    <span className="font-mono">{analysis.contentMicroTopicId}</span>
                  </div>
                )}
                {analysis.contentMappingConfidence != null && (
                  <div>
                    <span className="text-indigo-600/90">映射置信度 · </span>
                    <span>{(analysis.contentMappingConfidence * 100).toFixed(1)}%</span>
                  </div>
                )}
                {analysis.contentMicroKeywords && (
                  <div>
                    <div className="text-indigo-600/90 text-xs mb-0.5">微观关键词</div>
                    <p className="text-xs leading-relaxed">{analysis.contentMicroKeywords}</p>
                  </div>
                )}
                {analysis.contentTopicKeywords && (
                  <div>
                    <div className="text-indigo-600/90 text-xs mb-0.5">笔记关键词（CSV）</div>
                    <p className="text-xs leading-relaxed">{analysis.contentTopicKeywords}</p>
                  </div>
                )}
              </dl>
            </div>
          )}
          
          {/* 评论（全文） */}
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">本地评论（预测全文）</h3>
            {!analysis.contentMatched && (
              <p className="text-xs text-gray-500 mb-3">该笔记未在 content 主题表中命中，评论无 BERTopic 语境对照。</p>
            )}
            <div className="space-y-3">
              {analysis.topComments.map(comment => (
                <div key={comment.id} className="bg-gray-50 rounded-xl p-3 border border-gray-100">
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
                  {comment.noteTopicContext && (
                    <p className="text-xs text-indigo-800 bg-indigo-50/90 border border-indigo-100 rounded-lg px-2 py-1.5 mb-2 leading-snug">
                      <span className="text-indigo-600 font-medium">笔记主题 · </span>
                      {comment.noteTopicContext}
                    </p>
                  )}
                  <p className="text-sm text-gray-600">{comment.content}</p>
                  {comment.likes > 0 && (
                <div className="mt-1 text-xs text-gray-400">👍 {comment.likes}</div>
              )}
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
  classData,
  history,
  trainingHistoryMeta,
  evalViz,
  subView,
}: {
  config: RoBERTaConfig;
  metrics: EvaluationMetrics;
  classData: { name: string; precision: number; recall: number; f1: number; support: number }[];
  history: { epoch: number; trainLoss: number; valLoss: number; valAcc: number; valF1: number }[];
  trainingHistoryMeta: { checkpointDir: string; relativePath: string } | null;
  evalViz: EvalVizPayload | null;
  subView: ModelSubView;
}) {
  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain pr-0.5 -mr-0.5">
        {subView === 'overview' && (
          <ModelOverviewView config={config} metrics={metrics} classData={classData} evalViz={evalViz} />
        )}
        {subView === 'metrics' && <MetricsDetailView evalViz={evalViz} />}
        {subView === 'training' && (
          <TrainingHistoryView history={history} sourceMeta={trainingHistoryMeta} />
        )}
      </div>
    </div>
  );
}

// 模型概览视图（与 evaluate_model 中 03/04/05/06/08 图同源逻辑的交互式复现）
function ModelOverviewView({
  config,
  metrics,
  classData,
  evalViz,
}: {
  config: RoBERTaConfig;
  metrics: EvaluationMetrics;
  classData: { name: string; precision: number; recall: number; f1: number; support: number }[];
  evalViz: EvalVizPayload | null;
}) {
  const statCard =
    'rounded-2xl p-4 md:p-5 border shadow-sm transition-shadow hover:shadow-md';
  const fineTuneRows = [
    { label: '预训练权重', value: config.baseModel, mono: true },
    { label: 'max_length', value: String(config.maxLength) },
    { label: 'batch_size', value: String(config.batchSize) },
    { label: 'learning_rate', value: String(config.learningRate) },
    { label: 'epochs', value: String(config.epochs) },
  ];
  const splitRows = [
    { label: '训练集', value: config.trainSize, color: 'bg-blue-500' },
    { label: '验证集', value: config.valSize, color: 'bg-emerald-500' },
    { label: '测试集', value: config.testSize, color: 'bg-violet-500' },
  ];
  const splitTotal = Math.max(1, splitRows.reduce((sum, row) => sum + row.value, 0));

  return (
    <div className="space-y-6 pb-2">
      <div
        className={`grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 ${
          metrics.macroAuc > 0 ? 'lg:grid-cols-4' : 'lg:grid-cols-3'
        }`}
      >
        <div className={`${statCard} bg-gradient-to-br from-indigo-50 to-blue-50 border-indigo-200/70`}>
          <div className="flex items-center gap-2 mb-2 text-indigo-800">
            <Activity className="w-4 h-4 shrink-0" />
            <span className="text-sm font-semibold">测试准确率</span>
          </div>
          <div className="text-3xl font-bold tabular-nums text-indigo-900">
            {(metrics.accuracy * 100).toFixed(2)}
            <span className="text-lg font-semibold text-indigo-700/80">%</span>
          </div>
        </div>
        <div className={`${statCard} bg-gradient-to-br from-emerald-50 to-teal-50 border-emerald-200/70`}>
          <div className="flex items-center gap-2 mb-2 text-emerald-800">
            <BarChart3 className="w-4 h-4 shrink-0" />
            <span className="text-sm font-semibold">Macro F1</span>
          </div>
          <div className="text-3xl font-bold tabular-nums text-emerald-900">
            {(metrics.macroF1 * 100).toFixed(2)}
            <span className="text-lg font-semibold text-emerald-800/80">%</span>
          </div>
        </div>
        {metrics.macroAuc > 0 && (
          <div className={`${statCard} bg-gradient-to-br from-violet-50 to-purple-50 border-violet-200/70`}>
            <div className="flex items-center gap-2 mb-2 text-violet-800">
              <PieChart className="w-4 h-4 shrink-0" />
              <span className="text-sm font-semibold">Macro AUC</span>
            </div>
            <div className="text-3xl font-bold tabular-nums text-violet-900">
              {(metrics.macroAuc * 100).toFixed(2)}
              <span className="text-lg font-semibold text-violet-800/80">%</span>
            </div>
          </div>
        )}
        <div className={`${statCard} bg-gradient-to-br from-cyan-50 to-sky-50 border-cyan-200/70`}>
          <div className="flex items-center gap-2 mb-2 text-cyan-900">
            <Brain className="w-4 h-4 shrink-0" />
            <span className="text-sm font-semibold">情感类别</span>
          </div>
          <div className="text-3xl font-bold tabular-nums text-cyan-950">{config.numLabels} 类</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-2xl bg-gradient-to-br from-white to-indigo-50/35 p-5 border border-slate-200/80 shadow-sm">
          <div className="mb-4 flex items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
              <Settings className="w-4 h-4 text-indigo-600" />
              微调超参
            </h4>
            <span className="px-2.5 py-1 rounded-full text-[11px] font-medium text-indigo-700 bg-indigo-100/80 border border-indigo-200/70">
              MacBERT 微调
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {fineTuneRows.map((row) => (
              <div
                key={row.label}
                className="rounded-xl border border-slate-200/70 bg-white/85 px-3 py-2.5 shadow-[0_1px_0_rgba(15,23,42,0.03)]"
              >
                <p className="text-[11px] text-slate-500 mb-1">{row.label}</p>
                <p
                  className={`text-sm font-semibold text-slate-900 tabular-nums leading-snug ${
                    row.mono ? 'font-mono text-[12px] break-all' : ''
                  }`}
                >
                  {row.value}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-2xl bg-gradient-to-br from-white to-emerald-50/35 p-5 border border-slate-200/80 shadow-sm">
          <div className="mb-4 flex items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-emerald-600" />
              数据划分
            </h4>
            <span className="text-[11px] text-slate-500 tabular-nums">总计 {splitTotal.toLocaleString()} 条</span>
          </div>
          <div className="space-y-3">
            {splitRows.map((row) => {
              const ratio = row.value / splitTotal;
              return (
                <div key={row.label} className="rounded-xl border border-slate-200/70 bg-white/85 px-3 py-2.5">
                  <div className="flex items-center justify-between gap-3 mb-1.5">
                    <span className="text-sm font-medium text-slate-700">{row.label}</span>
                    <span className="text-sm font-semibold text-slate-900 tabular-nums">
                      {row.value.toLocaleString()} 条 · {(ratio * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                    <div className={`h-full ${row.color}`} style={{ width: `${Math.max(3, ratio * 100)}%` }} />
                  </div>
                </div>
              );
            })}
            <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50/90 px-3 py-2 text-[11px] text-slate-600 flex items-center justify-between gap-3">
              <span>产出目录</span>
              <span className="font-mono text-slate-800 break-all text-right">{config.trainingDate}</span>
            </div>
          </div>
        </div>
      </div>

      <MacroSentimentEvalCharts metrics={metrics} classData={classData} evalViz={evalViz} />
    </div>
  );
}

/** 「评估」子页：预测置信度分布（由 eval_viz_payload 驱动，与 evaluate_model 同源逻辑） */
function MetricsDetailView({ evalViz }: { evalViz: EvalVizPayload | null }) {
  const hasLiveViz =
    evalViz &&
    evalViz.true_labels.length > 0 &&
    evalViz.probs.length === evalViz.true_labels.length &&
    evalViz.pred_labels.length === evalViz.true_labels.length;

  const viz: EvalVizPayload = hasLiveViz ? evalViz : (evalVizDemoPayload as EvalVizPayload);

  return (
    <div className="space-y-6 pb-2">
      <div className="rounded-2xl border border-slate-200/80 bg-gradient-to-br from-slate-50/90 to-indigo-50/40 p-4 md:p-5">
        <h4 className="text-sm font-semibold text-slate-800 mb-2">本页说明</h4>
        <p className="text-xs text-slate-600 leading-relaxed">
          下图包含：按 softmax 最大概率分桶的「正确 vs 错误」直方图，以及 One-vs-Rest ROC（与 comment/results/06_roc_curves.png 同源算法）。用于观察校准与可分性。
          {!hasLiveViz && (
            <span className="text-amber-800">
              {' '}
              当前未检测到有效的 <span className="font-mono text-[10px]">eval_viz_payload.json</span>，展示内置演示数据；接入本地文件后与
              evaluate_model 导出一致。
            </span>
          )}
        </p>
      </div>
      <ConfidenceHistogramChart
        trueLabels={viz.true_labels}
        predLabels={viz.pred_labels}
        probs={viz.probs as number[][]}
      />
      <RocCurvesEcharts trueLabels={viz.true_labels} probs={viz.probs as number[][]} />
    </div>
  );
}

// 训练过程视图
function TrainingHistoryView({
  history,
  sourceMeta,
}: {
  history: { epoch: number; trainLoss: number; valLoss: number; valAcc: number; valF1: number }[];
  sourceMeta: { checkpointDir: string; relativePath: string } | null;
}) {
  if (!history.length) {
    return (
      <div className="rounded-2xl border border-amber-200/80 bg-amber-50/50 p-6 text-sm text-amber-950">
        <p className="font-medium text-amber-900 mb-1">暂无训练曲线数据</p>
        <p className="text-amber-900/85 leading-relaxed">
          开发服会从{' '}
          <code className="text-xs font-mono bg-white/60 px-1.5 py-0.5 rounded border border-amber-200/80">
            comment/checkpoint_temp/checkpoint-*/trainer_state.json
          </code>{' '}
          中自动选取<strong>步数编号最大</strong>且含该文件的目录解析曲线。若仍为空，请确认：已在本机完成 Hugging Face Trainer 训练；{' '}
          <code className="text-xs font-mono">checkpoint_temp</code> 与仓库内 <code className="text-xs font-mono">comment/</code>{' '}
          相对位置未改；并已重启 <code className="text-xs font-mono">npm run dev</code>。
        </p>
      </div>
    );
  }

  const last = history[history.length - 1]!;
  const bestF1 = history.reduce((a, b) => (b.valF1 > a.valF1 ? b : a), history[0]!);
  const bestValLoss = history.reduce((a, b) => (b.valLoss < a.valLoss ? b : a), history[0]!);

  return (
    <div className="space-y-6 pb-2">
      {sourceMeta && (
        <p className="text-[11px] text-slate-500">
          曲线由{' '}
          <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-700">
            comment/{sourceMeta.relativePath}
          </code>{' '}
          解析（当前选用 <span className="font-medium text-slate-700">{sourceMeta.checkpointDir}</span>，即 checkpoint 步数最大的目录）。
        </p>
      )}
      <div className="rounded-2xl border border-emerald-200/70 bg-emerald-50/35 p-4 md:p-5 text-[12px] text-slate-700 leading-relaxed">
        <p className="font-semibold text-slate-900 mb-2">如何从多轮微调里选「最好的」checkpoint</p>
        <ul className="list-disc space-y-1.5 pl-4">
          <li>
            仅看<strong>训练 Loss 下降</strong>不够：它只反映对训练集的拟合，验证集若变差说明<strong>过拟合</strong>，不宜作为上线依据。
          </li>
          <li>
            实务上常以<strong>验证集 Macro F1</strong>（或与任务一致的指标）取峰值对应 epoch；也可参考<strong>验证 Loss 最低</strong>，二者可能不在同一轮，需结合曲线形状取舍。
          </li>
          <li>
            图中<strong>绿色竖线</strong>标在验证 F1 最高的 epoch；<strong>玫瑰色竖线</strong>标在验证 Loss 最低的 epoch。若最后一轮不在峰值附近，宜保存峰值轮权重（类似 Trainer 的{' '}
            <code className="rounded bg-white/80 px-1 py-0.5 font-mono text-[10px]">load_best_model_at_end</code>
            ）。
          </li>
        </ul>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-2xl border border-slate-200/80 bg-gradient-to-br from-white to-indigo-50/40 p-4 shadow-sm">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">最后一轮</p>
          <p className="mt-1 text-xl font-bold tabular-nums text-slate-900">Epoch {last.epoch}</p>
          <div className="mt-2 space-y-0.5 text-xs text-slate-600">
            <p>Val Acc {(last.valAcc * 100).toFixed(2)}%</p>
            <p>Val F1 {(last.valF1 * 100).toFixed(2)}%</p>
            <p>Val Loss {last.valLoss.toFixed(4)}</p>
          </div>
        </div>
        <div className="rounded-2xl border border-slate-200/80 bg-gradient-to-br from-white to-emerald-50/40 p-4 shadow-sm">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">验证 F1 最高</p>
          <p className="mt-1 text-xl font-bold tabular-nums text-emerald-800">{(bestF1.valF1 * 100).toFixed(2)}%</p>
          <p className="mt-1 text-xs text-slate-500">Epoch {bestF1.epoch}</p>
        </div>
        <div className="rounded-2xl border border-slate-200/80 bg-gradient-to-br from-white to-rose-50/40 p-4 shadow-sm">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">验证 Loss 最低</p>
          <p className="mt-1 text-xl font-bold tabular-nums text-rose-800">{bestValLoss.valLoss.toFixed(4)}</p>
          <p className="mt-1 text-xs text-slate-500">Epoch {bestValLoss.epoch}</p>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200/80 bg-white p-4 md:p-5 shadow-sm">
        <h4 className="text-sm font-semibold text-slate-800 mb-1">Loss（训练 / 验证）</h4>
        <p className="text-[11px] text-slate-500 mb-3">
          横轴为 epoch；玫瑰色虚线为「验证 Loss 最低」所在轮次。
        </p>
        <div className="h-[280px] w-full min-w-0">
          <ResponsiveContainerAny width="100%" height="100%">
            <LineChartAny data={history} margin={{ top: 24, right: 12, left: 0, bottom: 0 }}>
              <CartesianGridAny strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxisAny dataKey="epoch" tick={{ fontSize: 10 }} />
              <YAxisAny tick={{ fontSize: 10 }} width={44} />
              <TooltipAny contentStyle={CHART_TOOLTIP_STYLE} />
              <LegendAny wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLineAny
                x={bestValLoss.epoch}
                stroke="#f43f5e"
                strokeDasharray="4 4"
                label={{ value: `Val Loss↓ ${bestValLoss.epoch}`, position: 'top', fill: '#be123c', fontSize: 10 }}
              />
              <LineAny type="monotone" dataKey="trainLoss" name="训练 Loss" stroke="#fb923c" strokeWidth={2} dot={false} />
              <LineAny type="monotone" dataKey="valLoss" name="验证 Loss" stroke="#e11d48" strokeWidth={2} dot={false} />
            </LineChartAny>
          </ResponsiveContainerAny>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200/80 bg-white p-4 md:p-5 shadow-sm">
        <h4 className="text-sm font-semibold text-slate-800 mb-1">验证准确率与 Macro F1</h4>
        <p className="text-[11px] text-slate-500 mb-3">绿色虚线为「验证 F1 最高」所在轮次。</p>
        <div className="h-[280px] w-full min-w-0">
          <ResponsiveContainerAny width="100%" height="100%">
            <LineChartAny data={history} margin={{ top: 24, right: 12, left: 0, bottom: 0 }}>
              <CartesianGridAny strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxisAny dataKey="epoch" tick={{ fontSize: 10 }} />
              <YAxisAny domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10 }} width={40} />
              <TooltipAny formatter={(v: number) => `${(Number(v) * 100).toFixed(2)}%`} contentStyle={CHART_TOOLTIP_STYLE} />
              <LegendAny wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLineAny
                x={bestF1.epoch}
                stroke="#10b981"
                strokeDasharray="4 4"
                label={{ value: `Val F1↑ ${bestF1.epoch}`, position: 'top', fill: '#047857', fontSize: 10 }}
              />
              <LineAny type="monotone" dataKey="valAcc" name="验证准确率" stroke="#6366f1" strokeWidth={2} dot={false} />
              <LineAny type="monotone" dataKey="valF1" name="验证 F1" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChartAny>
          </ResponsiveContainerAny>
        </div>
      </div>

      <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-[11px] text-slate-600 leading-relaxed">
        完整逐轮数值见本地{' '}
        <code className="rounded bg-white px-1 py-0.5 font-mono text-[10px]">checkpoint_temp/.../trainer_state.json</code>。
      </div>
    </div>
  );
}

