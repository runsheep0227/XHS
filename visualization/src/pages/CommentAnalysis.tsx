import { useState, useEffect, useMemo, useId } from 'react';
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
  Sparkles,
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
} from 'recharts';
import type { CommentTopic, CommentAnalysis, RoBERTaConfig, EvaluationMetrics } from '../data/commentData';
import { loadLiveCommentBundle, type LiveCommentBundle } from '../data/commentLiveData';
import { CONTENT_SERVER_BASE } from '../data/topicData';

/** Recharts 类组件与 @types/react 的 JSX 推断不兼容，与 InteractiveCharts.tsx 一致用 any 断言 */
const ResponsiveContainerAny = ResponsiveContainer as any;
const PieChartRecharts = RPie as any;
const PieAny = Pie as any;
const CellAny = Cell as any;
const TooltipAny = Tooltip as any;
const LegendAny = Legend as any;
const BarChartAny = BarChart as any;
const BarAny = Bar as any;
const XAxisAny = XAxis as any;
const YAxisAny = YAxis as any;
const CartesianGridAny = CartesianGrid as any;

/** 基于 MacBERT 的评论反馈（与笔记主题通过 note_id 对齐） */
type FeedbackView = 'overview' | 'sentiment' | 'notes';
type CommentModule = 'feedback' | 'model';
type ModelSubView = 'overview' | 'metrics' | 'training' | 'samples';

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
  { id: 'overview', label: '评论概览', short: '概览', icon: PieChart },
  { id: 'sentiment', label: '情感分布', short: '情感', icon: Heart },
  { id: 'notes', label: '笔记与主题', short: '笔记', icon: MessageCircle },
];

const MODEL_TABS: { id: ModelSubView; label: string; icon: LucideIcon }[] = [
  { id: 'overview', label: '概览', icon: PieChart },
  { id: 'metrics', label: '评估', icon: BarChart3 },
  { id: 'training', label: '训练', icon: Activity },
  { id: 'samples', label: '样例', icon: MessageSquare },
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
  const [feedbackView, setFeedbackView] = useState<FeedbackView>('overview');
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
      setFeedbackView('overview');
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

  const commentImgBase = CONTENT_SERVER_BASE.replace(/\/$/, '');

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
            <span className="text-sm text-gray-500">样本均长（字）</span>
          </div>
        </div>
      </header>

      <main className="flex h-[calc(100vh-180px)]">
        {/* 左侧评论主题列表 */}
        <aside className="w-72 bg-white/60 backdrop-blur-sm border-r border-blue-100 overflow-y-auto p-4">
          <h2 className="font-semibold text-gray-700 mb-4 flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-blue-500" />
            高评论笔记
          </h2>
          <div className="space-y-2">
            {filteredTopics.map((topic, index) => (
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
            ))}
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
                {commentModule === 'feedback' && feedbackView === 'overview' && bundle && (
                  <OverviewView bundle={bundle} selectedTopic={selectedTopic} />
                )}
                {commentModule === 'feedback' && feedbackView === 'sentiment' && bundle && (
                  <SentimentView bundle={bundle} />
                )}
                {commentModule === 'feedback' && feedbackView === 'notes' && (
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
                    confidenceData={[]}
                    history={bundle.trainingHistory}
                    samples={bundle.misclassifiedSamples}
                    confusionImageUrl={`${commentImgBase}/comment/results/01_confusion_matrix.png`}
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

/** 概览页 · 选中笔记：情感统计与评论列表（正文不在此重复展示） */
function OverviewSelectedNoteView({ topic }: { topic: CommentTopic }) {
  const lines = topic.noteComments ?? [];
  const sampleN = lines.length;
  const cPos = lines.filter((c) => c.sentiment === 'positive').length;
  const cNeu = lines.filter((c) => c.sentiment === 'neutral').length;
  const cNeg = lines.filter((c) => c.sentiment === 'negative').length;
  const denom = sampleN || 1;
  const pct = (x: number) => ((x / denom) * 100).toFixed(1);
  const useTopicRatios = sampleN === 0;
  const pPos = useTopicRatios ? topic.positiveRatio * 100 : Number(pct(cPos));
  const pNeu = useTopicRatios ? topic.neutralRatio * 100 : Number(pct(cNeu));
  const pNeg = useTopicRatios ? topic.negativeRatio * 100 : Number(pct(cNeg));

  const sentimentCards = [
    {
      key: 'pos',
      label: '积极',
      count: useTopicRatios ? null : cPos,
      pct: pPos,
      color: 'from-emerald-50 to-green-50',
      border: 'border-emerald-100/80',
      icon: Heart,
      iconBg: 'bg-emerald-500',
      desc: '样本内模型判为正向：感谢、认同、推荐等偏正面反馈。',
    },
    {
      key: 'neu',
      label: '中性',
      count: useTopicRatios ? null : cNeu,
      pct: pNeu,
      color: 'from-slate-50 to-slate-100/90',
      border: 'border-slate-200/80',
      icon: MessageCircle,
      iconBg: 'bg-slate-400',
      desc: '陈述、追问、补充信息等无明显褒贬倾向的语气。',
    },
    {
      key: 'neg',
      label: '消极',
      count: useTopicRatios ? null : cNeg,
      pct: pNeg,
      color: 'from-rose-50 to-red-50/80',
      border: 'border-rose-100/70',
      icon: ThumbsDown,
      iconBg: 'bg-rose-500',
      desc: '批评、抱怨、质疑或负面体验等与正向相反的表达。',
    },
  ] as const;

  return (
    <div className="h-full space-y-8">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-wide text-cyan-600/90 mb-1">评论概览 · 单篇</p>
          <h3 className="text-lg font-semibold tracking-tight text-slate-800">{topic.name}</h3>
        </div>
        {resolveNotePageHref(topic.noteUrl, topic.noteId) && (
          <a
            href={resolveNotePageHref(topic.noteUrl, topic.noteId)!}
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 inline-flex items-center gap-1 rounded-xl border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs font-semibold text-cyan-800 hover:bg-cyan-100/80"
          >
            查看原帖/评论区 →
          </a>
        )}
      </header>

      <section className="space-y-4">
        <div>
          <h4 className="text-sm font-semibold text-slate-800">本篇评论 · 情感统计与说明</h4>
          <p className="text-xs text-slate-500 mt-1">
            {sampleN > 0
              ? `基于样本内共 ${sampleN} 条评论的 MacBERT 预测标签。`
              : '当前无样本评论行，下列占比为话题聚合表中的比例（与左侧列表一致）。'}
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {sentimentCards.map((card) => {
            const Icon = card.icon;
            return (
              <div
                key={card.key}
                className={`rounded-2xl border ${card.border} bg-gradient-to-br ${card.color} p-4 shadow-sm`}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-10 h-10 ${card.iconBg} rounded-xl flex items-center justify-center shadow-sm`}>
                    <Icon className="w-5 h-5 text-white" aria-hidden />
                  </div>
                  <div>
                    <div className="text-2xl font-bold tabular-nums text-slate-800">{card.pct.toFixed(1)}%</div>
                    <div className="text-xs text-slate-600">
                      {card.label}
                      {card.count != null ? (
                        <span className="text-slate-500"> · {card.count} 条</span>
                      ) : null}
                    </div>
                  </div>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">{card.desc}</p>
              </div>
            );
          })}
        </div>
        <div className="rounded-2xl border border-slate-200/80 bg-slate-50/60 p-4">
          <p className="text-[11px] font-medium text-slate-500 mb-2">样本内情感结构</p>
          <div className="flex h-3 rounded-full overflow-hidden ring-1 ring-slate-200/80">
            <div className="h-full transition-all" style={{ width: `${pPos}%`, backgroundColor: SENT_COLORS.pos }} />
            <div className="h-full transition-all" style={{ width: `${pNeu}%`, backgroundColor: SENT_COLORS.neu }} />
            <div className="h-full transition-all" style={{ width: `${pNeg}%`, backgroundColor: SENT_COLORS.neg }} />
          </div>
          <div className="flex flex-wrap gap-4 mt-2 text-[11px] tabular-nums text-slate-600">
            <span style={{ color: SENT_COLORS.pos }}>积极 {pPos.toFixed(1)}%</span>
            <span style={{ color: SENT_COLORS.neu }}>中性 {pNeu.toFixed(1)}%</span>
            <span style={{ color: SENT_COLORS.neg }}>消极 {pNeg.toFixed(1)}%</span>
            <span className="text-slate-400">
              均值得分（0–1）<span className="text-slate-700 font-semibold ml-1">{topic.avgSentimentScore.toFixed(3)}</span>
            </span>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
        <h4 className="text-sm font-semibold text-slate-800 mb-1">评论全文（预测样本）</h4>
        <p className="text-xs text-slate-500 mb-4 leading-relaxed">
          每条附模型预测情感；完整语料以本地 CSV 为准。
        </p>
        <div className="space-y-3 max-h-[min(520px,55vh)] overflow-y-auto pr-1">
          {!lines.length ? (
            <p className="text-sm text-slate-400 text-center py-8">暂无样本评论</p>
          ) : (
            lines.map((c) => (
              <div key={c.id} className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 text-left">
                <div className="flex items-center justify-between gap-2 mb-2">
                  <span className="text-xs font-medium text-slate-800 truncate">{c.user}</span>
                  <span
                    className={`shrink-0 px-2 py-0.5 rounded-md text-[10px] font-semibold ${
                      c.sentiment === 'positive'
                        ? 'bg-emerald-100 text-emerald-800'
                        : c.sentiment === 'negative'
                          ? 'bg-rose-100 text-rose-800'
                          : 'bg-slate-200/90 text-slate-700'
                    }`}
                  >
                    {c.sentiment === 'positive' ? '积极' : c.sentiment === 'negative' ? '消极' : '中性'}
                  </span>
                </div>
                <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap break-words">{c.content}</p>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}

/** 全库极性（CSV）+ 预测样本词频条形图 + 词云；展示在「情感分布」页首部 */
function GlobalPolarityAndWordFreqCharts({ bundle }: { bundle: LiveCommentBundle }) {
  const gradId = useId().replace(/:/g, '');
  const barGradId = `bar-${gradId}`;
  const pieData = bundle.polarity
    ? [
        { name: '正向', value: bundle.polarity.positive, color: SENT_COLORS.pos },
        { name: '中性', value: bundle.polarity.neutral, color: SENT_COLORS.neu },
        { name: '负向', value: bundle.polarity.negative, color: SENT_COLORS.neg },
      ]
    : [];
  const totalPolarity = bundle.polarity?.total ?? 0;

  const barData = bundle.wordCloud.slice(0, 16).map((w) => ({
    word: w.text,
    n: w.weight,
  }));

  return (
    <div className="space-y-8 pb-2 border-b border-slate-100">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold tracking-tight text-slate-800">全库极性与样本词频</h3>
          <p className="text-xs text-slate-500 mt-1 max-w-xl">
            环形图为全量极性（prediction_stats_polarity.csv）；词频来自当前预测样本，已过滤 meaningless_word.txt
          </p>
        </div>
        {totalPolarity > 0 && (
          <div className="flex gap-3">
            {pieData.map((d) => (
              <div
                key={d.name}
                className="rounded-xl border border-slate-100 bg-slate-50/80 px-3 py-2 text-center min-w-[5.5rem]"
              >
                <div className="text-[10px] uppercase tracking-wider text-slate-500">{d.name}</div>
                <div className="text-sm font-semibold tabular-nums text-slate-800">
                  {((d.value / totalPolarity) * 100).toFixed(1)}%
                </div>
                <div className="text-[11px] text-slate-400 tabular-nums">{d.value.toLocaleString()} 条</div>
              </div>
            ))}
          </div>
        )}
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="rounded-2xl border border-slate-200/80 bg-gradient-to-b from-white to-slate-50/90 p-5 shadow-sm">
          <h4 className="text-sm font-semibold text-slate-700 mb-1">全库情感极性</h4>
          <p className="text-[11px] text-slate-500 mb-3">prediction_stats_polarity.csv · 共 {totalPolarity.toLocaleString()} 条</p>
          <div className="h-[300px] w-full">
            <ResponsiveContainerAny width="100%" height="100%">
              <PieChartRecharts>
                <PieAny
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius="42%"
                  outerRadius="72%"
                  paddingAngle={2}
                  stroke="#fff"
                  strokeWidth={2}
                >
                  {pieData.map((e, i) => (
                    <CellAny key={i} fill={e.color} />
                  ))}
                </PieAny>
                <TooltipAny
                  formatter={(v: number) => [v.toLocaleString(), '条']}
                  contentStyle={CHART_TOOLTIP_STYLE}
                />
                <LegendAny
                  verticalAlign="bottom"
                  formatter={(value) => <span className="text-xs text-slate-600">{value}</span>}
                />
              </PieChartRecharts>
            </ResponsiveContainerAny>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200/80 bg-gradient-to-br from-white to-cyan-50/30 p-5 shadow-sm">
          <h4 className="text-sm font-semibold text-slate-700 mb-1">样本高频词</h4>
          <p className="text-[11px] text-slate-500 mb-3">Top {barData.length} · predicted_comments 子集</p>
          <div className="h-[300px] w-full">
            <ResponsiveContainerAny width="100%" height="100%">
              <BarChartAny data={barData} layout="vertical" margin={{ left: 4, right: 16 }}>
                <CartesianGridAny strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                <XAxisAny type="number" tick={{ fontSize: 10, fill: '#64748b' }} />
                <YAxisAny type="category" dataKey="word" width={76} tick={{ fontSize: 10, fill: '#475569' }} />
                <TooltipAny
                  formatter={(v: number) => [v.toLocaleString(), '次']}
                  contentStyle={CHART_TOOLTIP_STYLE}
                />
                <defs>
                  <linearGradient id={barGradId} x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#22d3ee" />
                    <stop offset="100%" stopColor="#4f46e5" />
                  </linearGradient>
                </defs>
                <BarAny
                  dataKey="n"
                  fill={`url(#${barGradId})`}
                  radius={[0, 10, 10, 0]}
                  maxBarSize={24}
                />
              </BarChartAny>
            </ResponsiveContainerAny>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
        <h4 className="text-sm font-semibold text-slate-700 mb-1">词云（权重→字号）</h4>
        <p className="text-[11px] text-slate-500 mb-4">与上表同源，便于扫读</p>
        <div className="flex flex-wrap gap-2 justify-center sm:justify-start">
          {bundle.wordCloud.map((item, i) => (
            <span
              key={i}
              className="px-3 py-1.5 rounded-full transition-transform duration-200 hover:scale-[1.03] cursor-default border border-slate-100/80"
              style={{
                fontSize: `${Math.max(12, Math.min(22, 11 + item.weight / 24))}px`,
                backgroundColor: `rgba(14, 165, 233, ${0.06 + Math.min(0.28, item.weight / 140)})`,
                color: '#0c4a6e',
              }}
            >
              {item.text}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

/** 评论概览：仅当选中左侧笔记时展示单篇内容；未选中时为引导占位（全库图表见「情感分布」） */
function OverviewView({ selectedTopic }: { bundle: LiveCommentBundle; selectedTopic: CommentTopic | null }) {
  if (selectedTopic) {
    return <OverviewSelectedNoteView topic={selectedTopic} />;
  }

  return (
    <div className="h-full min-h-[280px] flex flex-col items-center justify-center text-center px-8 py-16 rounded-2xl border border-dashed border-slate-200 bg-slate-50/40">
      <MessageSquare className="w-12 h-12 text-slate-300 mb-4" aria-hidden />
      <p className="text-sm font-semibold text-slate-700 mb-2">请从左侧选择一篇高评论笔记</p>
      <p className="text-xs text-slate-500 max-w-md leading-relaxed">
        选择后将在此展示该笔记的情感统计与评论全文。全库情感极性、样本高频词与词云请切换到
        <span className="text-slate-700 font-medium"> 情感分布 </span>。
      </p>
    </div>
  );
}

// 情感分布组件
function SentimentView({ bundle }: { bundle: LiveCommentBundle }) {
  const m = bundle.metrics;
  const stackData = bundle.lengthBySentiment;
  const ipRows = bundle.ipDistribution;

  return (
    <div className="h-full space-y-8">
      <GlobalPolarityAndWordFreqCharts bundle={bundle} />
      <div>
        <h3 className="text-lg font-semibold tracking-tight text-slate-800">情感分布详情</h3>
        <p className="text-xs text-slate-500 mt-1">
          上为全库极性饼图与样本词频、词云；以下为全库比例卡片与样本维度拆解（均为 comment 管线产出）。
        </p>
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
            与 prediction_stats_polarity.csv 中 label=1 的比例一致。
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

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
          <h4 className="font-semibold text-slate-700 mb-2">评论字数 × 情感（样本）</h4>
          <p className="text-xs text-slate-500 mb-4">按字符长度分桶，堆叠展示当前样本内极性构成</p>
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

        <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
          <h4 className="font-semibold text-slate-700 mb-2">IP 归属 Top（样本）</h4>
          <div className="h-[300px]">
            <ResponsiveContainerAny width="100%" height="100%">
              <BarChartAny data={ipRows} layout="vertical" margin={{ left: 16 }}>
                <CartesianGridAny strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                <XAxisAny type="number" tick={{ fontSize: 11 }} />
                <YAxisAny type="category" dataKey="name" width={88} tick={{ fontSize: 10 }} />
                <TooltipAny contentStyle={CHART_TOOLTIP_STYLE} />
                <LegendAny wrapperStyle={{ fontSize: 12 }} />
                <BarAny dataKey="positive" stackId="ip" fill={SENT_COLORS.pos} name="正" />
                <BarAny dataKey="neutral" stackId="ip" fill={SENT_COLORS.neu} name="中" />
                <BarAny dataKey="negative" stackId="ip" fill={SENT_COLORS.neg} name="负" radius={[0, 6, 6, 0]} />
              </BarChartAny>
            </ResponsiveContainerAny>
          </div>
        </div>
      </div>

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

/** 笔记 × 主题 × 评论：合并原「笔记与主题」卡片式概览与「详细列表」点击详情，数据源一致（note_id 对齐） */
function NotesTopicsUnifiedView({
  analysis,
  topics,
  onNoteClick,
}: {
  analysis: CommentAnalysis[];
  topics: CommentTopic[];
  onNoteClick: (a: CommentAnalysis) => void;
}) {
  const topicByNoteId = useMemo(() => {
    const m = new Map<string, CommentTopic>();
    for (const t of topics) {
      if (t.noteId) m.set(t.noteId, t);
    }
    return m;
  }, [topics]);

  return (
    <div className="h-full overflow-auto">
      <div className="mb-6">
        <h3 className="text-lg font-semibold tracking-tight text-slate-800">笔记与主题（note_id 对齐）</h3>
        <p className="text-xs text-slate-500 mt-1 max-w-3xl leading-relaxed">
          每条对应一篇高评论笔记：主题侧与 content BERTopic 同 note_id；下列为评论侧情感结构与关键词。
          <span className="text-slate-600">点击卡片</span>可打开完整字段与 content 关联摘要（原「详细列表」行为）。
        </p>
      </div>
      {analysis.length === 0 ? (
        <p className="text-sm text-slate-500 py-10 text-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/50">
          当前筛选下暂无笔记，请调整顶部搜索或情感筛选。
        </p>
      ) : null}
      <div className="space-y-3">
        {analysis.map((item, i) => {
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
                  <span className="w-9 h-9 shrink-0 rounded-xl bg-gradient-to-br from-cyan-500/15 to-indigo-600/15 text-indigo-700 border border-indigo-100/80 flex items-center justify-center text-sm font-bold tabular-nums">
                    {i + 1}
                  </span>
                  <div className="min-w-0">
                    <span className="font-semibold text-slate-800 block truncate">{displayTitle}</span>
                    {macroLine && (
                      <span className="text-xs text-indigo-600 mt-0.5 block leading-snug truncate">
                        BERTopic 宏观 · {macroLine}
                        {microId != null ? ` · 微观#${microId}` : ''}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1 shrink-0">
                  <span className="text-sm text-slate-500 tabular-nums">
                    {(topic?.commentCount ?? item.commentCount).toLocaleString()} 条评论
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
        选中笔记后，<strong className="mx-0.5">评论概览</strong>可看本篇统计与评论全文；全库极性与词云在
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
            <h3 className="font-semibold text-gray-700 mb-2">关键词（评论样本挖掘）</h3>
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
          
          {/* 样本评论（全文） */}
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">本地评论（预测样本全文）</h3>
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
  confidenceData,
  history,
  samples,
  confusionImageUrl,
  subView,
}: {
  config: RoBERTaConfig;
  metrics: EvaluationMetrics;
  classData: { name: string; precision: number; recall: number; f1: number; support: number }[];
  confidenceData: { range: string; count: number; accuracy: number }[];
  history: { epoch: number; trainLoss: number; valLoss: number; valAcc: number; valF1: number }[];
  samples: { id: number; content: string; trueLabel: string; predLabel: string; confidence: number; correct: boolean }[];
  confusionImageUrl: string;
  subView: ModelSubView;
}) {
  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="rounded-2xl border border-indigo-200/60 bg-gradient-to-br from-white via-slate-50/90 to-cyan-50/40 shadow-sm ring-1 ring-indigo-900/[0.04] p-5 md:p-6 mb-5 shrink-0">
        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-600/10 text-indigo-800 px-3 py-1 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5" />
              MacBERT-large · 评论三分类微调
            </span>
            <span className="text-xs text-slate-500 tabular-nums">
              测试准确率 {(metrics.accuracy * 100).toFixed(2)}%
            </span>
          </div>
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Hugging Face 基座
            </p>
            <p className="mt-1 font-mono text-sm sm:text-base font-semibold text-slate-900 break-all">
              {config.baseModel}
            </p>
            <p className="mt-2 text-sm text-slate-600 leading-snug max-w-2xl">{config.modelName}</p>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain pr-0.5 -mr-0.5">
        {subView === 'overview' && <ModelOverviewView config={config} metrics={metrics} />}
        {subView === 'metrics' && (
          <MetricsDetailView
            metrics={metrics}
            classData={classData}
            confidenceData={confidenceData}
            confusionImageUrl={confusionImageUrl}
          />
        )}
        {subView === 'training' && <TrainingHistoryView history={history} />}
        {subView === 'samples' && <PredictionSamplesView samples={samples} />}
      </div>
    </div>
  );
}

// 模型概览视图
function ModelOverviewView({ config, metrics }: { config: RoBERTaConfig; metrics: EvaluationMetrics }) {
  const statCard =
    'rounded-2xl p-4 md:p-5 border shadow-sm transition-shadow hover:shadow-md';
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
        <div className="rounded-2xl bg-white p-5 border border-slate-200/80 shadow-sm">
          <h4 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <Settings className="w-4 h-4 text-indigo-600" />
            微调超参
          </h4>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between gap-4 py-2 border-b border-slate-100">
              <dt className="text-slate-500 shrink-0">预训练权重</dt>
              <dd className="font-mono text-xs sm:text-sm font-medium text-slate-900 text-right break-all">
                {config.baseModel}
              </dd>
            </div>
            <div className="flex justify-between gap-4 py-2 border-b border-slate-100">
              <dt className="text-slate-500">max_length</dt>
              <dd className="font-medium text-slate-900 tabular-nums">{config.maxLength}</dd>
            </div>
            <div className="flex justify-between gap-4 py-2 border-b border-slate-100">
              <dt className="text-slate-500">batch_size</dt>
              <dd className="font-medium text-slate-900 tabular-nums">{config.batchSize}</dd>
            </div>
            <div className="flex justify-between gap-4 py-2 border-b border-slate-100">
              <dt className="text-slate-500">learning_rate</dt>
              <dd className="font-medium text-slate-900 tabular-nums">{config.learningRate}</dd>
            </div>
            <div className="flex justify-between gap-4 py-2">
              <dt className="text-slate-500">epochs</dt>
              <dd className="font-medium text-slate-900 tabular-nums">{config.epochs}</dd>
            </div>
          </dl>
        </div>

        <div className="rounded-2xl bg-white p-5 border border-slate-200/80 shadow-sm">
          <h4 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-emerald-600" />
            数据划分
          </h4>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between gap-4 py-2 border-b border-slate-100">
              <dt className="text-slate-500">训练集</dt>
              <dd className="font-medium text-slate-900 tabular-nums">{config.trainSize.toLocaleString()} 条</dd>
            </div>
            <div className="flex justify-between gap-4 py-2 border-b border-slate-100">
              <dt className="text-slate-500">验证集</dt>
              <dd className="font-medium text-slate-900 tabular-nums">{config.valSize.toLocaleString()} 条</dd>
            </div>
            <div className="flex justify-between gap-4 py-2 border-b border-slate-100">
              <dt className="text-slate-500">测试集</dt>
              <dd className="font-medium text-slate-900 tabular-nums">{config.testSize.toLocaleString()} 条</dd>
            </div>
            <div className="flex justify-between gap-4 py-2">
              <dt className="text-slate-500">产出目录</dt>
              <dd className="font-mono text-xs text-slate-800 text-right break-all">{config.trainingDate}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="rounded-2xl bg-gradient-to-r from-indigo-50/90 via-white to-cyan-50/70 border border-indigo-100/80 p-5 shadow-sm">
        <p className="text-sm text-slate-700 leading-relaxed">
          <strong className="text-slate-900">模型说明：</strong>
          基座为 Hugging Face 仓库{' '}
          <code className="rounded-md bg-white/80 px-1.5 py-0.5 text-xs font-mono border border-slate-200/80">
            {config.baseModel}
          </code>
          （{config.modelName}），在本仓库评论语料上进行三分类微调，标签与极性 CSV 一致（−1 / 0 /
          1）。指标与图表来自{' '}
          <code className="rounded-md bg-white/80 px-1.5 py-0.5 text-xs font-mono border border-slate-200/80">
            evaluation_report.txt
          </code>
          及训练导出文件；当前测试集准确率{' '}
          <span className="font-semibold tabular-nums">{((metrics.accuracy * 100).toFixed(2))}%</span>。
        </p>
      </div>
    </div>
  );
}

// 评估指标详细视图
function MetricsDetailView({
  metrics: _metrics,
  classData,
  confidenceData,
  confusionImageUrl,
}: {
  metrics: EvaluationMetrics;
  classData: { name: string; precision: number; recall: number; f1: number; support: number }[];
  confidenceData: { range: string; count: number; accuracy: number }[];
  confusionImageUrl: string;
}) {
  const sentimentColors = ['#22c55e', '#a1a1aa', '#ef4444'];

  return (
    <div className="space-y-6 pb-2">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
          <h4 className="text-sm font-semibold text-slate-800 mb-4">混淆矩阵（测试集）</h4>
          <div className="rounded-xl bg-slate-50/80 border border-slate-100 p-3">
            <img
              src={confusionImageUrl}
              alt="混淆矩阵"
              className="w-full max-h-[360px] object-contain rounded-lg"
            />
          </div>
          <p className="text-xs text-slate-500 mt-3 leading-relaxed">
            图片来自 <span className="font-mono text-[11px]">comment/results/01_confusion_matrix.png</span>
          </p>
        </div>

        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
          <h4 className="text-sm font-semibold text-slate-800 mb-4">各类别评估指标</h4>
          <div className="space-y-3">
            {classData.map((cls, i) => (
              <div
                key={cls.name}
                className="p-4 rounded-xl border border-slate-100 shadow-sm"
                style={{ backgroundColor: `${sentimentColors[i]}0d` }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold" style={{ color: sentimentColors[i] }}>
                    {cls.name}
                  </span>
                  <span className="text-xs text-slate-500 tabular-nums">n = {cls.support}</span>
                </div>
                <div className="grid grid-cols-3 gap-3 text-center text-sm">
                  <div className="rounded-lg bg-white/70 py-2 border border-white/80">
                    <div className="text-slate-500 text-xs mb-0.5">精确率</div>
                    <div className="font-bold tabular-nums text-slate-900">{(cls.precision * 100).toFixed(1)}%</div>
                  </div>
                  <div className="rounded-lg bg-white/70 py-2 border border-white/80">
                    <div className="text-slate-500 text-xs mb-0.5">召回率</div>
                    <div className="font-bold tabular-nums text-slate-900">{(cls.recall * 100).toFixed(1)}%</div>
                  </div>
                  <div className="rounded-lg bg-white/70 py-2 border border-white/80">
                    <div className="text-slate-500 text-xs mb-0.5">F1</div>
                    <div className="font-bold tabular-nums text-slate-900">{(cls.f1 * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {confidenceData.length > 0 && (
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
          <h4 className="text-sm font-semibold text-slate-800 mb-4">置信度分布与准确率</h4>
          <div className="space-y-3">
            {confidenceData.map((item, i) => (
              <div key={item.range} className="flex items-center gap-3">
                <span className="w-16 text-xs text-gray-500">{item.range}</span>
                <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${(item.count / 500) * 100}%`,
                      backgroundColor: sentimentColors[i % 3],
                    }}
                  />
                </div>
                <span className="w-20 text-xs text-gray-600">{item.count} 条</span>
                <span
                  className="w-16 text-xs text-right"
                  style={{
                    color:
                      item.accuracy > 0.8
                        ? '#22c55e'
                        : item.accuracy > 0.5
                          ? '#f59e0b'
                          : '#ef4444',
                  }}
                >
                  准确率 {(item.accuracy * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// 训练过程视图
function TrainingHistoryView({ history }: { history: { epoch: number; trainLoss: number; valLoss: number; valAcc: number; valF1: number }[] }) {
  if (!history.length) {
    return (
      <div className="rounded-2xl border border-amber-200/80 bg-amber-50/50 p-6 text-sm text-amber-950">
        <p className="font-medium text-amber-900 mb-1">暂无训练曲线数据</p>
        <p className="text-amber-900/85 leading-relaxed">
          未读取到{' '}
          <code className="text-xs font-mono bg-white/60 px-1.5 py-0.5 rounded border border-amber-200/80">
            checkpoint_temp/checkpoint-8000/trainer_state.json
          </code>
          ，启动开发服务器并确认 comment 目录完整后可显示 Loss / 准确率变化。
        </p>
      </div>
    );
  }
  const maxLoss = Math.max(0.001, ...history.map(h => Math.max(h.trainLoss, h.valLoss)));
  
  return (
    <div className="space-y-6 pb-2">
      <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
        <h4 className="text-sm font-semibold text-slate-800 mb-4">训练过程指标变化</h4>
        
        {/* Loss曲线 */}
        <div className="mb-6">
          <h5 className="text-xs font-medium text-slate-500 mb-2">Loss 变化曲线</h5>
          <div className="flex items-end gap-2 h-32">
            {history.map((h, i) => (
              <div key={i} className="flex-1 flex flex-col items-center">
                <div className="w-full flex flex-col-reverse h-28">
                  <div 
                    className="bg-rose-400 rounded-t transition-all"
                    style={{ height: `${(h.valLoss / maxLoss) * 100}%` }}
                    title={`验证Loss: ${h.valLoss.toFixed(3)}`}
                  />
                </div>
                <span className="text-xs text-slate-500 mt-1">Epoch {h.epoch}</span>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-4 mt-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-1 bg-rose-400 rounded" />
              <span className="text-xs text-slate-500">验证 Loss</span>
            </div>
          </div>
        </div>

        <div>
          <h5 className="text-xs font-medium text-slate-500 mb-2">准确率与 F1 变化曲线</h5>
          <div className="flex items-end gap-2 h-32">
            {history.map((h, i) => (
              <div key={i} className="flex-1 flex flex-col items-center">
                <div className="w-full flex flex-col-reverse h-28 gap-1">
                  <div
                    className="bg-indigo-500 rounded-t transition-all"
                    style={{ height: `${h.valAcc * 100}%` }}
                    title={`验证准确率: ${(h.valAcc * 100).toFixed(1)}%`}
                  />
                  <div
                    className="bg-emerald-500 rounded-t transition-all"
                    style={{ height: `${h.valF1 * 100}%` }}
                    title={`验证F1: ${(h.valF1 * 100).toFixed(1)}%`}
                  />
                </div>
                <span className="text-xs text-slate-500 mt-1">Epoch {h.epoch}</span>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-4 mt-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-1 bg-indigo-500 rounded" />
              <span className="text-xs text-slate-500">验证准确率</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-1 bg-emerald-500 rounded" />
              <span className="text-xs text-slate-500">验证 F1</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm overflow-x-auto">
        <h4 className="text-sm font-semibold text-slate-800 mb-4">训练指标详情</h4>
        <table className="w-full text-sm min-w-[420px]">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50/80">
              <th className="text-left py-3 px-2 text-slate-500 font-medium rounded-tl-lg">Epoch</th>
              <th className="text-right py-3 px-2 text-slate-500 font-medium">训练 Loss</th>
              <th className="text-right py-3 px-2 text-slate-500 font-medium">验证 Loss</th>
              <th className="text-right py-3 px-2 text-slate-500 font-medium">验证准确率</th>
              <th className="text-right py-3 px-2 text-slate-500 font-medium rounded-tr-lg">验证 F1</th>
            </tr>
          </thead>
          <tbody>
            {history.map((h, i) => (
              <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/50">
                <td className="py-2.5 px-2 font-medium text-slate-800">{h.epoch}</td>
                <td className="py-2.5 px-2 text-right tabular-nums text-slate-600">{h.trainLoss.toFixed(4)}</td>
                <td className="py-2.5 px-2 text-right tabular-nums text-slate-600">{h.valLoss.toFixed(4)}</td>
                <td className="py-2.5 px-2 text-right text-indigo-600 font-semibold tabular-nums">
                  {(h.valAcc * 100).toFixed(2)}%
                </td>
                <td className="py-2.5 px-2 text-right text-emerald-600 font-semibold tabular-nums">
                  {(h.valF1 * 100).toFixed(2)}%
                </td>
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
    <div className="space-y-4 pb-2">
      <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
        <h4 className="text-sm font-semibold text-slate-800 mb-1">测试集误判样例（节选）</h4>
        <p className="text-xs text-slate-500 mb-4 leading-relaxed">
          来自{' '}
          <span className="font-mono text-[11px] text-slate-600">comment/results/misclassified_test.csv</span>
          ；均为预测与真实标签不一致的样本
        </p>
        <div className="space-y-3">
          {samples.map(sample => (
            <div
              key={sample.id}
              className="p-4 rounded-xl border border-amber-200/70 bg-gradient-to-br from-amber-50/90 to-orange-50/40 shadow-sm"
            >
              <div className="flex items-start justify-between gap-3 mb-3">
                <p className="text-sm text-slate-800 leading-relaxed flex-1">{sample.content}</p>
                <div
                  className="w-7 h-7 shrink-0 rounded-full flex items-center justify-center bg-amber-500 text-white text-xs font-bold shadow"
                  title="误判"
                >
                  !
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-slate-600">
                <span>
                  真实{' '}
                  <span className="font-semibold text-slate-900">{sample.trueLabel}</span>
                </span>
                <span>
                  预测{' '}
                  <span className="font-semibold text-rose-600">{sample.predLabel}</span>
                </span>
                <span className="tabular-nums">
                  置信度 <span className="font-semibold text-slate-900">{(sample.confidence * 100).toFixed(1)}%</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-2xl bg-gradient-to-r from-indigo-50/90 to-cyan-50/60 border border-indigo-100/80 p-5 shadow-sm">
        <h4 className="text-sm font-semibold text-indigo-900 mb-2">说明</h4>
        <p className="text-sm text-slate-700 leading-relaxed">
          完整指标与混淆矩阵请以「评估」子页与导出图为准；误判样例便于检查{' '}
          <span className="font-mono text-xs">hfl/chinese-macbert-large</span> 微调后在边界语气、讽刺与口语缩略上的局限。
        </p>
      </div>
    </div>
  );
}
