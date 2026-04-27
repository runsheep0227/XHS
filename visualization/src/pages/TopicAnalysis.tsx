import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  Search,
  BarChart3,
  X,
  Network,
  ArrowLeft,
  Loader2,
  GitCompare,
  MapPin,
  CalendarClock,
  BookOpen,
  Brain,
  Cloud,
  Layers,
  LayoutDashboard,
  LayoutGrid,
  NotebookText,
  Boxes,
  Percent,
  BadgeCheck,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Sankey,
  Legend,
} from 'recharts';
import {
  loadTopicCSV,
  aggregateTopics,
  buildIPDistribution,
  buildTemporalMacroSeries,
  buildWordCloudEntriesForTopic,
  formatTopicTimestamp,
  toNotes,
  Topic,
  TopicRecord,
} from '../data/topicData';
import { EmptyState } from '../components/common/States';
import { ModelQualityDashboard } from '../components/ModelQualityCharts';
import { formatNumber } from '../utils/responsive';
import { CompareView } from '../components/InteractiveCharts';
import { ChinaIPMapChart } from '../components/ChinaIPMapChart';
import { RegionTopicMixChart } from '../components/RegionTopicMixChart';
import { sameIpRegion } from '../utils/ipLocationMap';
import { macroTopicDisplayColor, MACRO_ANCHOR_ORDER } from '../theme/macroTopicColors';

const PieChartRC = PieChart as any;
const PieRC = Pie as any;
const CellRC = Cell as any;
const BarRC = Bar as any;
const BarChartRC = BarChart as any;
const CartesianGridRC = CartesianGrid as any;
const XAxisRC = XAxis as any;
const YAxisRC = YAxis as any;
const ResponsiveContainerRC = ResponsiveContainer as any;
const TooltipRC = Tooltip as any;
const SankeyRC = Sankey as any;
const LegendRC = Legend as any;

/**
 * 桑基中间列 T* 专用色：与五大宏观锚点（MACRO_ANCHOR_COLORS）及右侧回退色刻意区分，
 * 避免用户误以为 T* 条与某一宏观主题一一对应。
 */
const SANKEY_MICRO_TOPIC_COLORS = [
  '#475569',
  '#6366f1',
  '#0e7490',
  '#9d174d',
  '#713f12',
  '#1e40af',
  '#0f766e',
  '#831843',
  '#155e75',
  '#4c1d95',
  '#854d0e',
  '#334155',
];

function isNoiseMacroLabel(name: string): boolean {
  const n = name.trim().toLowerCase();
  return n.includes('噪声') || n.includes('outlier');
}

function isNoiseTopicRecord(r: TopicRecord): boolean {
  if (r.is_noise === true) return true;
  return isNoiseMacroLabel(r.macro_topic || '');
}

/** 桑基聚合用宏观名：trim，避免「同主题」因首尾空格拆成两列、连线配色错位 */
function normalizedMacroForSankey(r: TopicRecord): string {
  if (isNoiseTopicRecord(r)) return '噪声数据';
  return (r.macro_topic ?? '').trim() || '未分类';
}

/** 桑基图 micro↔macro 复合键分隔符（避免 macro 名含 __ 导致 split 错位） */
const SANKEY_PART_SEP = '\u001f';

function microNodeLabel(id: number): string {
  return `T${id}`;
}

/** 右侧宏观主题节点：与环形图图例 / macroTopicDisplayColor 一致 */
function sankeyMacroNodeFill(name: string): string {
  return macroTopicDisplayColor(name);
}

/** 中间列 T*：按微观 id 轮换（色谱与宏观锚点无关） */
function sankeyMicroNodeFill(displayName: string): string {
  const m = /^T(-?\d+)$/.exec(displayName.trim());
  if (!m) return '#64748b';
  const id = parseInt(m[1], 10);
  return SANKEY_MICRO_TOPIC_COLORS[Math.abs(id) % SANKEY_MICRO_TOPIC_COLORS.length];
}

/** 右侧宏观列顺序：先五大锚点固定序，再其余名称，噪声类置底 */
function sankeyOrderedMacroNames(macroSet: Set<string>): string[] {
  const anchorSlice = MACRO_ANCHOR_ORDER.filter((n) => macroSet.has(n));
  const anchorNames = new Set<string>(anchorSlice);
  const rest = [...macroSet]
    .filter((n) => !anchorNames.has(n))
    .sort((a, b) => {
      if (a === '噪声数据') return 1;
      if (b === '噪声数据') return -1;
      return a.localeCompare(b, 'zh-CN');
    });
  return [...anchorSlice, ...rest];
}

/** 某微观主题的主要汇入宏观（篇数最多；并列时取宏观列中更靠上的项） */
function sankeyPrimaryMacroIndex(
  microId: number,
  microToMacro: Map<string, number>,
  macroList: string[],
): number {
  let bestIdx = 0;
  let bestVal = -1;
  for (let idx = 0; idx < macroList.length; idx++) {
    const key = `${microId}${SANKEY_PART_SEP}${macroList[idx]}`;
    const v = microToMacro.get(key) || 0;
    if (v > bestVal || (v === bestVal && idx < bestIdx)) {
      bestVal = v;
      bestIdx = idx;
    }
  }
  return bestIdx;
}

/** 将常见 hex 规范为 6 位，便于算对比色 */
function normalizeHex6(fillHex: string): string | null {
  let h = fillHex.replace('#', '').trim();
  if (h.length === 3) {
    h = h
      .split('')
      .map((c) => c + c)
      .join('');
  }
  return h.length === 6 ? h : null;
}

/** 根据节点底色选高对比文字色（右侧宏观条上的主题名更清晰） */
function contrastingTextOnFill(fillHex: string): string {
  const h = normalizeHex6(fillHex);
  if (!h) return '#0f172a';
  const r = parseInt(h.slice(0, 2), 16) / 255;
  const g = parseInt(h.slice(2, 4), 16) / 255;
  const b = parseInt(h.slice(4, 6), 16) / 255;
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return lum > 0.62 ? '#0f172a' : '#fafaf9';
}

function sankeyNodeColors(name: string, depth: number): { fill: string; textFill: string } {
  // 节点文本绘制在矩形外侧白底区域，因此需用深色文字保证可读性
  if (depth === 0) return { fill: '#64748b', textFill: '#334155' };
  if (depth === 1) return { fill: sankeyMicroNodeFill(name), textFill: '#ffffff' };
  const fill = sankeyMacroNodeFill(name);
  return { fill, textFill: '#334155' };
}

/** 指向 T* 的连线保持中性灰；指向右侧宏观主题的连线与锚点色一致 */
const SANKEY_LINK_GRAY = '#94a3b8';

function sankeyLinkTargetDisplayName(payload?: {
  target?: { name?: string };
  source?: { name?: string; depth?: number };
}): string {
  const tDirect = payload?.target;
  const tNested = (payload as { payload?: { target?: { name?: string } } } | undefined)?.payload?.target;
  const t = tDirect ?? tNested;
  if (t && typeof t === 'object' && t.name != null && String(t.name).trim() !== '') {
    return String(t.name).trim();
  }
  return '';
}

function SankeyThemedLink(props: {
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourceControlX: number;
  targetControlX: number;
  linkWidth: number;
  payload?: { target?: { name?: string; depth?: number }; source?: { name?: string; depth?: number } };
}) {
  const {
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourceControlX,
    targetControlX,
    linkWidth,
    payload,
  } = props;
  const tn = sankeyLinkTargetDisplayName(payload);
  const targetDepth =
    payload?.target?.depth ??
    (payload as { payload?: { target?: { depth?: number } } } | undefined)?.payload?.target?.depth;
  const toMacroStage = typeof targetDepth === 'number' ? targetDepth >= 2 : !/^T-?\d+$/.test(tn);
  const color = toMacroStage && tn ? sankeyMacroNodeFill(tn) : SANKEY_LINK_GRAY;
  const opacity = toMacroStage && tn ? 0.52 : 0.42;
  /** 略压细流量带，避免大屏下 ribbon 过粗占满视觉 */
  const w = Math.max(1.2, linkWidth * 0.68);
  const d = `M${sourceX},${sourceY} C${sourceControlX},${sourceY} ${targetControlX},${targetY} ${targetX},${targetY}`;
  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeOpacity={opacity}
      strokeWidth={w}
      strokeLinecap="round"
      className="recharts-sankey-link"
    />
  );
}

/** Recharts Sankey 默认节点无文字；需在自定义节点里画 rect + text */
function SankeyNodeWithLabel(props: {
  x: number;
  y: number;
  width: number;
  height: number;
  payload?: { name?: string; value?: number; depth?: number };
}) {
  const { x, y, width, height, payload } = props;
  const depth = payload?.depth ?? 0;
  const raw = String(payload?.name ?? '');
  const maxLen = depth === 1 ? 10 : 14;
  const short = raw.length > maxLen ? `${raw.slice(0, maxLen - 1)}…` : raw;
  const minLabelHeight = depth === 1 ? 12 : depth >= 2 ? 14 : 10;
  const showText = height >= minLabelHeight && short.length > 0;
  const { fill, textFill } = sankeyNodeColors(raw, depth);
  let tx = x + width + 5;
  let anchor: 'start' | 'end' = 'start';
  if (depth === 0) {
    tx = x - 5;
    anchor = 'end';
  } else if (depth >= 2) {
    tx = x + width + 6;
    anchor = 'start';
  } else {
    tx = x + width + 4;
    anchor = 'start';
  }
  const ty = y + height / 2;
  return (
    <g className="recharts-sankey-node-with-label">
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={3}
        ry={3}
        fill={fill}
        stroke="#ffffff"
        strokeWidth={1.5}
      />
      {showText ? (
        <text
          x={tx}
          y={ty}
          textAnchor={anchor}
          dominantBaseline="middle"
          fill={textFill}
          fontSize={depth === 1 ? 9 : depth >= 2 ? 12 : 11}
          fontWeight={depth >= 2 ? 700 : depth === 0 ? 700 : 600}
          style={
            depth === 1
              ? { pointerEvents: 'none', paintOrder: 'stroke', stroke: 'rgba(15,23,42,0.28)', strokeWidth: 2 }
              : depth >= 2 || depth === 0
                ? { pointerEvents: 'none', paintOrder: 'stroke', stroke: 'rgba(255,255,255,0.92)', strokeWidth: 2 }
                : { pointerEvents: 'none' }
          }
        >
          {short}
        </text>
      ) : null}
    </g>
  );
}

/** 外环：虚线引向环外，展示主题名 + 该主题平均置信度 + 篇数占比 */
function MacroOutsideLeaderLabel(props: {
  cx: number;
  cy: number;
  midAngle: number;
  innerRadius: number;
  outerRadius: number;
  name: string;
  value: number;
  percent: number;
  avgConfidence?: number;
  payload?: { avgConfidence?: number };
}) {
  const { cx, cy, midAngle, outerRadius, name, value, percent } = props;
  if (percent < 0.03) return null;
  const avgConf =
    typeof props.avgConfidence === 'number'
      ? props.avgConfidence
      : typeof props.payload?.avgConfidence === 'number'
        ? props.payload.avgConfidence
        : 0;
  const RAD = Math.PI / 180;
  const angle = -midAngle * RAD;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const x0 = cx + outerRadius * cos;
  const y0 = cy + outerRadius * sin;
  const r1 = outerRadius + 8;
  const x1 = cx + r1 * cos;
  const y1 = cy + r1 * sin;
  const tangent = 24;
  const x2 = x1 + (cos >= 0 ? tangent : -tangent);
  const y2 = y1;
  const textAnchor = cos >= 0 ? 'start' : 'end';
  const tx = x2 + (cos >= 0 ? 2 : -2);
  const shortName = name.length > 9 ? `${name.slice(0, 9)}…` : name;
  return (
    <g className="recharts-pie-outside-label" pointerEvents="none">
      <polyline
        points={`${x0},${y0} ${x1},${y1} ${x2},${y2}`}
        fill="none"
        stroke="#94a3b8"
        strokeWidth={1}
        strokeDasharray="2 4"
      />
      <text x={tx} y={y2} textAnchor={textAnchor} dominantBaseline="middle" fill="#0f172a" fontSize={10} fontWeight={700}>
        <tspan x={tx} dy="-0.55em">
          {shortName}
        </tspan>
        <tspan x={tx} dy="1.1em" fill="#475569" fontSize={9} fontWeight={600}>
          平均置信 {(avgConf * 100).toFixed(0)}% · {formatNumber(value)}篇 · {(percent * 100).toFixed(1)}%
        </tspan>
      </text>
    </g>
  );
}

/** 内环：与外环同构，虚线从内环外缘径向引出至两环间隙再折线，文字一律深灰（不用白字） */
function ConfidenceBetweenRingsLeaderLabel(props: {
  cx: number;
  cy: number;
  midAngle: number;
  innerRadius: number;
  outerRadius: number;
  name: string;
  value: number;
  percent: number;
}) {
  const { cx, cy, midAngle, outerRadius, name, value, percent } = props;
  if (percent < 0.02 || name === '—') return null;
  const RAD = Math.PI / 180;
  const angle = -midAngle * RAD;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const x0 = cx + outerRadius * cos;
  const y0 = cy + outerRadius * sin;
  const r1 = outerRadius + 7;
  const x1 = cx + r1 * cos;
  const y1 = cy + r1 * sin;
  const tangent = 22;
  const x2 = x1 + (cos >= 0 ? tangent : -tangent);
  const y2 = y1;
  const textAnchor = cos >= 0 ? 'start' : 'end';
  const tx = x2 + (cos >= 0 ? 2 : -2);
  const short = name.length > 6 ? `${name.slice(0, 6)}…` : name;
  return (
    <g className="recharts-pie-confidence-outside-label" pointerEvents="none">
      <polyline
        points={`${x0},${y0} ${x1},${y1} ${x2},${y2}`}
        fill="none"
        stroke="#64748b"
        strokeWidth={1}
        strokeDasharray="2 4"
      />
      <text x={tx} y={y2} textAnchor={textAnchor} dominantBaseline="middle" fill="#0f172a" fontSize={9} fontWeight={700}>
        <tspan x={tx} dy="-0.5em">
          {short}
        </tspan>
        <tspan x={tx} dy="1.05em" fill="#475569" fontSize={8.5} fontWeight={600}>
          {formatNumber(value)}篇 · {(percent * 100).toFixed(1)}%
        </tspan>
      </text>
    </g>
  );
}

type ViewMode = 'bertopic' | 'confidence' | 'geo' | 'timeline' | 'compare';

type MainModuleId = 'notes' | 'model';

function mainModuleOf(mode: ViewMode): MainModuleId {
  return mode === 'confidence' ? 'model' : 'notes';
}

/** 笔记级内容搜索（不区分大小写）：标题、正文、分词、标签、关键词与宏观主题名 */
function recordMatchesNoteSearch(r: TopicRecord, qRaw: string): boolean {
  const q = qRaw.trim().toLowerCase();
  if (!q) return true;
  const blob = [
    r.title,
    r.desc,
    r.content,
    r.segmented_text,
    r.tag_list,
    r.micro_topic_keywords,
    r.keywords,
    r.macro_topic,
  ]
    .filter(Boolean)
    .join('\n')
    .toLowerCase();
  return blob.includes(q);
}

function excerptAroundQuery(text: string | undefined, q: string, radius = 56): string {
  const t = (text || '').trim();
  if (!t) return '';
  const lower = t.toLowerCase();
  const idx = lower.indexOf(q.trim().toLowerCase());
  if (idx < 0) return t.length > radius * 2 ? `${t.slice(0, radius * 2)}…` : t;
  const start = Math.max(0, idx - radius);
  const end = Math.min(t.length, idx + q.trim().length + radius);
  return `${start > 0 ? '…' : ''}${t.slice(start, end)}${end < t.length ? '…' : ''}`;
}

export default function TopicAnalysis() {
  const [viewMode, setViewMode] = useState<ViewMode>('bertopic');
  const [topics, setTopics] = useState<Topic[]>([]);
  const [allRecords, setAllRecords] = useState<TopicRecord[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'noteCount' | 'avgLikes' | 'avgComments' | 'avgCollects'>('noteCount');
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  // ─── 数据加载：loadTopicCSV 内已合并全部 rawdata JSON ───
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setLoadError(null);
      try {
        const { records: topicRecords, meta } = await loadTopicCSV();

        if (topicRecords.length === 0) {
          if (meta) {
            console.warn('[TopicAnalysis] 主题表无有效行', meta);
          }
          setLoadError(
            '未能加载到有效的主题聚类数据。请运行开发服务器并确认 BERTopic 结果与原始笔记数据已就绪（详见控制台）。',
          );
          setAllRecords([]);
          setTopics([]);
          return;
        }

        setAllRecords(topicRecords);
        setTopics(aggregateTopics(topicRecords));
      } catch (e) {
        console.error('[TopicAnalysis] 数据加载失败:', e);
        setLoadError('数据加载失败，请检查控制台错误信息。');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  /** 按顶部排序选项排序后的全部主题（侧边栏与各视图共用同一列表） */
  const sortedTopics = useMemo(() => {
    return [...topics].sort((a, b) => {
      if (sortBy === 'avgLikes') return b.avgLikes - a.avgLikes;
      if (sortBy === 'avgComments') return b.avgComments - a.avgComments;
      if (sortBy === 'avgCollects') return b.avgCollects - a.avgCollects;
      return b.noteCount - a.noteCount;
    });
  }, [topics, sortBy]);

  const isNoteSearchActive = searchQuery.trim().length > 0;

  /** 在当前全部主题范围内，按正文/标题等匹配的笔记列表 */
  const noteSearchMatches = useMemo(() => {
    const q = searchQuery.trim();
    if (!q) return [];
    const allowedMacros = new Set(sortedTopics.map((t) => t.name));
    return allRecords
      .filter((r) => allowedMacros.has(r.macro_topic) && recordMatchesNoteSearch(r, q))
      .sort((a, b) => (b.liked_count || 0) - (a.liked_count || 0));
  }, [allRecords, sortedTopics, searchQuery]);

  /** 有命中笔记的主题（按命中篇数排序）；无搜索时与 sortedTopics 相同 */
  const filteredTopics = useMemo(() => {
    if (!isNoteSearchActive) return sortedTopics;
    const hitByMacro = new Map<string, number>();
    for (const r of noteSearchMatches) {
      hitByMacro.set(r.macro_topic, (hitByMacro.get(r.macro_topic) || 0) + 1);
    }
    return sortedTopics
      .filter((t) => hitByMacro.has(t.name))
      .sort((a, b) => (hitByMacro.get(b.name) || 0) - (hitByMacro.get(a.name) || 0));
  }, [sortedTopics, isNoteSearchActive, noteSearchMatches]);

  const topicByName = useMemo(() => new Map(topics.map((t) => [t.name, t])), [topics]);

  const hitCountByTopicName = useMemo(() => {
    const m = new Map<string, number>();
    if (!isNoteSearchActive) return m;
    for (const r of noteSearchMatches) {
      m.set(r.macro_topic, (m.get(r.macro_topic) || 0) + 1);
    }
    return m;
  }, [isNoteSearchActive, noteSearchMatches]);

  const topicListScrollRef = useRef<HTMLDivElement>(null);
  const topicSidebarVirtualizer = useVirtualizer({
    count: filteredTopics.length,
    getScrollElement: () => topicListScrollRef.current,
    estimateSize: () => 96,
    overscan: 8,
  });

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

  // ─── 派生数据（useMemo 必须在所有 return 之前调用）───
  const ipDistData = useMemo(() => buildIPDistribution(allRecords), [allRecords]);

  /** 两大模块：笔记数据（地域、对比、主题结构） / 模型效果（置信度诊断） */
  const mainModules = [
    {
      id: 'notes' as const,
      label: '笔记数据',
      hint: '细分主题、对比、地域与时序',
      icon: BookOpen,
      defaultTab: 'bertopic' as const,
      tabs: [
        { id: 'bertopic' as const, label: '细分主题', icon: Network },
        { id: 'compare' as const, label: '主题对比', icon: GitCompare },
        { id: 'geo' as const, label: '地理分布', icon: MapPin },
        { id: 'timeline' as const, label: '时序演变', icon: CalendarClock },
      ],
    },
    {
      id: 'model' as const,
      label: '模型效果',
      hint: '训练侧簇统计与样本推断诊断',
      icon: Brain,
      defaultTab: 'confidence' as const,
      tabs: [{ id: 'confidence' as const, label: '推断诊断', icon: Layers }],
    },
  ] as const;

  const activeMain = mainModuleOf(viewMode);
  const activeMainConfig = mainModules.find((m) => m.id === activeMain)!;

  const switchMain = (id: MainModuleId) => {
    if (id === activeMain) return;
    const cfg = mainModules.find((m) => m.id === id)!;
    setViewMode(cfg.defaultTab);
  };

  // ─── 加载状态 ───
  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
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
      <div className="min-h-screen flex items-center justify-center p-6">
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

  return (
    <div className="min-h-screen pb-10 md:pb-14">
      {/* ─── 顶部导航（与评论分析页一致：min-h-screen + sticky，便于整体微微拖动） ─── */}
      <header className="bg-white/80 backdrop-blur-md border-b border-rose-100 px-3 sm:px-4 lg:px-6 py-3 sm:py-4 sticky top-[65px] z-30">
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
                基于 BERTopic 模型进行主题聚类
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
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" aria-hidden />
              <span id="topic-search-hint" className="sr-only">
                在所有宏观主题范围内搜索笔记标题、正文、分词文本、标签与模型关键词；结果列出笔记并标明所属宏观主题。清空搜索框可返回图表视图。
              </span>
              <input
                id="topic-analysis-search"
                type="search"
                placeholder="搜索笔记标题或正文…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                title="在全部主题的笔记中检索：标题、正文、分词、标签、微观/宏观关键词。不区分大小写。左侧列表仅保留「至少含一篇命中笔记」的宏观主题，并显示命中篇数。清空搜索可恢复可视化。"
                aria-describedby="topic-search-hint"
                autoComplete="off"
                className="w-full sm:w-48 lg:w-72 pl-9 pr-3 py-2 bg-gray-50 border border-gray-200 rounded-lg sm:rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-rose-300 transition-all"
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

      {/* ─── 主内容：视口内高度与评论分析 h-[calc(100vh-180px)] 同策略；中间卡片仍单独滚动 ─── */}
      <main className="flex h-[calc(100vh-180px)] flex-col lg:flex-row min-h-0">
        {/* 左侧主题列表 */}
        <aside
          className={`
            lg:w-64 xl:w-72 lg:shrink-0 bg-white/60 backdrop-blur-sm border-r border-rose-100
            flex flex-col min-h-0 overflow-hidden p-3 sm:p-4
            ${isMobileSidebarOpen ? 'block' : 'hidden'}
            lg:block
            fixed lg:relative lg:self-stretch top-44 lg:top-auto left-0 right-0 z-30 lg:z-auto
            max-h-[50vh] lg:max-h-none
            shadow-xl lg:shadow-none
          `}
        >
          <div className="flex items-center justify-between lg:hidden mb-3 shrink-0">
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
              title={isNoteSearchActive ? '没有命中的笔记' : '未找到匹配主题'}
              description={
                isNoteSearchActive ? '请尝试其他关键词' : '当前无可用主题数据'
              }
            />
          ) : (
            <>
              {isNoteSearchActive && (
                <p className="text-[11px] text-rose-600/90 font-medium px-1 mb-1 shrink-0">
                  以下为含命中笔记的主题
                </p>
              )}
              <div
                ref={topicListScrollRef}
                className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden -mx-0.5 px-0.5"
              >
                <div
                  className="relative w-full"
                  style={{ height: topicSidebarVirtualizer.getTotalSize() }}
                >
                  {topicSidebarVirtualizer.getVirtualItems().map((vRow) => {
                    const index = vRow.index;
                    const topic = filteredTopics[index];
                    const hits = hitCountByTopicName.get(topic.name);
                    const isSelected = selectedTopic?.id === topic.id;
                    const accent = macroTopicDisplayColor(topic.name);
                    const cardStyle = isSelected
                      ? {
                          background: `linear-gradient(135deg, ${accent}26 0%, ${accent}14 100%)`,
                          borderColor: `${accent}66`,
                        }
                      : {
                          background: `linear-gradient(135deg, ${accent}12 0%, #ffffff 62%)`,
                          borderColor: `${accent}2b`,
                        };
                    return (
                      <div
                        key={topic.id}
                        className="absolute left-0 top-0 w-full pb-2"
                        style={{ transform: `translateY(${vRow.start}px)` }}
                      >
                        <div
                          onClick={() => {
                            setSelectedTopic(topic);
                            setIsMobileSidebarOpen(false);
                          }}
                          className={`group relative overflow-hidden border p-2.5 sm:p-3 rounded-xl cursor-pointer transition-all duration-200 hover:shadow-md hover:-translate-y-[1px] ${
                            isSelected ? 'text-gray-900 shadow-md' : 'text-gray-800'
                          }`}
                          style={cardStyle}
                        >
                          <span
                            aria-hidden
                            className="absolute left-0 top-0 h-full w-1.5 rounded-l-xl"
                            style={{ backgroundColor: accent, opacity: isSelected ? 0.95 : 0.72 }}
                          />
                          <div className="flex items-center justify-between mb-1 gap-2">
                            <span className="font-medium text-sm truncate pl-2.5">{topic.name}</span>
                            <span
                              className={`text-xs shrink-0 ${isSelected ? 'text-gray-600' : 'text-gray-400'}`}
                            >
                              #{index + 1}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-xs flex-wrap pl-2.5">
                            {hits != null && (
                              <span
                                className={
                                  isSelected
                                    ? 'text-amber-800 font-medium bg-amber-50/80 px-1.5 py-0.5 rounded-md'
                                    : 'text-amber-700 font-medium bg-amber-50 px-1.5 py-0.5 rounded-md'
                                }
                              >
                                命中 {hits} 篇
                              </span>
                            )}
                            <span className={isSelected ? 'text-gray-700' : 'text-gray-500'}>
                              共 {formatNumber(topic.noteCount)} 篇
                            </span>
                            <span style={{ color: isSelected ? '#475569' : accent }}>
                              ❤️ {formatNumber(topic.avgLikes)}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </aside>

        {/* 中间主可视化：工具条固定，仅 white 卡片区域滚动 */}
        <section className="flex-1 flex flex-col min-w-0 min-h-0 p-3 sm:p-4 lg:p-6 overflow-hidden">
          <div className="mb-4 sm:mb-6 space-y-3 shrink-0">
            {isNoteSearchActive && (
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 rounded-2xl border border-amber-200/80 bg-amber-50/80 px-4 py-3 text-sm text-amber-950">
                <span>
                  正在按<strong className="mx-1">笔记内容</strong>搜索，共{' '}
                  <strong>{formatNumber(noteSearchMatches.length)}</strong> 条命中（在全库主题范围内）
                </span>
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="shrink-0 inline-flex items-center justify-center px-3 py-1.5 rounded-lg bg-white border border-amber-200 text-amber-900 text-xs font-semibold hover:bg-amber-100/80 transition-colors"
                >
                  清空搜索，返回图表
                </button>
              </div>
            )}
            {/* 一级：模块 */}
            <div
              className={`flex flex-wrap gap-2 p-1.5 rounded-2xl bg-white/70 border border-rose-100/80 shadow-sm backdrop-blur-sm ${
                isNoteSearchActive ? 'opacity-40 pointer-events-none select-none' : ''
              }`}
            >
              {mainModules.map((mod) => {
                const ModIcon = mod.icon;
                const isOn = mod.id === activeMain;
                return (
                  <button
                    key={mod.id}
                    type="button"
                    onClick={() => switchMain(mod.id)}
                    className={`flex-1 min-w-[140px] sm:min-w-[200px] inline-flex items-center gap-2.5 px-3 sm:px-4 py-2.5 rounded-xl text-left transition-all border ${
                      isOn
                        ? 'bg-gradient-to-r from-rose-500 to-pink-500 text-white border-transparent shadow-md shadow-rose-200/50'
                        : 'bg-transparent text-gray-700 border-transparent hover:bg-rose-50/80'
                    }`}
                  >
                    <ModIcon className={`w-5 h-5 shrink-0 ${isOn ? 'text-white' : 'text-rose-500'}`} aria-hidden />
                    <span className="min-w-0">
                      <span className={`block text-sm font-semibold leading-tight ${isOn ? 'text-white' : 'text-gray-800'}`}>
                        {mod.label}
                      </span>
                      <span className={`block text-[11px] mt-0.5 leading-snug ${isOn ? 'text-white/85' : 'text-gray-500'}`}>
                        {mod.hint}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
            {/* 二级：仅保留笔记模块分页；模型模块点击后直接展示推断诊断 */}
            {activeMain === 'notes' && (
              <div
                className={`flex flex-wrap items-center gap-2 ${isNoteSearchActive ? 'opacity-40 pointer-events-none select-none' : ''}`}
              >
                <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wide mr-1 w-full sm:w-auto sm:mr-0">
                  笔记视图
                </span>
                {mainModules[0].tabs.map((tab) => {
                  const TabIcon = tab.icon;
                  const isOn = viewMode === tab.id;
                  return (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setViewMode(tab.id)}
                      className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs sm:text-sm font-medium transition-all shrink-0 border ${
                        isOn
                          ? 'bg-rose-500 text-white border-rose-500 shadow-md'
                          : 'bg-white text-gray-700 border-gray-100 hover:bg-rose-50 hover:border-rose-100'
                      }`}
                    >
                      <TabIcon className="w-4 h-4 shrink-0 opacity-90" aria-hidden />
                      <span className="whitespace-nowrap">{tab.label}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* 与评论分析页中间主卡片同构：圆角、描边、滚动在内层 */}
          <div className="relative flex-1 min-h-[min(400px,70vh)] rounded-2xl border border-slate-200/90 bg-white/80 backdrop-blur-sm shadow-xl overflow-hidden flex flex-col">
            <div className="flex-1 overflow-y-auto overscroll-contain p-5 md:p-8">
              {filteredTopics.length === 0 ? (
                <EmptyState type="search" />
              ) : isNoteSearchActive ? (
                <>
                  <NoteSearchResults
                    records={noteSearchMatches}
                    query={searchQuery.trim()}
                    topicByName={topicByName}
                    onOpenNote={(noteId) => setSelectedNoteId(noteId)}
                    onOpenTopic={(topic) => {
                      setSelectedTopic(topic);
                      setIsMobileSidebarOpen(false);
                    }}
                  />
                </>
              ) : (
                <>
                  {viewMode === 'compare' && (
                    <CompareView topics={filteredTopics} onBack={() => setViewMode('bertopic')} />
                  )}
                  {viewMode !== 'compare' && (
                    <>
                      {viewMode === 'bertopic' && (
                        <SubTopicStructureView topics={filteredTopics} allRecords={allRecords} />
                      )}
                      {viewMode === 'confidence' && (
                        <ModelQualityDashboard topics={topics} allRecords={allRecords} />
                      )}
                      {viewMode === 'geo' && (
                        <GeoDistributionView data={ipDistData} records={allRecords} topics={filteredTopics} />
                      )}
                      {viewMode === 'timeline' && <TemporalEvolutionView records={allRecords} />}
                    </>
                  )}
                </>
              )}
            </div>
          </div>
      </section>

        {/* 右侧详情面板 */}
        <aside
          className={`
            lg:w-80 xl:w-96 lg:shrink-0 bg-white/60 backdrop-blur-sm border-l border-rose-100
            overflow-y-auto p-3 sm:p-4 min-h-0
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

function NoteSearchResults({
  records,
  query,
  topicByName,
  onOpenNote,
  onOpenTopic,
}: {
  records: TopicRecord[];
  query: string;
  topicByName: Map<string, Topic>;
  onOpenNote: (noteId: string) => void;
  onOpenTopic: (t: Topic) => void;
}) {
  const scrollParentRef = useRef<HTMLDivElement>(null);
  const rowVirtualizer = useVirtualizer({
    count: records.length,
    getScrollElement: () => scrollParentRef.current,
    estimateSize: () => 172,
    overscan: 6,
    getItemKey: (index) => records[index]?.note_id ?? index,
  });

  if (records.length === 0) {
    return <EmptyState type="search" title="无命中笔记" description="请更换关键词或缩短检索词" />;
  }

  return (
    <div className="space-y-4 flex flex-col min-h-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2 shrink-0">
        <h3 className="text-base font-semibold text-gray-800">搜索结果</h3>
        <p className="text-xs text-gray-500">
          已按点赞数排序；共 {records.length.toLocaleString()} 条，可滚动浏览全部；点击卡片打开笔记，点击主题标签可在右侧查看该主题
        </p>
      </div>
      <div
        ref={scrollParentRef}
        className="max-h-[min(65vh,620px)] overflow-y-auto overflow-x-hidden rounded-xl border border-gray-100/80 pr-1"
      >
        <div className="relative w-full" style={{ height: rowVirtualizer.getTotalSize() }}>
          {rowVirtualizer.getVirtualItems().map((vRow) => {
            const r = records[vRow.index];
            const topic = topicByName.get(r.macro_topic);
            const title = r.title?.trim() || '（无标题）';
            const snippetSource = r.content?.trim() || r.desc?.trim() || r.segmented_text || '';
            const snippet = excerptAroundQuery(snippetSource, query, 64);
            const q = query.trim();
            const markSnippet = () => {
              if (!q) return <span className="text-gray-600 text-sm leading-relaxed">{snippet || '—'}</span>;
              const i = snippet.toLowerCase().indexOf(q.toLowerCase());
              if (i < 0) return <span className="text-gray-600 text-sm leading-relaxed">{snippet}</span>;
              return (
                <span className="text-gray-600 text-sm leading-relaxed">
                  {snippet.slice(0, i)}
                  <mark className="bg-amber-200/90 text-gray-900 rounded px-0.5">{snippet.slice(i, i + q.length)}</mark>
                  {snippet.slice(i + q.length)}
                </span>
              );
            };
            const markTitle = () => {
              if (!q) return <span className="font-medium text-gray-800">{title}</span>;
              const tl = title;
              const i = tl.toLowerCase().indexOf(q.toLowerCase());
              if (i < 0) return <span className="font-medium text-gray-800">{tl}</span>;
              return (
                <span className="font-medium text-gray-800">
                  {tl.slice(0, i)}
                  <mark className="bg-amber-200/90 text-gray-900 rounded px-0.5">{tl.slice(i, i + q.length)}</mark>
                  {tl.slice(i + q.length)}
                </span>
              );
            };
            return (
              <div
                key={vRow.key}
                className="absolute left-0 top-0 w-full pb-3"
                style={{ transform: `translateY(${vRow.start}px)` }}
              >
                <div className="rounded-2xl border border-gray-100 bg-gradient-to-br from-white to-rose-50/20 p-4 shadow-sm hover:border-rose-200/80 transition-colors">
                  <div className="flex flex-wrap items-start justify-between gap-2 mb-2">
                    <button
                      type="button"
                      onClick={() => onOpenNote(r.note_id)}
                      className="text-left flex-1 min-w-0"
                    >
                      {markTitle()}
                    </button>
                    <button
                      type="button"
                      onClick={() => topic && onOpenTopic(topic)}
                      disabled={!topic}
                      className="shrink-0 text-xs font-semibold px-2.5 py-1 rounded-lg bg-rose-100 text-rose-700 hover:bg-rose-200 disabled:opacity-50 disabled:cursor-not-allowed"
                      title={topic ? '在右侧面板打开该主题' : undefined}
                    >
                      {r.macro_topic}
                    </button>
                  </div>
                  <div className="mb-2">{markSnippet()}</div>
                  <div className="flex flex-wrap items-center gap-3 text-xs text-gray-400">
                    <span>❤️ {formatNumber(r.liked_count || 0)}</span>
                    <span>微观 #{r.micro_topic_id}</span>
                    {r.confidence > 0 && <span>置信 {(r.confidence * 100).toFixed(0)}%</span>}
                    <button
                      type="button"
                      onClick={() => onOpenNote(r.note_id)}
                      className="ml-auto text-rose-600 font-medium hover:underline"
                    >
                      查看笔记
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/** 数据概览：宏观分布环形图 + 有效样本覆盖率 + 指标卡 */
function SubTopicPanoramaOverview({ topics, allRecords }: { topics: Topic[]; allRecords: TopicRecord[] }) {
  const totalMicroTopics = useMemo(() => topics.reduce((s, t) => s + t.microTopics.length, 0), [topics]);
  const macroTopicCount = useMemo(() => topics.filter((t) => !isNoiseMacroLabel(t.name)).length, [topics]);
  const totalRecords = allRecords.length;
  const avgConf = totalRecords > 0 ? allRecords.reduce((s, r) => s + r.confidence, 0) / totalRecords : 0;
  const highConfN = useMemo(() => allRecords.filter((r) => r.confidence > 0.7).length, [allRecords]);

  const macroPieData = useMemo(() => {
    const confByMacro = new Map<string, { sum: number; n: number }>();
    for (const r of allRecords) {
      if (isNoiseTopicRecord(r)) continue;
      const key = normalizedMacroForSankey(r);
      const cur = confByMacro.get(key) ?? { sum: 0, n: 0 };
      const c = typeof r.confidence === 'number' && !Number.isNaN(r.confidence) ? r.confidence : 0;
      cur.sum += c;
      cur.n += 1;
      confByMacro.set(key, cur);
    }
    const validTopics = topics.filter((t) => !isNoiseMacroLabel(t.name));
    const sum = validTopics.reduce((s, t) => s + t.noteCount, 0);
    return validTopics.map((t) => {
      const key = t.name.trim();
      const agg = confByMacro.get(key) ?? { sum: 0, n: 0 };
      const avgConfidence = agg.n > 0 ? agg.sum / agg.n : 0;
      return {
        name: t.name,
        value: t.noteCount,
        fill: macroTopicDisplayColor(t.name),
        pctLabel: sum > 0 ? ((t.noteCount / sum) * 100).toFixed(1) : '0.0',
        avgConfidence,
      };
    });
  }, [topics, allRecords]);

  const coverageStats = useMemo(() => {
    const noise = allRecords.filter(isNoiseTopicRecord).length;
    const valid = Math.max(0, totalRecords - noise);
    const pct = totalRecords > 0 ? (valid / totalRecords) * 100 : 0;
    return { noise, valid, pct };
  }, [allRecords, totalRecords]);

  /** 内环：全量笔记的归类置信度分档 + 噪声类（与宏观主题外环形成「主题结构 × 置信度」对照） */
  const confidenceInnerPie = useMemo(() => {
    let hi = 0;
    let mid = 0;
    let low = 0;
    let none = 0;
    let noise = 0;
    for (const r of allRecords) {
      if (isNoiseTopicRecord(r)) {
        noise++;
        continue;
      }
      const c = typeof r.confidence === 'number' && !Number.isNaN(r.confidence) ? r.confidence : 0;
      if (c >= 0.7) hi++;
      else if (c >= 0.3) mid++;
      else if (c > 0) low++;
      else none++;
    }
    return [
      { name: '高置信', value: hi, fill: '#059669' },
      { name: '中置信', value: mid, fill: '#d97706' },
      { name: '低置信', value: low, fill: '#ea580c' },
      { name: '无置信度', value: none, fill: '#cbd5e1' },
      { name: '噪声等', value: noise, fill: '#94a3b8' },
    ].filter((e) => e.value > 0);
  }, [allRecords]);

  const sankeyData = useMemo(() => {
    if (allRecords.length === 0) {
      return {
        nodes: [],
        links: [] as Array<{ source: number; target: number; value: number }>,
        microCount: 0,
      };
    }

    const leftNodeName = '原始笔记';
    const microTotals = new Map<number, number>();
    const microToMacro = new Map<string, number>();

    for (const r of allRecords) {
      const microId = Number.isFinite(r.micro_topic_id) ? r.micro_topic_id : -1;
      const macroName = normalizedMacroForSankey(r);
      microTotals.set(microId, (microTotals.get(microId) || 0) + 1);
      const key = `${microId}${SANKEY_PART_SEP}${macroName}`;
      microToMacro.set(key, (microToMacro.get(key) || 0) + 1);
    }

    const macroSet = new Set<string>();
    for (const r of allRecords) {
      macroSet.add(normalizedMacroForSankey(r));
    }

    const macroList = sankeyOrderedMacroNames(macroSet);
    const microList = [...microTotals.keys()].sort((a, b) => {
      const pa = sankeyPrimaryMacroIndex(a, microToMacro, macroList);
      const pb = sankeyPrimaryMacroIndex(b, microToMacro, macroList);
      if (pa !== pb) return pa - pb;
      const ca = microTotals.get(a) || 0;
      const cb = microTotals.get(b) || 0;
      /** 同主宏观下篇数少的在上、多的在下，避免细条被压在粗条下方导致跨线杂乱 */
      if (ca !== cb) return ca - cb;
      return a - b;
    });

    const nodes = [
      { name: leftNodeName },
      ...microList.map((id) => ({ name: microNodeLabel(id) })),
      ...macroList.map((name) => ({ name })),
    ];

    const microIndex = new Map<number, number>();
    microList.forEach((id, i) => microIndex.set(id, i + 1));
    const macroStart = 1 + microList.length;
    const macroIndex = new Map<string, number>();
    macroList.forEach((name, i) => macroIndex.set(name, macroStart + i));

    const links: Array<{ source: number; target: number; value: number }> = [];

    microList.forEach((id) => {
      const cnt = microTotals.get(id) || 0;
      const target = microIndex.get(id);
      if (cnt > 0 && target != null) {
        links.push({ source: 0, target, value: cnt });
      }
    });

    for (const [key, cnt] of microToMacro.entries()) {
      if (cnt <= 0) continue;
      const sep = key.indexOf(SANKEY_PART_SEP);
      if (sep < 0) continue;
      const microRaw = key.slice(0, sep);
      const macroName = key.slice(sep + SANKEY_PART_SEP.length);
      const microId = Number(microRaw);
      const source = microIndex.get(microId);
      const target = macroIndex.get(macroName);
      if (source != null && target != null) {
        links.push({ source, target, value: cnt });
      }
    }

    return {
      nodes,
      links,
      microCount: microList.length,
    };
  }, [allRecords]);

  return (
    <div className="space-y-5 sm:space-y-6">
      <div className="rounded-2xl border border-rose-100/70 bg-gradient-to-br from-white to-rose-50/20 p-4 sm:p-5 shadow-md">
        <h4 className="font-semibold text-gray-800 text-sm mb-1">主题分布与置信度（合并视图）</h4>
        <p className="text-[11px] text-gray-400 mb-2">
          外环：各宏观主题篇数，点线引出环外标注该主题平均置信度与篇数占比。内环：置信度分档与噪声，点线引出至内外环间隙。分档与下方「高置信」阈值一致。
        </p>
        {macroPieData.length === 0 ? (
          <p className="text-sm text-gray-400 py-12 text-center">暂无有效宏观主题</p>
        ) : (
          <div className="relative h-[min(420px,56vw)] min-h-[380px] w-full">
            <ResponsiveContainerRC width="100%" height="100%">
              <PieChartRC margin={{ top: 8, right: 108, bottom: 8, left: 108 }}>
                <PieRC
                  data={macroPieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={92}
                  outerRadius={118}
                  paddingAngle={2}
                  labelLine={false}
                  label={(p: {
                    cx: number;
                    cy: number;
                    midAngle: number;
                    innerRadius: number;
                    outerRadius: number;
                    name: string;
                    value: number;
                    percent: number;
                    avgConfidence?: number;
                    payload?: { avgConfidence?: number };
                  }) => <MacroOutsideLeaderLabel {...p} />}
                >
                  {macroPieData.map((e) => (
                    <CellRC key={e.name} fill={e.fill} />
                  ))}
                </PieRC>
                <PieRC
                  data={confidenceInnerPie.length > 0 ? confidenceInnerPie : [{ name: '—', value: 1, fill: '#f1f5f9' }]}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={54}
                  outerRadius={76}
                  stroke="#fff"
                  strokeWidth={2}
                  labelLine={false}
                  label={(p: {
                    cx: number;
                    cy: number;
                    midAngle: number;
                    innerRadius: number;
                    outerRadius: number;
                    name: string;
                    value: number;
                    percent: number;
                    payload?: { fill?: string; name?: string };
                  }) => (p.name === '—' ? null : <ConfidenceBetweenRingsLeaderLabel {...p} />)}
                >
                  {(confidenceInnerPie.length > 0 ? confidenceInnerPie : [{ name: '—', value: 1, fill: '#f1f5f9' }]).map(
                    (e) => (
                      <CellRC key={e.name} fill={e.fill} />
                    ),
                  )}
                </PieRC>
                <TooltipRC
                  content={({
                    active,
                    payload,
                  }: {
                    active?: boolean;
                    payload?: Array<{ payload: { name: string; value: number; avgConfidence?: number } }>;
                  }) => {
                    if (!active || !payload?.length) return null;
                    const p = payload[0].payload;
                    const pct = totalRecords > 0 ? (p.value / totalRecords) * 100 : 0;
                    const hasThemeConf = typeof p.avgConfidence === 'number';
                    return (
                      <div className="rounded-xl border border-rose-100/70 bg-white/95 shadow-lg px-3 py-2 text-xs">
                        <div className="font-semibold text-gray-800">{p.name}</div>
                        <div className="text-gray-600 mt-0.5">
                          {formatNumber(p.value)} 篇（占全量 {pct.toFixed(1)}%）
                        </div>
                        {hasThemeConf ? (
                          <div className="text-gray-600 mt-1">该主题平均置信度 {(p.avgConfidence! * 100).toFixed(1)}%</div>
                        ) : null}
                      </div>
                    );
                  }}
                />
              </PieChartRC>
            </ResponsiveContainerRC>
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-3xl sm:text-4xl font-bold text-slate-700 tabular-nums">
                  {coverageStats.pct.toFixed(1)}%
                </div>
                <div className="text-[11px] text-slate-500 mt-1">有效样本覆盖率</div>
              </div>
            </div>
          </div>
        )}
        {totalRecords > 0 && (
          <div className="flex flex-wrap justify-center gap-x-5 gap-y-1 text-xs text-gray-600 mt-2">
            <span>
              有效样本{' '}
              <strong className="text-cyan-700 tabular-nums">{formatNumber(coverageStats.valid)}</strong> 篇
            </span>
            <span>
              高置信(≥0.7){' '}
              <strong className="text-emerald-700 tabular-nums">{formatNumber(highConfN)}</strong> 篇
            </span>
            <span>
              噪声等 <strong className="text-gray-500 tabular-nums">{formatNumber(coverageStats.noise)}</strong> 篇
            </span>
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-rose-100/70 bg-gradient-to-br from-white to-sky-50/30 p-4 sm:p-5 shadow-md">
        <h4 className="font-semibold text-gray-800 text-sm mb-1">主题聚类流向（桑基图）</h4>
        <p className="text-[11px] text-gray-400 mb-3">
          从左到右：原始笔记 → 细分主题 → 宏观主题。带状宽度表示篇数；汇入宏观一侧为主题色，指向细分主题一侧为中性灰。
        </p>
        {sankeyData.nodes.length === 0 || sankeyData.links.length === 0 ? (
          <p className="text-sm text-gray-400 py-12 text-center">暂无可绘制的流向数据</p>
        ) : (
          <div className="w-full max-w-[min(100%,780px)] mx-auto h-[520px] min-h-[520px] overflow-x-auto overflow-y-hidden overscroll-contain rounded-xl border border-slate-100/80 bg-white/40">
            <ResponsiveContainerRC width="100%" height="100%" minHeight={520} className="[&_.recharts-surface]:overflow-visible">
              <SankeyRC
                data={sankeyData}
                nameKey="name"
                margin={{ top: 20, left: 88, right: 96, bottom: 20 }}
                nodePadding={3}
                nodeWidth={14}
                linkCurvature={0.52}
                iterations={64}
                /** false：碰撞整理不按 y 重排节点，纵向顺序与 data.nodes 一致（同宏观组内细条在上） */
                sort={false}
                link={(linkProps: {
                  sourceX: number;
                  sourceY: number;
                  targetX: number;
                  targetY: number;
                  sourceControlX: number;
                  targetControlX: number;
                  linkWidth: number;
                  payload?: { target?: { name?: string } };
                }) => <SankeyThemedLink {...linkProps} />}
                node={(nodeProps: {
                  x?: number;
                  y?: number;
                  width?: number;
                  height?: number;
                  payload?: { name?: string; value?: number; depth?: number };
                }) => (
                  <SankeyNodeWithLabel
                    x={nodeProps.x ?? 0}
                    y={nodeProps.y ?? 0}
                    width={nodeProps.width ?? 0}
                    height={nodeProps.height ?? 0}
                    payload={nodeProps.payload}
                  />
                )}
              >
                <TooltipRC
                  isAnimationActive={false}
                  animationDuration={0}
                  wrapperStyle={{ pointerEvents: 'none' }}
                  content={({
                    active,
                    payload: tipPayload,
                  }: {
                    active?: boolean;
                    payload?: Array<{
                      name?: string;
                      value?: number | string;
                      payload?: {
                        payload?: {
                          source?: { name?: string };
                          target?: { name?: string };
                          value?: number;
                          name?: string;
                        };
                      };
                    }>;
                  }) => {
                    if (!active || !tipPayload?.length) return null;
                    const row = tipPayload[0];
                    const outer = row.payload as { payload?: { value?: number; source?: unknown; target?: unknown } } | undefined;
                    const merged = outer?.payload;
                    let vRaw = row.value;
                    if (vRaw == null || vRaw === '') {
                      vRaw = merged?.value;
                    }
                    const v = Number(vRaw ?? 0);
                    if (
                      merged &&
                      merged.source &&
                      merged.target &&
                      typeof merged.source === 'object' &&
                      typeof merged.target === 'object'
                    ) {
                      const s = String((merged.source as { name?: string }).name ?? '');
                      const t = String((merged.target as { name?: string }).name ?? '');
                      return (
                        <div className="rounded-xl border border-rose-100/70 bg-white/95 shadow-lg px-3 py-2 text-xs">
                          <div className="font-semibold text-gray-800">
                            {s} → {t}
                          </div>
                          <div className="text-gray-600 mt-0.5">{formatNumber(v)} 篇</div>
                        </div>
                      );
                    }
                    const title =
                      row.name ||
                      String((merged as { name?: string } | undefined)?.name ?? '') ||
                      String((outer as { name?: string } | undefined)?.name ?? '');
                    return (
                      <div className="rounded-xl border border-rose-100/70 bg-white/95 shadow-lg px-3 py-2 text-xs">
                        <div className="font-semibold text-gray-800">{title}</div>
                        <div className="text-gray-600 mt-0.5">{formatNumber(v)} 篇</div>
                      </div>
                    );
                  }}
                />
              </SankeyRC>
            </ResponsiveContainerRC>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3 sm:gap-4">
        {(
          [
            {
              label: '宏观主题',
              hint: '已排除噪声类名',
              value: String(macroTopicCount),
              Icon: LayoutGrid,
              ring: 'ring-rose-100/70',
              iconClass: 'from-rose-500/25 to-pink-500/15 text-rose-600',
              border: 'border-rose-200/45 hover:border-rose-300/60',
            },
            {
              label: '笔记总数',
              hint: '最终合并行数',
              value: formatNumber(totalRecords),
              Icon: NotebookText,
              ring: 'ring-sky-100/80',
              iconClass: 'from-sky-500/25 to-cyan-500/15 text-sky-600',
              border: 'border-sky-200/40 hover:border-sky-300/55',
            },
            {
              label: '微观主题',
              hint: 'BERTopic 细分类数',
              value: String(totalMicroTopics),
              Icon: Boxes,
              ring: 'ring-violet-100/80',
              iconClass: 'from-violet-500/25 to-purple-500/15 text-violet-600',
              border: 'border-violet-200/40 hover:border-violet-300/55',
            },
            {
              label: '平均置信度',
              hint: '映射置信度均值',
              value: `${(avgConf * 100).toFixed(1)}%`,
              Icon: Percent,
              ring: 'ring-fuchsia-100/70',
              iconClass: 'from-fuchsia-500/25 to-pink-500/15 text-fuchsia-600',
              border: 'border-fuchsia-200/40 hover:border-fuchsia-300/55',
              valueClass: 'text-fuchsia-700',
            },
            {
              label: '高置信笔记',
              hint: '置信度 > 0.7',
              value: `${formatNumber(highConfN)} 篇`,
              Icon: BadgeCheck,
              ring: 'ring-emerald-100/80',
              iconClass: 'from-emerald-500/25 to-teal-500/15 text-emerald-600',
              border: 'border-emerald-200/40 hover:border-emerald-300/55',
              valueClass: 'text-emerald-700',
            },
          ] as const
        ).map((item) => {
          const IconCmp = item.Icon;
          return (
          <div
            key={item.label}
            className={`group rounded-2xl border bg-gradient-to-br from-white via-white to-rose-50/20 p-4 shadow-md shadow-gray-200/30 transition-all duration-200 hover:shadow-lg hover:shadow-rose-100/25 ${item.border}`}
          >
            <div className="flex items-start gap-3">
              <div
                className={`rounded-xl bg-gradient-to-br p-2.5 ring-1 ${item.ring} ${item.iconClass}`}
              >
                <IconCmp className="w-5 h-5" strokeWidth={2} aria-hidden />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">{item.label}</p>
                <p
                  className={`text-xl sm:text-2xl font-bold tracking-tight tabular-nums mt-1 ${'valueClass' in item && item.valueClass ? item.valueClass : 'text-gray-900'}`}
                >
                  {item.value}
                </p>
                <p className="text-[10px] text-gray-400 mt-1.5 leading-snug">{item.hint}</p>
              </div>
            </div>
          </div>
        );
        })}
      </div>
    </div>
  );
}

function MacroWordCloudCard({
  title,
  color,
  entries,
  noteCount,
}: {
  title: string;
  color: string;
  entries: { name: string; value: number }[];
  noteCount: number;
}) {
  if (entries.length === 0) {
    return (
      <div className="rounded-xl border border-gray-100 bg-white p-4 shadow-sm">
        <h4 className="font-semibold text-sm text-gray-800 mb-1 truncate">{title}</h4>
        <p className="text-xs text-gray-400">该分类下暂无可用关键词（请检查 CSV 中关键词列）。</p>
      </div>
    );
  }
  const max = entries[0]!.value;
  const min = entries[entries.length - 1]!.value;
  const span = max - min || 1;

  return (
    <div className="rounded-xl border border-gray-100 bg-gradient-to-br from-white to-rose-50/25 p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
          <h4 className="font-semibold text-sm text-gray-800 truncate">{title}</h4>
        </div>
        <span className="text-xs text-gray-400 shrink-0">{noteCount} 篇</span>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-x-2.5 gap-y-2 min-h-[200px] py-3 content-center">
        {entries.map((e, idx) => {
          const t = (e.value - min) / span;
          const fontSize = 12 + t * 17;
          const opacity = 0.42 + t * 0.58;
          const rot = ((idx * 17) % 7) - 3;
          return (
            <span
              key={`${e.name}-${idx}`}
              className="inline-block font-semibold leading-snug tracking-tight select-none"
              style={{
                color,
                fontSize: `${fontSize}px`,
                opacity,
                transform: `rotate(${rot}deg)`,
              }}
              title={`${e.name} · 权重 ${e.value}`}
            >
              {e.name}
            </span>
          );
        })}
      </div>
    </div>
  );
}

/** 宏观关键词横向条形图 + 代表词（与词云页同主题左右对照） */
function MacroKeywordSummaryBlock({ topic, accent }: { topic: Topic; accent: string }) {
  const ranked = buildWordCloudEntriesForTopic(topic).slice(0, 14);
  const keywordBars = useMemo(
    () =>
      ranked
        .slice(0, 12)
        .map((row) => ({ ...row }))
        .reverse()
        .map((row) => ({
          ...row,
          short: row.name.length > 8 ? `${row.name.slice(0, 8)}…` : row.name,
        })),
    [ranked],
  );
  const maxKeywordWeight = keywordBars.length ? Math.max(...keywordBars.map((k) => k.value), 1) : 1;

  return (
    <div className="rounded-xl border border-gray-100 bg-gradient-to-br from-white to-rose-50/20 p-4 shadow-sm flex flex-col min-h-0 h-full">
      <h5 className="text-xs font-semibold text-gray-600 mb-3">宏观关键词汇总</h5>
      {keywordBars.length === 0 ? (
        <p className="text-xs text-gray-400">无关键词数据</p>
      ) : (
        <div className="flex-1 min-h-[260px] h-[min(340px,42vh)] w-full rounded-xl border border-gray-100/80 bg-white/80 p-2">
          <ResponsiveContainerRC width="100%" height="100%">
            <BarChartRC layout="vertical" data={keywordBars} margin={{ top: 6, right: 36, left: 4, bottom: 6 }}>
              <CartesianGridRC strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
              <XAxisRC type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxisRC type="category" dataKey="short" width={68} tick={{ fontSize: 11, fill: '#475569' }} />
              <TooltipRC
                content={({ active, payload }: any) => {
                  if (!active || !payload?.length) return null;
                  const row = payload[0].payload as { name: string; value: number };
                  return (
                    <div className="rounded-xl border border-gray-100 bg-white/95 px-3 py-2 text-xs shadow-lg">
                      <p className="font-semibold text-gray-800">{row.name}</p>
                      <p className="text-gray-600 mt-1">累积权重 {row.value.toFixed(1)}</p>
                    </div>
                  );
                }}
              />
              <BarRC dataKey="value" radius={[0, 6, 6, 0]} maxBarSize={20}>
                {keywordBars.map((row) => {
                  const alpha = 0.4 + 0.6 * (row.value / maxKeywordWeight);
                  return <CellRC key={`${topic.id}-${row.name}`} fill={accent} fillOpacity={alpha} />;
                })}
              </BarRC>
            </BarChartRC>
          </ResponsiveContainerRC>
        </div>
      )}
      {topic.keywords.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <p className="text-[10px] text-gray-500 mb-2">代表词</p>
          <div className="flex flex-wrap gap-1.5">
            {topic.keywords.slice(0, 10).map((kw) => (
              <span
                key={kw}
                className="px-2 py-0.5 rounded-full text-[11px] font-medium"
                style={{ backgroundColor: `${accent}1f`, color: accent }}
              >
                {kw}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** 全部宏观下的微观主题合并为一张横向条形图（同色表示所属宏观） */
function MacroMicroUnifiedChart({ topics }: { topics: Topic[] }) {
  const { rows, legendTopics, maxVal } = useMemo(() => {
    type Row = {
      key: string;
      short: string;
      macro: string;
      microId: number;
      value: number;
      fill: string;
      kw: string;
    };
    const out: Row[] = [];
    for (const t of topics) {
      const fill = macroTopicDisplayColor(t.name);
      for (const mt of t.microTopics) {
        const macro = t.name;
        const macroShort = macro.length > 9 ? `${macro.slice(0, 8)}…` : macro;
        const short = `${macroShort} · T${mt.id}`;
        out.push({
          key: `${t.id}-${mt.id}`,
          short: short.length > 26 ? `${short.slice(0, 24)}…` : short,
          macro,
          microId: mt.id,
          value: mt.noteCount,
          fill,
          kw: mt.keywords.slice(0, 5).join('、') || '—',
        });
      }
    }
    out.sort((a, b) => b.value - a.value);
    const mv = out.length ? Math.max(...out.map((r) => r.value), 1) : 1;
    const leg = topics.filter((t) => t.microTopics.length > 0);
    return { rows: out, legendTopics: leg, maxVal: mv };
  }, [topics]);

  const chartHeight = Math.min(Math.max(rows.length * 22 + 88, 180), 680);

  if (rows.length === 0) {
    return (
      <div className="rounded-xl border border-gray-100/90 bg-white p-6 shadow-sm text-center text-sm text-gray-400">
        暂无微观主题数据
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-100/90 bg-white p-3 sm:p-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-2 mb-3">
        <div>
          <h4 className="text-sm font-semibold text-gray-800">全部分类 · 微观主题规模</h4>
          <p className="text-[10px] text-gray-500 mt-0.5">按篇数降序；条形色块与左侧宏观主题一致。</p>
        </div>
        <div className="flex flex-wrap justify-end gap-x-3 gap-y-1 max-w-[min(100%,520px)]">
          {legendTopics.map((t) => (
            <span key={t.id} className="inline-flex items-center gap-1.5 text-[10px] text-gray-600">
              <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: macroTopicDisplayColor(t.name) }} />
              <span className="truncate max-w-[9rem]">{t.name}</span>
            </span>
          ))}
        </div>
      </div>
      <div className="w-full min-w-0 overflow-x-auto">
        <ResponsiveContainerRC width="100%" height={chartHeight}>
          <BarChartRC layout="vertical" data={rows} margin={{ top: 4, right: 16, left: 4, bottom: 4 }}>
            <CartesianGridRC strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
            <XAxisRC
              type="number"
              domain={[0, maxVal]}
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              tickFormatter={(v: number) => formatNumber(v)}
            />
            <YAxisRC type="category" dataKey="short" width={148} tick={{ fontSize: 10, fill: '#475569' }} interval={0} />
            <TooltipRC
              content={({ active, payload }: any) => {
                if (!active || !payload?.length) return null;
                const row = payload[0].payload as (typeof rows)[0];
                return (
                  <div className="rounded-xl border border-gray-100 bg-white/95 px-3 py-2 text-xs shadow-lg max-w-[min(90vw,320px)]">
                    <p className="font-semibold text-gray-800">{row.macro}</p>
                    <p className="text-gray-600 mt-0.5">
                      T{row.microId} · {formatNumber(row.value)} 篇
                    </p>
                    <p className="text-[11px] text-gray-500 mt-1.5 leading-snug line-clamp-4" title={row.kw}>
                      {row.kw}
                    </p>
                  </div>
                );
              }}
            />
            <BarRC dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false}>
              {rows.map((r) => (
                <CellRC key={r.key} fill={r.fill} fillOpacity={0.82} />
              ))}
            </BarRC>
          </BarChartRC>
        </ResponsiveContainerRC>
      </div>
    </div>
  );
}

function SubTopicStructureView({
  topics,
  allRecords,
}: {
  topics: Topic[];
  allRecords: TopicRecord[];
}) {
  const [subView, setSubView] = useState<'micro' | 'data' | 'wordcloud'>('data');

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3 sm:mb-4">
        <div>
          <h3 className="text-base sm:text-lg font-semibold text-gray-700">细分主题结构</h3>
        </div>
        <div className="flex flex-wrap gap-2">
          {[
            { id: 'data' as const, label: '数据概览', icon: LayoutDashboard },
            { id: 'micro' as const, label: '微观主题', icon: Layers },
            { id: 'wordcloud' as const, label: '词云', icon: Cloud },
          ].map((btn) => {
            const Icon = 'icon' in btn ? btn.icon : undefined;
            return (
              <button
                key={btn.id}
                type="button"
                onClick={() => setSubView(btn.id)}
                className={`inline-flex items-center gap-1.5 px-2 sm:px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium ${
                  subView === btn.id ? 'bg-rose-500 text-white' : 'bg-gray-100 text-gray-600'
                }`}
              >
                {Icon ? <Icon className="w-3.5 h-3.5 opacity-90 shrink-0" aria-hidden /> : null}
                {btn.label}
              </button>
            );
          })}
        </div>
      </div>

      {subView === 'data' && <SubTopicPanoramaOverview topics={topics} allRecords={allRecords} />}

      {(subView === 'micro' || subView === 'wordcloud') && (
        <div key={subView} className="subtopic-chart-panel-enter">
          {subView === 'micro' && <MacroMicroUnifiedChart topics={topics} />}
          {subView === 'wordcloud' && (
            <div className="space-y-8">
              {topics.map((topic) => {
                const accent = macroTopicDisplayColor(topic.name);
                return (
                  <div
                    key={topic.id}
                    className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-stretch"
                  >
                    <MacroWordCloudCard
                      title={topic.name}
                      color={accent}
                      entries={buildWordCloudEntriesForTopic(topic)}
                      noteCount={topic.noteCount}
                    />
                    <MacroKeywordSummaryBlock topic={topic} accent={accent} />
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
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

// ─── 时序演变（按 raw `time` + 宏观主题）──
function TemporalEvolutionView({ records }: { records: TopicRecord[] }) {
  const [bucket, setBucket] = useState<'month' | 'week'>('month')
  const series = useMemo(() => buildTemporalMacroSeries(records, bucket), [records, bucket])
  const { rows, macroKeys, withTimeCount, missingTimeCount, totalCount, coveragePct } = series

  const xAxisInterval = rows.length > 16 ? Math.max(0, Math.floor(rows.length / 14)) : 0

  return (
    <div className="space-y-6 md:space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
        <div>
          <h3 className="text-base md:text-lg font-semibold text-gray-800 tracking-tight">宏观主题 × 发布时间</h3>
          <p className="text-xs md:text-sm text-gray-500 mt-1 leading-relaxed max-w-3xl">
            时间来自原始笔记 JSON 的 <code className="text-rose-600/90 text-[11px]">time</code> 字段（与 final_pro_topics 按{' '}
            <code className="text-rose-600/90 text-[11px]">note_id</code> 合并）；支持 Unix 秒/毫秒与常见日期字符串。
          </p>
          <p className="text-xs text-gray-400 mt-2">
            可解析时间：<span className="font-medium text-rose-600">{formatNumber(withTimeCount)}</span> /{' '}
            {formatNumber(totalCount)} 条（{coveragePct.toFixed(1)}%）
            {missingTimeCount > 0 && (
              <span className="ml-2">· 无时间 {formatNumber(missingTimeCount)} 条未计入下图</span>
            )}
          </p>
        </div>
        <div className="flex rounded-xl border border-rose-100 bg-white/90 p-1 shrink-0 shadow-sm">
          {(
            [
              { id: 'month' as const, label: '按月' },
              { id: 'week' as const, label: '按周' },
            ] as const
          ).map((b) => (
            <button
              key={b.id}
              type="button"
              onClick={() => setBucket(b.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                bucket === b.id ? 'bg-rose-500 text-white shadow' : 'text-gray-600 hover:bg-rose-50'
              }`}
            >
              {b.label}
            </button>
          ))}
        </div>
      </div>

      {rows.length === 0 || withTimeCount === 0 ? (
        <EmptyState
          type="data"
          title="暂无可展示的时间序列"
          description="请确认 rawdata 中笔记包含有效 time，且已成功与主题表按 note_id 合并。"
        />
      ) : (
        <div className="rounded-2xl border border-rose-200/50 bg-gradient-to-b from-rose-50/30 via-white to-white p-3 sm:p-5 shadow-lg shadow-rose-100/20">
          <div className="h-[min(440px,55vh)] w-full min-h-[280px]">
            <ResponsiveContainerRC width="100%" height="100%">
              <BarChartRC
                data={rows}
                margin={{ top: 12, right: 12, left: 4, bottom: bucket === 'month' ? 56 : 72 }}
              >
                <CartesianGridRC strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxisRC
                  dataKey="label"
                  tick={{ fontSize: 10, fill: '#64748b' }}
                  interval={xAxisInterval}
                  angle={bucket === 'week' ? -32 : -22}
                  textAnchor="end"
                  height={bucket === 'week' ? 68 : 52}
                />
                <YAxisRC tick={{ fontSize: 11, fill: '#64748b' }} allowDecimals={false} width={42} />
                <TooltipRC
                  contentStyle={{ borderRadius: 12, border: '1px solid #fecdd3', fontSize: 12 }}
                  formatter={(value: number) => [formatNumber(value), '篇']}
                />
                <LegendRC wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
                {macroKeys.map((m) => (
                  <BarRC
                    key={m}
                    dataKey={m}
                    name={m}
                    stackId="macro"
                    fill={macroTopicDisplayColor(m)}
                    maxBarSize={56}
                  />
                ))}
              </BarChartRC>
            </ResponsiveContainerRC>
          </div>
          <p className="text-xs text-gray-400 mt-3 text-center sm:text-left">
            堆叠柱高度 = 该{bucket === 'month' ? '月' : '周'}各宏观主题笔记篇数之和；配色与环形图宏观图例一致。
          </p>
        </div>
      )}

    </div>
  )
}

// ─── 地理分布视图 ──
function GeoDistributionView({
  data,
  records,
  topics,
}: {
  data: ReturnType<typeof buildIPDistribution>
  records: TopicRecord[]
  topics: Topic[]
}) {
  const [selectedLoc, setSelectedLoc] = useState<string | null>(null)
  const total = records.length || 1
  const hasIPRecords = data.filter(d => d.location !== '未知IP')
  const maxCount = hasIPRecords[0]?.count || 1
  const unknownItem = data.find(d => d.location === '未知IP')
  const knownTotal = total - (unknownItem?.count || 0)

  const selectedRecords = useMemo(() => {
    if (!selectedLoc) return []
    if (selectedLoc === '未知IP') {
      return records.filter(r => !r.ip_location || r.ip_location.trim() === '')
    }
    return records.filter(r => sameIpRegion(selectedLoc, r.ip_location))
  }, [selectedLoc, records])

  const ipMapData = useMemo(
    () => hasIPRecords.map(d => ({ rawLocation: d.location, count: d.count })),
    [hasIPRecords],
  )

  const onMapRegionClick = useCallback((raw: string | null) => {
    setSelectedLoc(raw !== null && raw === selectedLoc ? null : raw)
  }, [selectedLoc])

  /** 交叉表展示的主题列数（略增宽版面） */
  const crossTopicCols = useMemo(() => topics.slice(0, 5), [topics])

  const crossRows = useMemo(() => {
    return hasIPRecords.slice(0, 10).map((item) => {
      const locRecords = records.filter((r) => sameIpRegion(item.location, r.ip_location))
      const locTopics = new Map<string, number>()
      for (const r of locRecords) {
        const m = r.macro_topic || '未分类'
        locTopics.set(m, (locTopics.get(m) || 0) + 1)
      }
      const cells = crossTopicCols.map((t) => ({
        topicId: t.id,
        topicName: t.name,
        cnt: locTopics.get(t.name) || 0,
      }))
      return { item, cells }
    })
  }, [hasIPRecords, records, crossTopicCols])

  return (
    <div className="space-y-8 md:space-y-10">
      {/* 地图 + 排行（与外层主卡片一体，不再套内层白盒） */}
      <div>
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 mb-1">
          <h3 className="text-base md:text-lg font-semibold text-gray-800 tracking-tight">创作者 IP 属地分布</h3>
          {unknownItem && (
            <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded-full">
              {unknownItem.count} 条笔记无IP记录（{(unknownItem.pct * 100).toFixed(1)}%）
            </span>
          )}
        </div>
        <p className="text-xs md:text-sm text-gray-500 mb-5">
          基于 {knownTotal} 条有效 IP 的 <code className="text-rose-600/90 text-[11px]">ip_location</code> 统计
        </p>
        {data.length === 0 ? (
          <EmptyState type="data" title="暂无地理数据" description="ip_location 字段为空" />
        ) : (
          <div className="flex flex-col lg:flex-row gap-8 lg:gap-10 items-start">
            <div className="w-full lg:w-[min(620px,100%)] xl:w-[min(720px,100%)] shrink-0">
              <ChinaIPMapChart
                data={ipMapData}
                maxCount={maxCount}
                selectedRaw={selectedLoc}
                onRegionClick={onMapRegionClick}
                height={496}
                className="rounded-2xl border border-rose-200/50 bg-gradient-to-b from-rose-50/40 via-white to-white min-h-[300px] shadow-lg shadow-rose-100/30"
              />
              <p className="text-xs text-gray-400 mt-3 text-center lg:text-left leading-relaxed">
                地图为省级示意；点击省/区与右侧列表联动（与「广东 / 广东省」等同匹配）
              </p>
            </div>
            <div className="flex-1 min-w-0 space-y-2 max-h-[min(640px,58vh)] overflow-y-auto pr-1">
            {data.map((item, i) => {
              const isUnknown = item.location === '未知IP'
              const isSelected =
                selectedLoc !== null &&
                (isUnknown ? selectedLoc === '未知IP' : sameIpRegion(selectedLoc, item.location))
              return (
                <button
                  key={item.location}
                  onClick={() => setSelectedLoc(isSelected ? null : item.location)}
                  className={`w-full flex items-center gap-3 p-3 md:p-3.5 rounded-xl transition-all text-left ${
                    isSelected
                      ? isUnknown ? 'bg-gray-100 ring-1 ring-gray-300' : 'bg-rose-50 ring-1 ring-rose-300'
                      : 'hover:bg-gray-50'
                  }`}
                >
                  <span className="text-xs text-gray-400 w-5 text-right shrink-0 font-mono">{i + 1}</span>
                  <span className={`text-sm md:text-[15px] min-w-[3.5rem] shrink-0 font-medium ${isSelected ? 'text-rose-600' : 'text-gray-700'} ${isUnknown ? 'italic text-gray-500' : ''}`}>
                    {item.location}
                  </span>
                  <div className="flex-1 bg-gray-100 rounded-full h-3.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${isUnknown ? 'bg-gray-300' : 'bg-gradient-to-r from-rose-400 to-pink-400'}`}
                      style={{ width: `${(item.count / maxCount) * 100}%` }}
                    />
                  </div>
                  <span className={`text-sm font-medium w-16 text-right shrink-0 ${isSelected ? 'text-rose-600' : 'text-gray-600'}`}>
                    {item.count}
                  </span>
                  <span className="text-xs text-gray-400 w-14 text-right shrink-0">
                    {(item.pct * 100).toFixed(1)}%
                  </span>
                </button>
              )
            })}
            </div>
          </div>
        )}
      </div>

      {/* 选中地区的笔记 */}
      {selectedLoc && (
        <div className="pt-6 md:pt-8 border-t border-rose-100/50">
          <h3 className="text-base md:text-lg font-semibold text-gray-800 mb-1">
            {selectedLoc === '未知IP' ? '无IP记录' : selectedLoc} 创作者笔记
          </h3>
          <p className="text-xs text-gray-400 mb-4">{selectedRecords.length} 篇</p>
          {selectedRecords.length === 0 ? (
            <p className="text-sm text-gray-400 text-center py-6">暂无数据</p>
          ) : (
            <div className="space-y-2 max-h-[min(380px,42vh)] overflow-y-auto">
              {selectedRecords.slice(0, 20).map(r => (
                <div key={r.note_id} className="p-3 bg-gray-50 rounded-xl">
                  <div className="text-sm font-medium text-gray-700 line-clamp-1">{r.title || r.content?.slice(0, 60)}</div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-gray-400">
                    <span>❤️ {formatNumber(r.liked_count || 0)}</span>
                    <span>⭐ {formatNumber(r.collected_count || 0)}</span>
                    <span className="text-rose-400">{r.macro_topic}</span>
                    {selectedLoc === '未知IP' && (
                      <span className="ml-auto italic text-gray-400">无IP</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 地理 × 主题交叉（与地图同规则聚合 IP） */}
      {hasIPRecords.length >= 2 && crossTopicCols.length > 0 && (
        <div className="pt-6 md:pt-10 border-t border-rose-200/40">
          <RegionTopicMixChart
            rows={crossRows.map(({ item, cells }) => ({
              region: item.location,
              total: item.count,
              cells,
            }))}
            topicOrder={crossTopicCols}
          />
        </div>
      )}
    </div>
  )
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

  const timeStr = formatTopicTimestamp(record.time);

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
              {timeStr ? ` · ${timeStr}` : ''}
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
