/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useMemo } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Activity, Layers, PieChart as PieChartIcon, ShieldCheck, AlertTriangle, Library, BarChart2 } from 'lucide-react';
import type { Topic, TopicRecord, TopicStats } from '../data/topicData';
import { EmptyState } from './common/States';
import { formatNumber } from '../utils/responsive';

const RC = {
  ComposedChart: ComposedChart as any,
  BarChart: BarChart as any,
  Bar: Bar as any,
  Line: Line as any,
  CartesianGrid: CartesianGrid as any,
  XAxis: XAxis as any,
  YAxis: YAxis as any,
  Tooltip: Tooltip as any,
  Legend: Legend as any,
  PieChart: PieChart as any,
  Pie: Pie as any,
  Cell: Cell as any,
  ResponsiveContainer: ResponsiveContainer as any,
};

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const a = sorted[base];
  const b = sorted[base + 1];
  if (b === undefined) return a;
  return a + rest * (b - a);
}

const HIST_BINS = 24;
const PIE_COLORS = { normal: '#f43f5e', noise: '#94a3b8', track: '#e2e8f0' };

function clusterConfidenceColor(c: number): string {
  if (c >= 0.92) return '#16a34a';
  if (c >= 0.85) return '#22c55e';
  if (c >= 0.75) return '#84cc16';
  if (c >= 0.65) return '#eab308';
  return '#f97316';
}

export interface TopicStructureTrainingViewProps {
  stats: TopicStats[];
}

/**
 * 基于 topic_distribution_stats.csv：BERTopic 导出阶段对「微观簇」的 note_count 与簇级 confidence，
 * 反映训练/映射管线输出的主题结构质量（非单条笔记的互动指标）。
 */
export function TopicStructureTrainingView({ stats }: TopicStructureTrainingViewProps) {
  const filtered = useMemo(
    () =>
      stats.filter(
        (s) => s.micro_topic_id >= 0 && !s.macro_topic.includes('噪声') && !s.macro_topic.toLowerCase().includes('outlier'),
      ),
    [stats],
  );

  const macroBars = useMemo(() => {
    const m = new Map<string, { w: number; n: number }>();
    for (const s of filtered) {
      const cur = m.get(s.macro_topic) || { w: 0, n: 0 };
      cur.w += s.confidence * s.note_count;
      cur.n += s.note_count;
      m.set(s.macro_topic, cur);
    }
    return [...m.entries()]
      .map(([fullName, v]) => ({
        fullName,
        name: fullName.length > 8 ? `${fullName.slice(0, 8)}…` : fullName,
        avgConf: v.n > 0 ? v.w / v.n : 0,
        notes: v.n,
        micros: filtered.filter((x) => x.macro_topic === fullName).length,
      }))
      .sort((a, b) => b.notes - a.notes);
  }, [filtered]);

  const microTop = useMemo(
    () =>
      [...filtered]
        .sort((a, b) => b.note_count - a.note_count)
        .slice(0, 28)
        .map((s) => ({
          key: `${s.macro_topic}-${s.micro_topic_id}`,
          label: `T${s.micro_topic_id}`,
          fullName: `${s.macro_topic} · 微观 ${s.micro_topic_id}`,
          note_count: s.note_count,
          confidence: s.confidence,
        })),
    [filtered],
  );

  const kpis = useMemo(() => {
    if (!filtered.length) return null;
    const totalN = filtered.reduce((s, x) => s + x.note_count, 0);
    const w = filtered.reduce((s, x) => s + x.confidence * x.note_count, 0);
    const avg = totalN > 0 ? w / totalN : 0;
    const confs = filtered.map((x) => x.confidence);
    const macroSet = new Set(filtered.map((x) => x.macro_topic));
    return {
      avg,
      microN: filtered.length,
      macroN: macroSet.size,
      minC: Math.min(...confs),
      maxC: Math.max(...confs),
      totalN,
    };
  }, [filtered]);

  if (!stats.length) {
    return (
      <EmptyState
        type="data"
        title="暂无训练侧主题统计"
        description="请确认 content/bertopic_results_optimized 下存在 topic_distribution_stats.csv，并已重新运行开发服务。"
      />
    );
  }

  if (!filtered.length) {
    return (
      <EmptyState
        type="data"
        title="无有效微观簇行"
        description="统计表内可能仅有噪声或离群行，请检查导出结果。"
      />
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold text-gray-800 tracking-tight flex items-center gap-2">
          <Library className="w-5 h-5 text-indigo-500" aria-hidden />
          主题簇统计（训练导出）
        </h3>
        <p className="text-xs text-gray-500 mt-1 max-w-2xl leading-relaxed">
          数据来自 <span className="font-mono text-[11px] text-gray-600">topic_distribution_stats.csv</span>
          ：每个微观簇的文档数与<strong className="text-gray-600">簇级 confidence</strong>
          （BERTopic/映射管线对各簇的代表性评分），用于评估<strong className="text-gray-600">模型划簇结果是否紧凑、可信</strong>
          ，与笔记点赞等互动数据无关。
        </p>
      </div>

      {kpis && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="rounded-2xl border border-indigo-100/80 bg-gradient-to-br from-white to-indigo-50/50 p-4 shadow-sm">
            <p className="text-[11px] font-medium uppercase tracking-wider text-indigo-600/80">簇级加权置信</p>
            <p className="mt-1 text-2xl font-bold tabular-nums text-gray-900">{(kpis.avg * 100).toFixed(1)}%</p>
            <p className="mt-2 text-xs text-gray-500">按簇文档数加权</p>
          </div>
          <div className="rounded-2xl border border-slate-100/80 bg-gradient-to-br from-white to-slate-50/60 p-4 shadow-sm">
            <p className="text-[11px] font-medium uppercase tracking-wider text-slate-600/80">微观簇数量</p>
            <p className="mt-1 text-2xl font-bold tabular-nums text-gray-900">{formatNumber(kpis.microN)}</p>
            <p className="mt-2 text-xs text-gray-500">有效 T&gt;0</p>
          </div>
          <div className="rounded-2xl border border-violet-100/80 bg-gradient-to-br from-white to-violet-50/40 p-4 shadow-sm">
            <p className="text-[11px] font-medium uppercase tracking-wider text-violet-600/80">宏观类数</p>
            <p className="mt-1 text-2xl font-bold tabular-nums text-violet-900">{kpis.macroN}</p>
            <p className="mt-2 text-xs text-gray-500">锚点映射类</p>
          </div>
          <div className="rounded-2xl border border-amber-100/80 bg-gradient-to-br from-white to-amber-50/40 p-4 shadow-sm">
            <p className="text-[11px] font-medium uppercase tracking-wider text-amber-700/80">簇置信区间</p>
            <p className="mt-1 text-lg font-bold tabular-nums text-gray-900">
              {(kpis.minC * 100).toFixed(0)}% – {(kpis.maxC * 100).toFixed(0)}%
            </p>
            <p className="mt-2 text-xs text-gray-500">各微观簇极值</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="rounded-2xl border border-gray-100/90 bg-white/90 p-4 sm:p-5 shadow-lg shadow-gray-200/50">
          <h4 className="text-sm font-semibold text-gray-800 mb-1 flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-indigo-500" aria-hidden />
            各宏观类簇质量（加权平均簇置信度）
          </h4>
          <p className="text-xs text-gray-400 mb-4">对类内各微观簇按文档数加权平均；柱越高表示该宏观下簇划分越一致。</p>
          <div className="h-[min(320px,50vh)] w-full min-h-[260px]">
            <RC.ResponsiveContainer width="100%" height="100%">
              <RC.BarChart data={macroBars} margin={{ top: 8, right: 8, left: 0, bottom: 48 }}>
                <RC.CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <RC.XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} angle={-22} textAnchor="end" height={56} interval={0} />
                <RC.YAxis domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <RC.Tooltip
                  content={({ active, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    const r = payload[0].payload;
                    return (
                      <div className="rounded-xl border border-gray-100 bg-white/95 px-3 py-2 text-xs shadow-lg">
                        <p className="font-semibold text-gray-800">{r.fullName}</p>
                        <p className="text-gray-600 mt-1">加权簇置信 {(r.avgConf * 100).toFixed(1)}%</p>
                        <p className="text-gray-500">文档 {formatNumber(r.notes)} · 微观簇 {r.micros} 个</p>
                      </div>
                    );
                  }}
                />
                <RC.Bar dataKey="avgConf" fill="#6366f1" radius={[6, 6, 0, 0]} maxBarSize={40} name="簇置信" />
              </RC.BarChart>
            </RC.ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-gray-100/90 bg-white/90 p-4 sm:p-5 shadow-lg shadow-gray-200/50">
          <h4 className="text-sm font-semibold text-gray-800 mb-1">大规模微观簇（Top 按文档数）</h4>
          <p className="text-xs text-gray-400 mb-4">条长为簇中文档数；颜色表示该簇代表置信度（绿高橙低）。</p>
          <div className="w-full" style={{ height: Math.max(280, microTop.length * 28) }}>
            <RC.ResponsiveContainer width="100%" height="100%">
              <RC.BarChart layout="vertical" data={microTop} margin={{ top: 4, right: 12, left: 4, bottom: 4 }}>
                <RC.CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                <RC.XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} allowDecimals={false} />
                <RC.YAxis type="category" dataKey="label" width={40} tick={{ fontSize: 11, fill: '#475569' }} />
                <RC.Tooltip
                  content={({ active, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    const r = payload[0].payload;
                    return (
                      <div className="max-w-xs rounded-xl border border-gray-100 bg-white/95 px-3 py-2 text-xs shadow-lg">
                        <p className="font-semibold text-gray-800">{r.fullName}</p>
                        <p className="text-gray-600 mt-1">文档 {formatNumber(r.note_count)}</p>
                        <p className="text-gray-600">簇置信 {(r.confidence * 100).toFixed(1)}%</p>
                      </div>
                    );
                  }}
                />
                <RC.Bar dataKey="note_count" radius={[0, 6, 6, 0]} maxBarSize={22} name="文档数">
                  {microTop.map((e) => (
                    <RC.Cell key={e.key} fill={clusterConfidenceColor(e.confidence)} />
                  ))}
                </RC.Bar>
              </RC.BarChart>
            </RC.ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

export interface ModelQualityDashboardProps {
  topics: Topic[];
  allRecords: TopicRecord[];
}

export function ModelQualityDashboard({ topics, allRecords }: ModelQualityDashboardProps) {
  const records = useMemo(() => {
    if (!topics.length) return [];
    const names = new Set(topics.map((t) => t.name));
    return allRecords.filter((r) => names.has(r.macro_topic));
  }, [topics, allRecords]);

  const hasNoiseFlag = useMemo(
    () => records.some((r) => typeof r.is_noise === 'boolean'),
    [records],
  );
  const noiseCount = useMemo(() => records.filter((r) => r.is_noise === true).length, [records]);

  const histAndCdf = useMemo(() => {
    if (!records.length) return { hist: [] as any[], cdf: [] as any[] };
    const hist = Array.from({ length: HIST_BINS }, (_, i) => {
      const lo = i / HIST_BINS;
      const hi = i === HIST_BINS - 1 ? 1.001 : (i + 1) / HIST_BINS;
      const count = records.filter((r) => r.confidence >= lo && r.confidence < hi).length;
      const mid = (lo + Math.min(1, hi)) / 2;
      return {
        bin: i,
        label: `${(mid * 100).toFixed(0)}%`,
        count,
        density: count / records.length,
      };
    });
    let cum = 0;
    const cdf = hist.map((h) => {
      cum += h.count;
      return { ...h, cdf: records.length ? cum / records.length : 0 };
    });
    return { hist, cdf };
  }, [records]);

  const pieNoiseData = useMemo(() => {
    if (!records.length || !hasNoiseFlag) return [];
    const normal = records.length - noiseCount;
    return [
      { name: '结构化主题', value: normal, key: 'ok' },
      { name: '噪声 / 离群', value: noiseCount, key: 'noise' },
    ].filter((d) => d.value > 0);
  }, [records, hasNoiseFlag, noiseCount]);

  const topicDetailRows = useMemo(() => {
    return topics
      .map((t) => {
        const confs = records
          .filter((r) => r.macro_topic === t.name)
          .map((r) => r.confidence)
          .sort((a, b) => a - b);
        const nNoise = records.filter((r) => r.macro_topic === t.name && r.is_noise === true).length;
        if (confs.length === 0) {
          return {
            id: t.id,
            shortLabel: t.name.length > 10 ? `${t.name.slice(0, 9)}…` : t.name,
            fullName: t.name,
            n: 0,
            median: 0,
            avg: 0,
            p25: 0,
            p75: 0,
            min: 0,
            max: 0,
            noiseN: nNoise,
          };
        }
        return {
          id: t.id,
          shortLabel: t.name.length > 10 ? `${t.name.slice(0, 9)}…` : t.name,
          fullName: t.name,
          n: confs.length,
          median: quantile(confs, 0.5),
          avg: confs.reduce((s, v) => s + v, 0) / confs.length,
          p25: quantile(confs, 0.25),
          p75: quantile(confs, 0.75),
          min: confs[0],
          max: confs[confs.length - 1],
          noiseN: nNoise,
        };
      })
      .sort((a, b) => b.median - a.median);
  }, [topics, records]);

  const kpis = useMemo(() => {
    if (!records.length) return null;
    const avg = records.reduce((s, r) => s + r.confidence, 0) / records.length;
    const high = records.filter((r) => r.confidence >= 0.85).length;
    const low = records.filter((r) => r.confidence > 0 && r.confidence < 0.35).length;
    return {
      avg,
      highRate: (high / records.length) * 100,
      lowRate: (low / records.length) * 100,
      n: records.length,
      macroN: topics.length,
    };
  }, [records, topics.length]);

  if (!records.length) {
    return <EmptyState type="data" title="暂无置信度数据" description="请调整左侧筛选或搜索条件" />;
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2">
        <div>
          <h3 className="text-lg font-semibold text-gray-800 tracking-tight flex items-center gap-2">
            <Layers className="w-5 h-5 text-rose-500" aria-hidden />
            样本推断诊断
          </h3>
          <p className="text-xs text-gray-500 mt-1 max-w-xl">
            每条笔记在<strong className="text-gray-600">推断/映射阶段</strong>得到的样本级置信度（随左侧主题与筛选范围变化）。用于看长尾、噪声与分主题离散度；与「主题簇统计」页的簇级导出指标互补。
          </p>
        </div>
      </div>

      {/* KPI 卡片 */}
      {kpis && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="relative overflow-hidden rounded-2xl border border-rose-100/80 bg-gradient-to-br from-white to-rose-50/60 p-4 shadow-sm shadow-rose-100/40">
            <ShieldCheck className="absolute right-3 top-3 w-8 h-8 text-rose-200/90" aria-hidden />
            <p className="text-[11px] font-medium uppercase tracking-wider text-rose-500/80">平均置信度</p>
            <p className="mt-1 text-2xl font-bold tabular-nums text-gray-900">{(kpis.avg * 100).toFixed(1)}%</p>
            <p className="mt-2 text-xs text-gray-500">聚合 {formatNumber(kpis.n)} 条笔记</p>
          </div>
          <div className="relative overflow-hidden rounded-2xl border border-emerald-100/80 bg-gradient-to-br from-white to-emerald-50/50 p-4 shadow-sm">
            <Activity className="absolute right-3 top-3 w-8 h-8 text-emerald-200/90" aria-hidden />
            <p className="text-[11px] font-medium uppercase tracking-wider text-emerald-600/80">高置信 ≥85%</p>
            <p className="mt-1 text-2xl font-bold tabular-nums text-emerald-700">{kpis.highRate.toFixed(1)}%</p>
            <p className="mt-2 text-xs text-gray-500">质量稳定区占比</p>
          </div>
          <div className="relative overflow-hidden rounded-2xl border border-amber-100/80 bg-gradient-to-br from-white to-amber-50/40 p-4 shadow-sm">
            <AlertTriangle className="absolute right-3 top-3 w-8 h-8 text-amber-200/90" aria-hidden />
            <p className="text-[11px] font-medium uppercase tracking-wider text-amber-700/80">待复核 &lt;35%</p>
            <p className="mt-1 text-2xl font-bold tabular-nums text-amber-800">{kpis.lowRate.toFixed(1)}%</p>
            <p className="mt-2 text-xs text-gray-500">建议人工扫一眼</p>
          </div>
          <div className="relative overflow-hidden rounded-2xl border border-violet-100/80 bg-gradient-to-br from-white to-violet-50/40 p-4 shadow-sm">
            <PieChartIcon className="absolute right-3 top-3 w-8 h-8 text-violet-200/90" aria-hidden />
            <p className="text-[11px] font-medium uppercase tracking-wider text-violet-600/80">宏观主题数</p>
            <p className="mt-1 text-2xl font-bold tabular-nums text-violet-800">{kpis.macroN}</p>
            <p className="mt-2 text-xs text-gray-500">当前列表内</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* 直方图 + CDF */}
        <div className="xl:col-span-2 rounded-2xl border border-gray-100/90 bg-white/90 backdrop-blur-sm p-4 sm:p-5 shadow-lg shadow-gray-200/50">
          <h4 className="text-sm font-semibold text-gray-800 mb-1">置信度密度与累积分位</h4>
          <p className="text-xs text-gray-400 mb-4">柱：各分箱笔记数；线：累计占比（CDF）</p>
          <div className="h-[300px] w-full">
            <RC.ResponsiveContainer width="100%" height="100%">
              <RC.ComposedChart data={histAndCdf.cdf} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                <defs>
                  <linearGradient id="mqHist" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#fb7185" stopOpacity={0.95} />
                    <stop offset="100%" stopColor="#fda4af" stopOpacity={0.25} />
                  </linearGradient>
                </defs>
                <RC.CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <RC.XAxis dataKey="label" tick={{ fontSize: 9, fill: '#94a3b8' }} interval={3} />
                <RC.YAxis yAxisId="left" tick={{ fontSize: 11, fill: '#64748b' }} allowDecimals={false} />
                <RC.YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <RC.Tooltip
                  content={({ active, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    const row = payload[0]?.payload;
                    return (
                      <div className="rounded-xl border border-gray-100 bg-white/95 px-3 py-2.5 text-xs shadow-xl backdrop-blur-md">
                        <p className="font-semibold text-gray-800">分箱中心 ≈ {row.label}</p>
                        <p className="text-gray-600 mt-1">本箱笔记 {formatNumber(row.count)}</p>
                        <p className="text-gray-500">累计占比 {(row.cdf * 100).toFixed(1)}%</p>
                      </div>
                    );
                  }}
                />
                <RC.Bar yAxisId="left" dataKey="count" fill="url(#mqHist)" radius={[6, 6, 0, 0]} maxBarSize={28} />
                <RC.Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="cdf"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                  name="累计占比"
                />
              </RC.ComposedChart>
            </RC.ResponsiveContainer>
          </div>
        </div>

        {/* 噪声占比 */}
        <div className="rounded-2xl border border-gray-100/90 bg-white/90 backdrop-blur-sm p-4 sm:p-5 shadow-lg shadow-gray-200/50 flex flex-col">
          <h4 className="text-sm font-semibold text-gray-800 mb-1">样本结构</h4>
          <p className="text-xs text-gray-400 mb-2">BERTopic 噪声标记（is_noise）</p>
          {!hasNoiseFlag ? (
            <div className="flex-1 flex items-center justify-center text-center text-sm text-gray-400 px-4 py-12">
              当前 CSV 未提供 is_noise 列，无法拆分噪声占比。
            </div>
          ) : pieNoiseData.length === 0 ? (
            <div className="flex-1 flex items-center justify-center text-gray-400 text-sm py-12">无数据</div>
          ) : (
            <>
              <div className="h-[220px] w-full">
                <RC.ResponsiveContainer width="100%" height="100%">
                  <RC.PieChart>
                    <RC.Pie
                      data={pieNoiseData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={52}
                      outerRadius={78}
                      paddingAngle={2}
                      stroke="#fff"
                      strokeWidth={2}
                    >
                      {pieNoiseData.map((entry) => (
                        <RC.Cell key={entry.key} fill={entry.key === 'noise' ? PIE_COLORS.noise : PIE_COLORS.normal} />
                      ))}
                    </RC.Pie>
                    <RC.Tooltip
                      formatter={(value: number, _name: string, p: any) => [
                        `${formatNumber(value)} (${((value / records.length) * 100).toFixed(1)}%)`,
                        p.payload.name,
                      ]}
                      contentStyle={{ borderRadius: 12, border: '1px solid #f1f5f9', fontSize: 12 }}
                    />
                    <RC.Legend wrapperStyle={{ fontSize: 12 }} />
                  </RC.PieChart>
                </RC.ResponsiveContainer>
              </div>
              <p className="text-center text-xs text-gray-500 mt-1">
                噪声样本 {formatNumber(noiseCount)} / {formatNumber(records.length)}
              </p>
            </>
          )}
        </div>
      </div>

      {/* 分主题：中位数与分位宽 */}
      <div className="rounded-2xl border border-gray-100/90 bg-white/90 backdrop-blur-sm p-4 sm:p-5 shadow-lg shadow-gray-200/50">
        <h4 className="text-sm font-semibold text-gray-800 mb-1">各主题置信度中位数</h4>
        <p className="text-xs text-gray-400 mb-4">横向条为中位数；悬停查看 P25–P75 与噪声计数</p>
        <div className="w-full" style={{ height: Math.max(280, topicDetailRows.length * 36) }}>
          <RC.ResponsiveContainer width="100%" height="100%">
            <RC.BarChart layout="vertical" data={topicDetailRows} margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
              <defs>
                <linearGradient id="mqBar" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#fb7185" />
                  <stop offset="100%" stopColor="#a855f7" />
                </linearGradient>
              </defs>
              <RC.CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
              <RC.XAxis type="number" domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11, fill: '#64748b' }} />
              <RC.YAxis type="category" dataKey="shortLabel" width={108} tick={{ fontSize: 11, fill: '#334155' }} />
              <RC.Tooltip
                cursor={{ fill: 'rgba(251, 113, 133, 0.06)' }}
                content={({ active, payload }: any) => {
                  if (!active || !payload?.length) return null;
                  const r = payload[0].payload;
                  return (
                    <div className="max-w-[260px] rounded-xl border border-gray-100 bg-white/95 px-3 py-2.5 text-xs shadow-xl backdrop-blur-md">
                      <p className="font-semibold text-gray-900 leading-snug">{r.fullName}</p>
                      <ul className="mt-2 space-y-1 text-gray-600">
                        <li>笔记数 {formatNumber(r.n)}</li>
                        <li>中位数 {(r.median * 100).toFixed(1)}% · 均值 {(r.avg * 100).toFixed(1)}%</li>
                        <li>
                          P25–P75 {(r.p25 * 100).toFixed(1)}% – {(r.p75 * 100).toFixed(1)}%
                        </li>
                        {hasNoiseFlag && <li className="text-slate-500">噪声条数 {formatNumber(r.noiseN)}</li>}
                      </ul>
                    </div>
                  );
                }}
              />
              <RC.Bar dataKey="median" fill="url(#mqBar)" radius={[0, 8, 8, 0]} maxBarSize={18} name="中位置信度" />
            </RC.BarChart>
          </RC.ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
