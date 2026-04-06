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
import { Activity, Layers, PieChart as PieChartIcon, ShieldCheck, AlertTriangle } from 'lucide-react';
import type { Topic, TopicRecord } from '../data/topicData';
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
            模型效果与置信度
          </h3>
          <p className="text-xs text-gray-500 mt-1 max-w-xl">
            基于当前筛选主题的笔记集合：分布、累积分位与分主题指标。数据来自 BERTopic 与 CSV 中的置信度及噪声标注。
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
