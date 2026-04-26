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
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Activity, Layers, ShieldCheck, AlertTriangle, Library } from 'lucide-react';
import type { Topic, TopicRecord, TopicStats } from '../data/topicData';
import { EmptyState } from './common/States';
import { formatNumber } from '../utils/responsive';
import { MACRO_ANCHOR_COLORS, MACRO_ANCHOR_ORDER, macroTopicDisplayColor } from '../theme/macroTopicColors';

const RC = {
  ComposedChart: ComposedChart as any,
  BarChart: BarChart as any,
  Bar: Bar as any,
  Line: Line as any,
  Scatter: Scatter as any,
  ScatterChart: ScatterChart as any,
  ReferenceLine: ReferenceLine as any,
  CartesianGrid: CartesianGrid as any,
  XAxis: XAxis as any,
  YAxis: YAxis as any,
  Tooltip: Tooltip as any,
  Legend: Legend as any,
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
const LOW_CONF_THRESHOLD = 0.35;
const MACRO_CONF_ORDER = MACRO_ANCHOR_ORDER;

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

  const qualityBands = useMemo(() => {
    if (!filtered.length) return [];
    const totalMicro = filtered.length;
    const totalDocs = filtered.reduce((s, x) => s + x.note_count, 0);
    const defs = [
      { key: 'high', label: '高可信簇 (>=85%)', color: '#22c55e', match: (c: number) => c >= 0.85 },
      { key: 'mid', label: '观察簇 (70%-85%)', color: '#eab308', match: (c: number) => c >= 0.7 && c < 0.85 },
      { key: 'low', label: '待复核簇 (<70%)', color: '#f97316', match: (c: number) => c < 0.7 },
    ] as const;
    return defs.map((d) => {
      const rows = filtered.filter((x) => d.match(x.confidence));
      const microN = rows.length;
      const docs = rows.reduce((s, x) => s + x.note_count, 0);
      return {
        key: d.key,
        label: d.label,
        color: d.color,
        microN,
        microPct: totalMicro > 0 ? (microN / totalMicro) * 100 : 0,
        docs,
        docPct: totalDocs > 0 ? (docs / totalDocs) * 100 : 0,
      };
    });
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
          训练侧主题簇诊断
        </h3>
      </div>
      <div className="rounded-2xl border border-gray-100/90 bg-white/90 p-4 sm:p-5 shadow-lg shadow-gray-200/50">
          <h4 className="text-sm font-semibold text-gray-800 mb-1 flex items-center gap-2">
            <Library className="w-4 h-4 text-indigo-500" aria-hidden />
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
        <h4 className="text-sm font-semibold text-gray-800 mb-1">簇质量分层（训练侧）</h4>
        <p className="text-xs text-gray-400 mb-4">
          对微观簇按置信度分层；同时观察“簇数量占比”和“覆盖文档占比”，用于快速定位是否存在低质长尾簇。
        </p>
        <div className="h-[290px] w-full">
          <RC.ResponsiveContainer width="100%" height="100%">
            <RC.BarChart data={qualityBands} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
              <RC.CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <RC.XAxis dataKey="label" tick={{ fontSize: 11, fill: '#64748b' }} interval={0} />
              <RC.YAxis tickFormatter={(v: number) => `${v.toFixed(0)}%`} tick={{ fontSize: 11, fill: '#64748b' }} />
              <RC.Tooltip
                content={({ active, payload }: any) => {
                  if (!active || !payload?.length) return null;
                  const r = payload[0].payload;
                  return (
                    <div className="rounded-xl border border-gray-100 bg-white/95 px-3 py-2 text-xs shadow-lg">
                      <p className="font-semibold text-gray-800">{r.label}</p>
                      <p className="text-gray-600 mt-1">
                        簇数量 {formatNumber(r.microN)}（{r.microPct.toFixed(1)}%）
                      </p>
                      <p className="text-gray-500">
                        覆盖文档 {formatNumber(r.docs)}（{r.docPct.toFixed(1)}%）
                      </p>
                    </div>
                  );
                }}
              />
              <RC.Legend
                formatter={(value: string) => <span className="text-xs text-gray-600">{value}</span>}
                wrapperStyle={{ paddingTop: 6 }}
              />
              <RC.Bar dataKey="microPct" name="簇数量占比" fill="#6366f1" radius={[6, 6, 0, 0]} maxBarSize={42} />
              <RC.Bar dataKey="docPct" name="文档覆盖占比" fill="#fb7185" radius={[6, 6, 0, 0]} maxBarSize={42} />
            </RC.BarChart>
          </RC.ResponsiveContainer>
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

  const validRecords = useMemo(() => {
    return records.filter((r) => {
      const n = (r.macro_topic || '').trim().toLowerCase();
      return !(n.includes('噪声') || n.includes('outlier'));
    });
  }, [records]);

  const confidences = useMemo(
    () => validRecords.map((r) => r.confidence).filter((v) => Number.isFinite(v)).sort((a, b) => a - b),
    [validRecords],
  );

  const confidenceMedian = useMemo(() => quantile(confidences, 0.5), [confidences]);

  const histAndCdf = useMemo(() => {
    if (!confidences.length) return [] as Array<{ label: string; count: number; cdf: number; x: number }>;
    const hist = Array.from({ length: 30 }, (_, i) => {
      const lo = i / 30;
      const hi = i === 29 ? 1.001 : (i + 1) / 30;
      const count = confidences.filter((c) => c >= lo && c < hi).length;
      const x = (lo + Math.min(1, hi)) / 2;
      return {
        label: `${x.toFixed(2)}`,
        count,
        x,
      };
    });
    let cum = 0;
    return hist.map((h) => {
      cum += h.count;
      return { ...h, cdf: cum / confidences.length };
    });
  }, [confidences]);

  const macroBoxData = useMemo(() => {
    const rows: Array<{
      name: string;
      short: string;
      color: string;
      q1: number;
      q3: number;
      median: number;
      iqr: number;
      n: number;
    }> = [];
    for (const macro of MACRO_CONF_ORDER) {
      const vals = validRecords
        .filter((r) => (r.macro_topic || '').trim() === macro)
        .map((r) => r.confidence)
        .sort((a, b) => a - b);
      if (!vals.length) continue;
      const q1 = quantile(vals, 0.25);
      const med = quantile(vals, 0.5);
      const q3 = quantile(vals, 0.75);
      rows.push({
        name: macro,
        short: macro.length > 9 ? `${macro.slice(0, 9)}…` : macro,
        color: MACRO_ANCHOR_COLORS[macro],
        q1,
        q3,
        median: med,
        iqr: Math.max(0.001, q3 - q1),
        n: vals.length,
      });
    }
    return rows;
  }, [validRecords]);

  const topicScatterData = useMemo(() => {
    const grouped = new Map<string, { macro: string; count: number; confs: number[] }>();
    for (const r of validRecords) {
      const key = `${r.micro_topic_id}__${(r.macro_topic || '').trim()}`;
      const cur = grouped.get(key) || { macro: (r.macro_topic || '').trim(), count: 0, confs: [] };
      cur.count += 1;
      cur.confs.push(r.confidence);
      grouped.set(key, cur);
    }
    return [...grouped.entries()].map(([key, v]) => {
      const tid = Number(key.split('__')[0] ?? -1);
      const avg = v.confs.length ? v.confs.reduce((s, x) => s + x, 0) / v.confs.length : 0;
      return {
        tid,
        macro: v.macro,
        count: v.count,
        confidence: avg,
        color: macroTopicDisplayColor(v.macro),
      };
    });
  }, [validRecords]);

  if (!validRecords.length) {
    return <EmptyState type="data" title="暂无有效置信度数据" description="当前筛选下仅有噪声数据或无记录。" />;
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-800 tracking-tight flex items-center gap-2">
          <Layers className="w-5 h-5 text-rose-500" aria-hidden />
          映射置信度评估
        </h3>
      </div>

      <div className="flex flex-col gap-4">
        <div className="rounded-2xl border border-gray-100/90 bg-white/90 backdrop-blur-sm p-4 sm:p-5 shadow-lg shadow-gray-200/50">
          <h4 className="text-sm font-semibold text-gray-800 mb-1">全局置信度分布</h4>
          <div className="h-[300px] w-full">
            <RC.ResponsiveContainer width="100%" height="100%">
              <RC.ComposedChart data={histAndCdf} margin={{ top: 8, right: 8, left: 0, bottom: 20 }}>
                <RC.CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <RC.XAxis
                  type="number"
                  dataKey="x"
                  domain={[0, 1]}
                  ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                  tickFormatter={(v: number | string) => {
                    const n = typeof v === 'number' ? v : parseFloat(String(v));
                    return Number.isFinite(n) ? n.toFixed(2) : '';
                  }}
                  tick={{ fontSize: 10, fill: '#64748b' }}
                />
                <RC.YAxis yAxisId="left" tick={{ fontSize: 10, fill: '#64748b' }} allowDecimals={false} />
                <RC.YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <RC.Tooltip
                  content={({ active, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    const row = payload[0]?.payload;
                    if (!row) return null;
                    return (
                      <div className="rounded-xl border border-gray-100 bg-white/95 px-3 py-2 text-xs shadow-lg">
                        <div className="font-semibold text-gray-800">置信度中心 {row.x.toFixed(2)}</div>
                        <div className="text-gray-600 mt-1">样本 {formatNumber(row.count)}</div>
                        <div className="text-gray-500">CDF {(row.cdf * 100).toFixed(1)}%</div>
                      </div>
                    );
                  }}
                />
                <RC.ReferenceLine yAxisId="left" x={confidenceMedian} stroke="#ef4444" strokeDasharray="6 4" />
                <RC.ReferenceLine yAxisId="left" x={LOW_CONF_THRESHOLD} stroke="#f59e0b" strokeDasharray="2 3" />
                <RC.Bar yAxisId="left" dataKey="count" fill="#5ba6d9" radius={[4, 4, 0, 0]} maxBarSize={18} />
                <RC.Line yAxisId="right" dataKey="cdf" stroke="#2563eb" strokeWidth={2} dot={false} />
              </RC.ComposedChart>
            </RC.ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-gray-100/90 bg-white/90 backdrop-blur-sm p-4 sm:p-5 shadow-lg shadow-gray-200/50">
          <h4 className="text-sm font-semibold text-gray-800 mb-1">各类别置信度对比</h4>
          <div className="h-[300px] w-full">
            {macroBoxData.length === 0 ? (
              <div className="h-full flex items-center justify-center text-sm text-gray-400">暂无宏观类别置信度数据</div>
            ) : (
              <RC.ResponsiveContainer width="100%" height="100%">
                <RC.ComposedChart data={macroBoxData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                  <RC.CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <RC.XAxis dataKey="short" tick={{ fontSize: 10, fill: '#64748b' }} />
                  <RC.YAxis yAxisId="left" domain={[0.3, 1]} tick={{ fontSize: 10, fill: '#64748b' }} />
                  <RC.Tooltip
                    content={({ active, payload }: any) => {
                      if (!active || !payload?.length) return null;
                      const row = payload[0]?.payload;
                      if (!row) return null;
                      return (
                        <div className="rounded-xl border border-gray-100 bg-white/95 px-3 py-2 text-xs shadow-lg">
                          <div className="font-semibold text-gray-800">{row.name}</div>
                          <div className="text-gray-600 mt-1">中位数 {row.median.toFixed(3)}</div>
                          <div className="text-gray-500">P25-P75: {row.q1.toFixed(3)} ~ {row.q3.toFixed(3)}</div>
                          <div className="text-gray-500">样本数 {formatNumber(row.n)}</div>
                        </div>
                      );
                    }}
                  />
                  <RC.ReferenceLine yAxisId="left" y={LOW_CONF_THRESHOLD} stroke="#f59e0b" strokeDasharray="2 3" />
                  <RC.Bar yAxisId="left" dataKey="q1" stackId="a" fill="transparent" />
                  <RC.Bar yAxisId="left" dataKey="iqr" stackId="a" radius={[4, 4, 4, 4]}>
                    {macroBoxData.map((e) => (
                      <RC.Cell key={e.name} fill={e.color} fillOpacity={0.72} />
                    ))}
                  </RC.Bar>
                  <RC.Line yAxisId="left" type="monotone" dataKey="median" stroke="#1f2937" strokeWidth={1.8} dot={{ r: 3 }} />
                </RC.ComposedChart>
              </RC.ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-gray-100/90 bg-white/90 backdrop-blur-sm p-4 sm:p-5 shadow-lg shadow-gray-200/50">
          <h4 className="text-sm font-semibold text-gray-800 mb-1">主题规模 vs 置信度</h4>
          <div className="h-[300px] w-full">
            <RC.ResponsiveContainer width="100%" height="100%">
              <RC.ScatterChart margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <RC.CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <RC.XAxis type="number" dataKey="count" name="主题文档数量" tick={{ fontSize: 10, fill: '#64748b' }} />
                <RC.YAxis yAxisId="left" type="number" dataKey="confidence" domain={[0.3, 1]} name="映射置信度" tick={{ fontSize: 10, fill: '#64748b' }} />
                <RC.Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ active, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    const row = payload[0]?.payload;
                    if (!row) return null;
                    return (
                      <div className="rounded-xl border border-gray-100 bg-white/95 px-3 py-2 text-xs shadow-lg">
                        <div className="font-semibold text-gray-800">T{row.tid} · {row.macro}</div>
                        <div className="text-gray-600 mt-1">文档数 {formatNumber(row.count)}</div>
                        <div className="text-gray-500">置信度 {row.confidence.toFixed(3)}</div>
                      </div>
                    );
                  }}
                />
                <RC.ReferenceLine yAxisId="left" y={LOW_CONF_THRESHOLD} stroke="#f59e0b" strokeDasharray="2 3" />
                <RC.Scatter yAxisId="left" data={topicScatterData}>
                  {topicScatterData.map((p) => (
                    <RC.Cell key={`${p.tid}-${p.macro}`} fill={p.color} fillOpacity={0.78} />
                  ))}
                </RC.Scatter>
              </RC.ScatterChart>
            </RC.ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
