import { useEffect, useMemo, useRef } from 'react'
import * as echarts from 'echarts/core'
import { LineChart as EChartsLineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { EvaluationMetrics } from '../data/commentData'
import type { EvalVizPayload } from '../data/commentLiveData'
import {
  EVAL_CLASS_LABELS,
  EVAL_COLORS,
  buildConfidenceHistogram,
  maxSoftmaxPerRow,
  perClassMetricsFromLabels,
  rocMicroOvR,
  rocOvrClass,
  weightedF1FromClasses,
} from '../utils/evalVizMath'

echarts.use([EChartsLineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

const BarChartAny = BarChart as any
const BarAny = Bar as any
const XAxisAny = XAxis as any
const YAxisAny = YAxis as any
const CartesianGridAny = CartesianGrid as any
const TooltipAny = Tooltip as any
const LegendAny = Legend as any
const ResponsiveContainerAny = ResponsiveContainer as any
const LabelListAny = LabelList as any

function barLabelFmt(value: unknown) {
  const v = Number(value)
  if (!Number.isFinite(v)) return ''
  return `${(v * 100).toFixed(1)}%`
}

type ClassRow = { name: string; precision: number; recall: number; f1: number; support: number }

function MetricsBarChart({ rows }: { rows: ClassRow[] }) {
  const slice = rows.slice(0, 3)
  const data = slice.map((r, i) => ({
    name: EVAL_CLASS_LABELS[i] ?? r.name,
    Precision: r.precision,
    Recall: r.recall,
    'F1-Score': r.f1,
  }))
  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-4 md:p-5 shadow-sm">
      <h4 className="text-sm font-semibold text-slate-800 mb-4">各类别 Precision / Recall / F1</h4>

      <div className="h-[min(520px,52vh)] min-h-[380px] w-full min-w-0">
        <ResponsiveContainerAny width="100%" height="100%">
          <BarChartAny data={data} margin={{ top: 28, right: 12, left: 8, bottom: 8 }} barGap={4}>
            <CartesianGridAny strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxisAny dataKey="name" tick={{ fontSize: 12 }} height={36} />
            <YAxisAny domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} width={48} tick={{ fontSize: 11 }} />
            <TooltipAny
              formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
              contentStyle={{ borderRadius: 12, border: '1px solid #e2e8f0' }}
            />
            <LegendAny wrapperStyle={{ fontSize: 13 }} />
            <BarAny dataKey="Precision" fill="#4C72B0" radius={[3, 3, 0, 0]} maxBarSize={56}>
              <LabelListAny
                dataKey="Precision"
                position="top"
                formatter={barLabelFmt}
                style={{ fontSize: 11, fill: '#334155', fontWeight: 600 }}
              />
            </BarAny>
            <BarAny dataKey="Recall" fill="#55A868" radius={[3, 3, 0, 0]} maxBarSize={56}>
              <LabelListAny
                dataKey="Recall"
                position="top"
                formatter={barLabelFmt}
                style={{ fontSize: 11, fill: '#334155', fontWeight: 600 }}
              />
            </BarAny>
            <BarAny dataKey="F1-Score" fill="#C44E52" radius={[3, 3, 0, 0]} maxBarSize={56}>
              <LabelListAny
                dataKey="F1-Score"
                position="top"
                formatter={barLabelFmt}
                style={{ fontSize: 11, fill: '#334155', fontWeight: 600 }}
              />
            </BarAny>
          </BarChartAny>
        </ResponsiveContainerAny>
      </div>
    </div>
  )
}

/** 测试集整体三项指标：报告中的 Accuracy / Macro-F1 + 由各类 F1×support 加权的 Weighted-F1 */
function OverallTestMetricsPanel({
  metrics,
  chartRows,
}: {
  metrics: EvaluationMetrics
  chartRows: ClassRow[]
}) {
  const weightedF1 = useMemo(() => weightedF1FromClasses(chartRows), [chartRows])
  const items = [
    {
      label: 'Accuracy',
      raw: metrics.accuracy,
      barClass: 'bg-gradient-to-r from-indigo-500 to-blue-500',
      trackClass: 'bg-indigo-100/90',
    },
    {
      label: 'Macro-F1',
      raw: metrics.macroF1,
      barClass: 'bg-gradient-to-r from-emerald-500 to-teal-500',
      trackClass: 'bg-emerald-100/90',
    },
    {
      label: 'Weighted-F1',
      raw: weightedF1,
      barClass: 'bg-gradient-to-r from-violet-500 to-purple-500',
      trackClass: 'bg-violet-100/90',
    },
  ]

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-4 md:p-5 shadow-sm overflow-hidden">
      <h4 className="text-sm font-semibold text-slate-800 mb-1">测试集整体指标</h4>
      <p className="text-[11px] text-slate-500 mb-5 leading-relaxed">
        Accuracy / Macro-F1 与{' '}
        <span className="font-mono text-[10px] text-slate-600">evaluation_report.txt</span> 一致；Weighted-F1 为各类 F1 按
        support 加权（与 sklearn{' '}
        <span className="font-mono text-[10px]">average=&apos;weighted&apos;</span> 一致）。
      </p>
      <div className="space-y-6">
        {items.map((it) => {
          const pct = Math.min(100, Math.max(0, it.raw * 100))
          return (
            <div key={it.label}>
              <div className="flex items-end justify-between gap-3 mb-2">
                <span className="text-xs font-semibold text-slate-600 tracking-tight">{it.label}</span>
                <span className="font-mono text-lg font-bold tabular-nums text-slate-900 tracking-tight">
                  {it.raw.toFixed(4)}
                </span>
              </div>
              <div className={`h-2.5 rounded-full overflow-hidden ${it.trackClass}`}>
                <div
                  className={`h-full rounded-full ${it.barClass} transition-[width] duration-500 ease-out shadow-sm`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="mt-1 text-right text-[11px] tabular-nums text-slate-500">{pct.toFixed(2)}%</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function ConfidenceHistogramChart({
  trueLabels,
  predLabels,
  probs,
}: {
  trueLabels: number[]
  predLabels: number[]
  probs: number[][]
}) {
  const hist = useMemo(
    () => buildConfidenceHistogram(trueLabels, predLabels, probs, 40),
    [trueLabels, predLabels, probs],
  )
  const meanConf = useMemo(() => {
    const m = maxSoftmaxPerRow(probs)
    if (!m.length) return 0
    return m.reduce((a, b) => a + b, 0) / m.length
  }, [probs])

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-4 md:p-5 shadow-sm">
      <h4 className="text-sm font-semibold text-slate-800 mb-1">预测置信度分布（正确 vs 错误）</h4>
      <p className="text-[11px] text-slate-500 mb-4">对应 evaluate_model.plot_confidence_distribution · 平均置信度 {meanConf.toFixed(3)}</p>
      <div className="h-[320px] w-full min-w-0">
        <ResponsiveContainerAny width="100%" height="100%">
          <BarChartAny data={hist} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
            <CartesianGridAny strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxisAny dataKey="binLabel" tick={{ fontSize: 9 }} interval={3} angle={-35} textAnchor="end" height={52} />
            <YAxisAny width={44} />
            <TooltipAny />
            <LegendAny wrapperStyle={{ fontSize: 12 }} />
            <BarAny dataKey="correct" name="预测正确" fill="#55A868" fillOpacity={0.72} />
            <BarAny dataKey="wrong" name="预测错误" fill="#C44E52" fillOpacity={0.72} />
          </BarChartAny>
        </ResponsiveContainerAny>
      </div>
    </div>
  )
}

export function RocCurvesEcharts({ trueLabels, probs }: { trueLabels: number[]; probs: number[][] }) {
  const rocRef = useRef<HTMLDivElement>(null)
  const lineSeries = useMemo(() => {
    const out: {
      name: string
      color: string
      dashed?: boolean
      thick?: boolean
      pts: [number, number][]
    }[] = []
    for (let c = 0; c < 3; c++) {
      const { fpr, tpr, auc } = rocOvrClass(trueLabels, probs, c)
      const aucStr = Number.isFinite(auc) ? auc.toFixed(4) : '—'
      out.push({
        name: `${EVAL_CLASS_LABELS[c]} (AUC=${aucStr})`,
        color: EVAL_COLORS[c]!,
        pts: fpr.map((x, i) => [x, tpr[i]!] as [number, number]),
      })
    }
    const micro = rocMicroOvR(trueLabels, probs)
    const mStr = Number.isFinite(micro.auc) ? micro.auc.toFixed(4) : '—'
    out.push({
      name: `微平均 (AUC=${mStr})`,
      color: '#64748b',
      dashed: true,
      thick: true,
      pts: micro.fpr.map((x, i) => [x, micro.tpr[i]!] as [number, number]),
    })
    return out
  }, [trueLabels, probs])

  useEffect(() => {
    const el = rocRef.current
    if (!el || lineSeries.length === 0) return
    const chart = echarts.init(el, undefined, { renderer: 'canvas' })
    chart.setOption({
      tooltip: { trigger: 'axis', confine: true },
      legend: {
        orient: 'vertical',
        right: 8,
        bottom: 40,
        itemWidth: 22,
        itemHeight: 10,
        textStyle: { fontSize: 10, color: '#334155' },
        backgroundColor: 'rgba(255,255,255,0.92)',
        borderColor: '#e2e8f0',
        borderWidth: 1,
        padding: 8,
        borderRadius: 6,
      },
      grid: { left: 56, right: 148, top: 16, bottom: 40 },
      xAxis: {
        type: 'value',
        name: 'False Positive Rate',
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: { fontSize: 11, color: '#475569' },
        min: 0,
        max: 1,
        splitLine: { lineStyle: { type: 'dashed', color: '#e2e8f0' } },
      },
      yAxis: {
        type: 'value',
        name: 'True Positive Rate',
        nameLocation: 'middle',
        nameGap: 40,
        nameTextStyle: { fontSize: 11, color: '#475569' },
        min: 0,
        max: 1,
        splitLine: { lineStyle: { type: 'dashed', color: '#e2e8f0' } },
      },
      series: [
        {
          name: '随机分类',
          type: 'line',
          data: [
            [0, 0],
            [1, 1],
          ],
          lineStyle: { type: 'dashed', color: '#94a3b8', width: 1 },
          symbol: 'none',
          silent: true,
          z: 1,
        },
        ...lineSeries.map((s) => ({
          name: s.name,
          type: 'line' as const,
          step: 'end' as const,
          data: s.pts,
          lineStyle: {
            width: s.thick ? 2.5 : 1.8,
            color: s.color,
            type: s.dashed ? ('dashed' as const) : 'solid',
          },
          symbol: 'none',
          emphasis: { focus: 'series' as const },
          z: 2,
        })),
      ],
    })
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(el)
    return () => {
      ro.disconnect()
      chart.dispose()
    }
  }, [lineSeries])

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-4 md:p-5 shadow-sm">
      <h4 className="text-sm font-semibold text-slate-800 mb-1">ROC 曲线（One-vs-Rest）</h4>
      <p className="text-[11px] text-slate-500 mb-4">
        与 <span className="font-mono text-[10px]">comment/results/06_roc_curves.png</span> 同源算法（OvR + 微平均，阶梯连线）
      </p>
      <div ref={rocRef} className="h-[400px] w-full min-w-0" />
    </div>
  )
}

function isEvalVizUsable(v: EvalVizPayload | null): v is EvalVizPayload {
  return (
    v != null &&
    v.true_labels.length > 0 &&
    v.true_labels.length === v.pred_labels.length &&
    v.probs.length === v.true_labels.length
  )
}

export function MacroSentimentEvalCharts({
  metrics,
  classData,
  evalViz,
}: {
  metrics: EvaluationMetrics
  classData: ClassRow[]
  evalViz: EvalVizPayload | null
}) {
  /** 与 ROC / 置信度图一致：优先用 eval_viz_payload.json 的 true/pred 逐类重算；否则退回 evaluation_report 解析结果 */
  const chartRows = useMemo(() => {
    if (isEvalVizUsable(evalViz)) {
      return perClassMetricsFromLabels(evalViz.true_labels, evalViz.pred_labels)
    }
    return classData
  }, [evalViz, classData])

  return (
    <div className="space-y-6 pb-2">
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 items-stretch">
        <MetricsBarChart rows={chartRows} />
        <OverallTestMetricsPanel metrics={metrics} chartRows={chartRows} />
      </div>
    </div>
  )
}
