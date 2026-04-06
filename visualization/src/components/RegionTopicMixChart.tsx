/* eslint-disable @typescript-eslint/no-explicit-any */
import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { formatNumber } from '@/utils/responsive'

const XAxisAny = XAxis as any
const YAxisAny = YAxis as any
const CartesianGridAny = CartesianGrid as any
const TooltipAny = Tooltip as any
const LegendAny = Legend as any
const BarChartAny = BarChart as any
const BarAny = Bar as any
const ResponsiveContainerAny = ResponsiveContainer as any

const TOPIC_COLORS = ['#e11d48', '#7c3aed', '#0891b2', '#16a34a', '#d97706']

export interface MixCell {
  topicId: number
  topicName: string
  cnt: number
}

export interface MixRow {
  region: string
  total: number
  cells: MixCell[]
}

interface RegionTopicMixChartProps {
  rows: MixRow[]
  topicOrder: { id: number; name: string }[]
}

function MixTooltip({
  active,
  payload,
  label,
  mode,
}: {
  active?: boolean
  payload?: any[]
  label?: string
  mode: 'count' | 'percent'
}) {
  if (!active || !payload?.length) return null
  const parts = payload.filter((p: any) => Number(p.value) > 0)
  const sum = parts.reduce((s: number, p: any) => s + Number(p.value), 0)
  return (
    <div className="rounded-xl border border-rose-100 bg-white/95 px-3 py-2.5 shadow-lg backdrop-blur-sm min-w-[210px]">
      <p className="font-semibold text-gray-900 text-sm mb-2 border-b border-gray-100 pb-1">{label}</p>
      <ul className="space-y-1.5 text-xs">
        {parts.map((p: any) => (
          <li key={String(p.dataKey)} className="flex justify-between gap-4">
            <span className="flex items-center gap-1.5 text-gray-600">
              <span className="h-2 w-2 rounded-full shrink-0" style={{ background: p.color }} />
              {p.name}
            </span>
            <span className="tabular-nums font-medium text-gray-800">
              {mode === 'percent' ? `${Number(p.value).toFixed(1)}%` : formatNumber(Math.round(p.value))}
            </span>
          </li>
        ))}
      </ul>
      <p className="text-[11px] text-gray-400 mt-2 pt-1 border-t border-gray-50">
        {mode === 'percent'
          ? `条段合计 ${sum.toFixed(0)}%（四舍五入误差）`
          : `条段合计 ${formatNumber(Math.round(sum))} 篇`}
      </p>
    </div>
  )
}

export function RegionTopicMixChart({ rows, topicOrder }: RegionTopicMixChartProps) {
  const [mode, setMode] = useState<'count' | 'percent'>('percent')

  const dataKeys = useMemo(
    () => topicOrder.map((t) => ({ key: `t_${t.id}`, name: t.name, id: t.id })),
    [topicOrder],
  )

  const chartData = useMemo(() => {
    return rows.map((r) => {
      const row: Record<string, string | number> = { region: r.region, total: r.total }
      const byId = new Map(r.cells.map((c) => [c.topicId, c.cnt]))
      const denom = r.total > 0 ? r.total : 1
      for (const col of dataKeys) {
        const c = byId.get(col.id) ?? 0
        row[col.key] = mode === 'percent' ? (c / denom) * 100 : c
      }
      return row
    })
  }, [rows, dataKeys, mode])

  const chartHeight = Math.max(340, rows.length * 46 + 108)

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white/90 p-4 sm:p-5 md:p-6 shadow-md shadow-slate-200/40">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">各地区主题构成</h3>
          <p className="text-sm text-gray-500 mt-0.5">
            {mode === 'percent'
              ? '横向堆叠条表示 100%，每段为该主题在当地笔记中的占比'
              : '横向堆叠条总长度 ≈ 该地区笔记总数，便于对比各地体量'}
          </p>
        </div>
        <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50/80 p-0.5 shrink-0">
          <button
            type="button"
            onClick={() => setMode('percent')}
            className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              mode === 'percent'
                ? 'bg-white text-rose-600 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            占比视图
          </button>
          <button
            type="button"
            onClick={() => setMode('count')}
            className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              mode === 'count'
                ? 'bg-white text-rose-600 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            篇数视图
          </button>
        </div>
      </div>

      <ResponsiveContainerAny width="100%" height={chartHeight}>
        <BarChartAny
          layout="vertical"
          data={chartData}
          margin={{ top: 4, right: 20, left: 4, bottom: 4 }}
        >
          <CartesianGridAny strokeDasharray="3 3" stroke="#f1f5f9" horizontal />
          <XAxisAny
            type="number"
            tick={{ fontSize: 11, fill: '#64748b' }}
            domain={mode === 'percent' ? [0, 100] : [0, 'auto']}
            tickFormatter={(v: number) =>
              mode === 'percent' ? `${Math.round(v)}%` : formatNumber(v)
            }
          />
          <YAxisAny
            type="category"
            dataKey="region"
            width={76}
            tick={{ fontSize: 12, fill: '#334155' }}
            axisLine={false}
            tickLine={false}
          />
          <TooltipAny content={(props: any) => <MixTooltip {...props} mode={mode} />} />
          <LegendAny
            wrapperStyle={{ paddingTop: 8 }}
            formatter={(value: string) => <span className="text-xs text-gray-600">{value}</span>}
          />
          {dataKeys.map((col, i) => (
            <BarAny
              key={col.key}
              dataKey={col.key}
              name={col.name}
              stackId="mix"
              fill={TOPIC_COLORS[i % TOPIC_COLORS.length]}
              radius={i === dataKeys.length - 1 ? [0, 6, 6, 0] : [0, 0, 0, 0]}
              maxBarSize={32}
            />
          ))}
        </BarChartAny>
      </ResponsiveContainerAny>
    </div>
  )
}
