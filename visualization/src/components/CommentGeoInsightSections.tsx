import { Network } from 'lucide-react'
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
import type { LiveCommentBundle } from '../data/commentLiveData'

const SENT = { pos: '#10b981', neu: '#94a3b8', neg: '#f43f5e' }
const BarChartAny = BarChart as any
const BarAny = Bar as any
const XAxisAny = XAxis as any
const YAxisAny = YAxis as any
const CartesianGridAny = CartesianGrid as any
const TooltipAny = Tooltip as any
const LegendAny = Legend as any
const ResponsiveContainerAny = ResponsiveContainer as any

type Props = {
  bundle: LiveCommentBundle
  /** 嵌在「情感分布」内时展示顶部说明，并微调小节文案 */
  embedded?: boolean
}

export function CommentGeoInsightSections({ bundle, embedded }: Props) {
  const ipRows = bundle.ipDistribution
  const ipCaption = embedded
    ? '堆叠条为各地区预测评论的正向、中性、负向条数（与上方全库极性同源）。'
    : '堆叠条为各地区预测评论的正向、中性、负向条数。'

  return (
    <div className="space-y-6 md:space-y-10">
      {embedded && (
        <div className="rounded-xl border border-cyan-200/70 bg-gradient-to-r from-cyan-50/90 to-teal-50/50 px-4 py-3 text-sm text-cyan-950">
          <div className="font-semibold text-cyan-900">IP 归属与地域</div>
          <p className="text-xs text-slate-600 mt-1.5 leading-relaxed">
            IP 归属省市与情感极性来自全量预测中的 IP 字段聚合。
          </p>
        </div>
      )}

      <section className="rounded-2xl border border-slate-200/80 bg-white/90 p-5 md:p-6 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-800 flex items-center gap-2 mb-1">
          <Network className="w-4 h-4 text-cyan-600" />
          IP 归属与情感结构
        </h2>
        <p className="text-xs text-slate-500 mb-4">{ipCaption}</p>
        <div className="h-[min(420px,50vh)] w-full min-w-0">
          <ResponsiveContainerAny width="100%" height="100%">
            <BarChartAny data={ipRows} layout="vertical" margin={{ left: 8, right: 16 }}>
              <CartesianGridAny strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
              <XAxisAny type="number" tick={{ fontSize: 11 }} />
              <YAxisAny type="category" dataKey="name" width={96} tick={{ fontSize: 10 }} />
              <TooltipAny />
              <LegendAny wrapperStyle={{ fontSize: 12 }} />
              <BarAny dataKey="positive" stackId="ip" fill={SENT.pos} name="正向" />
              <BarAny dataKey="neutral" stackId="ip" fill={SENT.neu} name="中性" />
              <BarAny dataKey="negative" stackId="ip" fill={SENT.neg} name="负向" radius={[0, 6, 6, 0]} />
            </BarChartAny>
          </ResponsiveContainerAny>
        </div>
      </section>
    </div>
  )
}
