/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useState, useMemo, useRef } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  LineChart,
  Line,
} from 'recharts'
import { Download, X, CheckSquare, Square, Info, ChevronDown, FileSpreadsheet, FileText, Table2 } from 'lucide-react'
import { Topic, TopicRecord, Note } from '../data/topicData'
import { formatNumber } from '../utils/responsive'

// recharts components typed as any to avoid TS version conflicts
const XAxisAny = XAxis as any
const YAxisAny = YAxis as any
const CartesianGridAny = CartesianGrid as any
const TooltipAny = Tooltip as any
const LegendAny = Legend as any
const BarChartAny = BarChart as any
const BarAny = Bar as any
const ResponsiveContainerAny = ResponsiveContainer as any
const RadarChartAny = RadarChart as any
const RadarAny = Radar as any
const PolarGridAny = PolarGrid as any
const PolarAngleAxisAny = PolarAngleAxis as any
const PolarRadiusAxisAny = PolarRadiusAxis as any
const LineChartAny = LineChart as any
const LineAny = Line as any

type CompareIntervalMetric = 'likes' | 'comments' | 'collects' | 'shares'

const INTERVAL_METRIC_TABS: { id: CompareIntervalMetric; label: string }[] = [
  { id: 'likes', label: '点赞' },
  { id: 'comments', label: '评论' },
  { id: 'collects', label: '收藏' },
  { id: 'shares', label: '分享' },
]

function noteValueForInterval(n: Note, metric: CompareIntervalMetric): number {
  switch (metric) {
    case 'likes':
      return n.likes
    case 'comments':
      return n.comments
    case 'collects':
      return n.collects
    case 'shares':
      return n.shares
  }
}

/** 与对比页折线图一致：按单条笔记在某一互动维度上的计数落入分段 */
const METRIC_INTERVAL_BUCKETS = ['0-100', '101-500', '501-1k', '1k-5k', '5k+'] as const

const METRIC_BUCKET_LABELS: Record<(typeof METRIC_INTERVAL_BUCKETS)[number], string> = {
  '0-100': '0–100',
  '101-500': '101–500',
  '501-1k': '501–1k',
  '1k-5k': '1k–5k',
  '5k+': '5k+',
}

function countNotesInMetricBucket(
  notes: Note[],
  bucket: (typeof METRIC_INTERVAL_BUCKETS)[number],
  metric: CompareIntervalMetric,
): number {
  const v = (n: Note) => noteValueForInterval(n, metric)
  switch (bucket) {
    case '0-100':
      return notes.filter(n => v(n) <= 100).length
    case '101-500':
      return notes.filter(n => v(n) > 100 && v(n) <= 500).length
    case '501-1k':
      return notes.filter(n => v(n) > 500 && v(n) <= 1000).length
    case '1k-5k':
      return notes.filter(n => v(n) > 1000 && v(n) <= 5000).length
    case '5k+':
      return notes.filter(n => v(n) > 5000).length
  }
}

function exportCsvBasename(base: string, suffix: string): string {
  const d = new Date()
  const ts = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}-${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
  return `${base}-${suffix}-${ts}.csv`
}

// ================================================================
// 类型定义
// ================================================================

interface CompareViewProps {
  topics: Topic[]
  onBack: () => void
}

interface ExportButtonProps {
  topics: Topic[]
  notes?: Note[]
  filename?: string
}

// ================================================================
// 工具函数
// ================================================================

/** BERTopic 噪声/离群主题，默认不参与对比勾选 */
function isNoiseTopicName(name: string): boolean {
  const n = name.trim().toLowerCase()
  return n.includes('噪声') || n.includes('outlier')
}

/** 默认勾选前 5 个非噪声主题（不足 5 个则全选非噪声；无非噪声时退回前若干个以保证至少 1 个） */
function defaultCompareSelection(topics: Topic[]): Set<number> {
  const nonNoise = topics.filter((t) => !isNoiseTopicName(t.name))
  const picked = nonNoise.slice(0, 5)
  if (picked.length > 0) return new Set(picked.map((t) => t.id))
  const fallback = topics.slice(0, Math.min(5, topics.length))
  return new Set(fallback.map((t) => t.id))
}

function toNotes(topic: Topic): Note[] {
  return topic.rawRecords.map((r) => ({
    id: r.note_id,
    title: r.title || '',
    content: r.content || '',
    likes: r.liked_count || 0,
    comments: r.comment_count || 0,
    shares: r.share_count || 0,
    collects: r.collected_count || 0,
    discusses: 0,
    topicId: topic.id,
    topicName: topic.name,
    macroTopicName: topic.name,
    microTopicId: r.micro_topic_id,
    keywords: r.micro_topic_keywords.split(',').map((k: string) => k.trim()).filter(Boolean),
    createdAt: '',
    ipLocation: r.ip_location || '',
    noteUrl: r.note_url || '',
    confidence: r.confidence,
    commentList: [],
  }))
}

function downloadCSV(data: Record<string, any>[], filename: string) {
  if (!data.length) return
  const headers = Object.keys(data[0])
  const rows = data.map(row =>
    headers.map(h => {
      const val = String(row[h] ?? '')
      return val.includes(',') || val.includes('"') || val.includes('\n')
        ? `"${val.replace(/"/g, '""')}"`
        : val
    }).join(',')
  )
  const csv = [headers.join(','), ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// ================================================================
// 悬停详情气泡
// ================================================================

export function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: any[]; label?: string }) {
  if (!active || !payload?.length) return null
  const row0 = payload[0]?.payload as { fullName?: string; name?: string } | undefined
  const title = row0?.fullName || row0?.name || label
  return (
    <div className="bg-white/95 backdrop-blur-md rounded-xl shadow-xl border border-rose-100/60 p-3 min-w-[160px] shadow-rose-100/30">
      <p className="font-semibold text-gray-800 text-sm mb-2 leading-snug">{title}</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center justify-between gap-3 text-xs mb-1">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
            <span className="text-gray-500">{entry.name}</span>
          </div>
          <span className="font-medium text-gray-700">{formatNumber(entry.value)}</span>
        </div>
      ))}
    </div>
  )
}

// ================================================================
// 主题对比视图
// ================================================================

export function CompareView({ topics, onBack }: CompareViewProps) {
  const [selected, setSelected] = useState<Set<number>>(() => defaultCompareSelection(topics))
  const [showNoteModal, setShowNoteModal] = useState<{ topic: Topic; notes: Note[] } | null>(null)
  const [intervalMetric, setIntervalMetric] = useState<CompareIntervalMetric>('likes')

  const selectedTopics = topics.filter(t => selected.has(t.id))
  const COLORS = ['#f43f5e', '#8b5cf6', '#06b6d4', '#22c55e', '#f59e0b', '#ec4899']

  const toggleTopic = (id: number) => {
    const next = new Set(selected)
    if (next.has(id)) {
      if (next.size > 1) next.delete(id)
    } else {
      if (next.size < 6) next.add(id)
    }
    setSelected(next)
  }

  const topicShortLabels = useMemo(
    () =>
      selectedTopics.map(t =>
        t.name.length > 8 ? `${t.name.slice(0, 8)}…` : t.name,
      ),
    [selectedTopics],
  )

  // ── 雷达图：四维互动在「当前勾选组内」归一化到 0–100，便于看形状差异 ──
  const radarData = useMemo(() => {
    if (selectedTopics.length === 0) return []
    const dims = [
      { key: 'avgLikes' as const, label: '均赞' },
      { key: 'avgComments' as const, label: '均评' },
      { key: 'avgCollects' as const, label: '均藏' },
      { key: 'avgShares' as const, label: '均享' },
    ]
    const maxes = {
      avgLikes: Math.max(...selectedTopics.map(t => t.avgLikes), 1),
      avgComments: Math.max(...selectedTopics.map(t => t.avgComments), 1),
      avgCollects: Math.max(...selectedTopics.map(t => t.avgCollects), 1),
      avgShares: Math.max(...selectedTopics.map(t => t.avgShares), 1),
    }
    return dims.map(({ key, label }) => {
      const row: Record<string, string | number> = { subject: label, fullMark: 100 }
      selectedTopics.forEach((t, i) => {
        row[`s${i}`] = Math.round((t[key] / maxes[key]) * 100)
      })
      return row
    })
  }, [selectedTopics])

  // ── 按所选维度（赞/评/藏/享）的数值分段折线：横轴分段一致，纵轴为落在该段的笔记篇数 ──
  const intervalLineData = useMemo(() => {
    if (selectedTopics.length < 1) return []
    return METRIC_INTERVAL_BUCKETS.map(bucket => {
      const row: Record<string, string | number> = {
        interval: METRIC_BUCKET_LABELS[bucket],
      }
      selectedTopics.forEach((t, i) => {
        row[`s${i}`] = countNotesInMetricBucket(toNotes(t), bucket, intervalMetric)
      })
      return row
    })
  }, [selectedTopics, intervalMetric])

  const intervalMetricLabel = INTERVAL_METRIC_TABS.find(t => t.id === intervalMetric)?.label ?? '互动'

  return (
    <div className="space-y-6">
      {/* ── 顶部操作栏 ── */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
            <span className="text-2xl">⚡</span>
            主题对比模式
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <ExportButton topics={selectedTopics} />

          <button
            onClick={onBack}
            className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-500 hover:text-gray-700 border border-gray-200 rounded-lg transition-colors"
          >
            <X className="w-4 h-4" />
            退出对比
          </button>
        </div>
      </div>

      {/* ── 主题勾选列表 ── */}
      <div className="rounded-2xl p-4 border border-rose-100/60 bg-gradient-to-br from-white to-rose-50/30 shadow-md shadow-rose-100/20">
        <p className="text-xs font-medium text-gray-600 mb-3 flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-rose-400" />
          点击卡片勾选主题（最多 6 个）
        </p>
        <div className="flex flex-wrap gap-2">
          {topics.map(topic => {
            const isSel = selected.has(topic.id)
            return (
              <button
                key={topic.id}
                onClick={() => toggleTopic(topic.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-all border-2 ${
                  isSel
                    ? 'bg-gradient-to-r from-rose-50 to-pink-50 border-rose-400 text-rose-700 shadow-md shadow-rose-100/50 ring-1 ring-rose-100/60'
                    : 'bg-white/80 border-gray-100 text-gray-500 hover:border-rose-200 hover:bg-rose-50/40'
                }`}
              >
                {isSel
                  ? <CheckSquare className="w-4 h-4 text-rose-500 shrink-0" />
                  : <Square className="w-4 h-4 shrink-0" />
                }
                <span className="max-w-[100px] truncate">{topic.name}</span>
                <span className={`text-xs ${isSel ? 'text-rose-400' : 'text-gray-400'}`}>
                  {topic.noteCount}篇
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* ── 已选主题统计卡片 ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {selectedTopics.map((t, i) => (
          <div
            key={t.id}
            className="rounded-xl p-3 border-2 cursor-pointer transition-all hover:shadow-lg hover:shadow-rose-100/40 hover:-translate-y-0.5"
            style={{ borderColor: COLORS[i % COLORS.length], backgroundColor: `${COLORS[i % COLORS.length]}10` }}
            onClick={() => {
              const notes = toNotes(t)
              setShowNoteModal({ topic: t, notes })
            }}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
              <span className="font-semibold text-gray-700 text-sm truncate">{t.name}</span>
            </div>
            <div className="grid grid-cols-2 gap-1 text-xs text-gray-600">
              <div>❤️ {formatNumber(t.avgLikes)}</div>
              <div>💬 {formatNumber(t.avgComments)}</div>
              <div>⭐ {formatNumber(t.avgCollects)}</div>
              <div>📤 {formatNumber(t.avgShares)}</div>
            </div>
            <p className="text-xs text-gray-400 mt-1.5">{t.noteCount} 篇笔记 · 点击查看笔记</p>
          </div>
        ))}
      </div>

      {/* ── 图表区域：雷达画像 + 互动区间折线（可切换维度）── */}
      <div className="rounded-2xl p-4 sm:p-6 border border-rose-100/50 bg-gradient-to-b from-white via-white to-rose-50/20 shadow-lg shadow-gray-200/40 space-y-8">
        <div>
          <h4 className="font-semibold text-gray-800 mb-1 text-sm tracking-tight">互动画像（雷达）</h4>
          <p className="text-xs text-gray-400 mb-4">
            每条线为所选主题；四角为均赞、均评、均藏、均享在<strong className="text-gray-500">当前勾选组内</strong>的相对强度（最高=100）。绝对数值见上方卡片。
          </p>
          <div className="h-[min(400px,75vw)] min-h-[300px] w-full">
            <ResponsiveContainerAny width="100%" height="100%">
              <RadarChartAny cx="50%" cy="52%" outerRadius="72%" data={radarData}>
                <PolarGridAny stroke="#e2e8f0" />
                <PolarAngleAxisAny dataKey="subject" tick={{ fontSize: 12, fill: '#64748b' }} />
                <PolarRadiusAxisAny angle={45} domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <TooltipAny />
                <LegendAny wrapperStyle={{ fontSize: 12 }} formatter={(v: string) => <span className="text-gray-600">{v}</span>} />
                {selectedTopics.map((t, i) => (
                  <RadarAny
                    key={t.id}
                    name={topicShortLabels[i] || t.name}
                    dataKey={`s${i}`}
                    stroke={COLORS[i % COLORS.length]}
                    fill={COLORS[i % COLORS.length]}
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                ))}
              </RadarChartAny>
            </ResponsiveContainerAny>
          </div>
        </div>

        <div className="pt-6 border-t border-rose-100/60">
          <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
            <div>
              <h4 className="font-semibold text-gray-800 mb-1 text-sm tracking-tight">互动区间趋势（折线）</h4>
              <p className="text-xs text-gray-400 max-w-2xl leading-relaxed">
                按单条笔记的<strong className="text-gray-500">点赞 / 评论 / 收藏 / 分享</strong>计数落入同一套数值分段（0–100 … 5k+），纵轴为篇数；切换维度即可对比不同互动指标的分段分布。
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-1.5 shrink-0">
              <span className="text-[11px] text-gray-400 mr-0.5">维度</span>
              {INTERVAL_METRIC_TABS.map(tab => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setIntervalMetric(tab.id)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${
                    intervalMetric === tab.id
                      ? 'bg-rose-500 text-white border-rose-500 shadow-sm'
                      : 'bg-white text-gray-600 border-gray-200 hover:border-rose-200 hover:bg-rose-50/50'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
          <div className="h-[320px] w-full min-h-[280px]">
            <ResponsiveContainerAny width="100%" height="100%">
              <LineChartAny data={intervalLineData} margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
                <CartesianGridAny strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxisAny dataKey="interval" tick={{ fontSize: 11, fill: '#64748b' }} />
                <YAxisAny tick={{ fontSize: 11, fill: '#94a3b8' }} allowDecimals={false} />
                <TooltipAny
                  content={({ active, label, payload }) => {
                    if (!active || !payload?.length) return null
                    return (
                      <div className="bg-white/95 backdrop-blur-md rounded-xl shadow-lg border border-rose-100/60 p-3 text-xs min-w-[160px]">
                        <p className="font-semibold text-gray-800 mb-2 border-b border-gray-100 pb-1.5">
                          {intervalMetricLabel} {label}
                        </p>
                        {payload.map((e: any) => (
                          <div key={e.dataKey} className="flex justify-between gap-6 text-gray-600 py-0.5">
                            <span className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: e.color }} />
                              {e.name}
                            </span>
                            <span className="font-medium tabular-nums">{formatNumber(e.value)} 篇</span>
                          </div>
                        ))}
                      </div>
                    )
                  }}
                />
                <LegendAny wrapperStyle={{ fontSize: 11 }} />
                {selectedTopics.map((t, i) => (
                  <LineAny
                    key={t.id}
                    type="monotone"
                    dataKey={`s${i}`}
                    name={topicShortLabels[i] || t.name}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 3, strokeWidth: 1, fill: '#fff' }}
                    activeDot={{ r: 5 }}
                  />
                ))}
              </LineChartAny>
            </ResponsiveContainerAny>
          </div>
        </div>
      </div>

      {/* ── 笔记详情弹窗 ── */}
      {showNoteModal && (
        <div
          className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          onClick={() => setShowNoteModal(null)}
        >
          <div
            className="bg-white rounded-2xl max-w-lg w-full max-h-[80vh] overflow-hidden flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between shrink-0">
              <div>
                <h3 className="font-semibold text-gray-700">{showNoteModal.topic.name}</h3>
                <p className="text-xs text-gray-400 mt-0.5">{showNoteModal.notes.length} 篇笔记 · 点击按点赞排序</p>
              </div>
              <button onClick={() => setShowNoteModal(null)} className="p-1 hover:bg-gray-100 rounded-lg">
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto space-y-2">
              {showNoteModal.notes
                .sort((a, b) => b.likes - a.likes)
                .slice(0, 20)
                .map(note => (
                  <div key={note.id} className="p-3 bg-gray-50 rounded-xl hover:bg-rose-50 transition-colors">
                    <div className="text-sm font-medium text-gray-700 line-clamp-2 mb-1.5">{note.title || '无标题'}</div>
                    <div className="flex items-center gap-3 text-xs text-gray-400">
                      <span>❤️ {formatNumber(note.likes)}</span>
                      <span>💬 {formatNumber(note.comments)}</span>
                      <span>⭐ {formatNumber(note.collects)}</span>
                      <span className="ml-auto text-cyan-500">{(note.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ================================================================
// 导出按钮
// ================================================================

export function ExportButton({ topics, notes, filename = 'topic-comparison' }: ExportButtonProps) {
  const [open, setOpen] = useState(false)

  const handleExportCSV = (type: 'summary' | 'detail' | 'intervalMatrix') => {
    if (topics.length === 0) {
      setOpen(false)
      return
    }
    if (type === 'summary') {
      const data = topics.map(t => ({
        主题: t.name,
        笔记数: t.noteCount,
        平均点赞: Math.round(t.avgLikes),
        平均评论: Math.round(t.avgComments),
        平均收藏: Math.round(t.avgCollects),
        平均分享: Math.round(t.avgShares),
        置信度: `${(t.avgConfidence * 100).toFixed(1)}%`,
        关键词: t.keywords.join(' | '),
      }))
      downloadCSV(data, exportCsvBasename(filename, '主题摘要'))
    } else if (type === 'detail') {
      const allNotes = topics.flatMap(t =>
        toNotes(t).map(n => ({
          主题: t.name,
          笔记ID: n.id,
          标题: n.title,
          点赞: n.likes,
          评论: n.comments,
          收藏: n.collects,
          分享: n.shares,
          置信度: `${(n.confidence * 100).toFixed(1)}%`,
          IP属地: n.ipLocation,
        })),
      )
      downloadCSV(allNotes, exportCsvBasename(filename, '笔记明细'))
    } else {
      const rows: Record<string, string | number>[] = []
      for (const t of topics) {
        const noteList = toNotes(t)
        for (const tab of INTERVAL_METRIC_TABS) {
          for (const bucket of METRIC_INTERVAL_BUCKETS) {
            rows.push({
              主题: t.name,
              笔记数: t.noteCount,
              互动维度: tab.label,
              数值区间: METRIC_BUCKET_LABELS[bucket],
              该区间的笔记篇数: countNotesInMetricBucket(noteList, bucket, tab.id),
            })
          }
        }
      }
      downloadCSV(rows, exportCsvBasename(filename, '互动区间汇总'))
    }
    setOpen(false)
  }

  return (
    <div className="relative">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={() => setOpen(!open)}
        className={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-xl border transition-all ${
          open
            ? 'text-rose-600 border-rose-300 bg-rose-50/90 shadow-sm'
            : 'text-gray-600 border-gray-200 hover:text-rose-600 hover:border-rose-200 hover:bg-rose-50/40'
        }`}
      >
        <Download className="w-4 h-4 shrink-0 opacity-90" aria-hidden />
        导出 CSV
        <ChevronDown className={`w-3.5 h-3.5 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" aria-hidden onClick={() => setOpen(false)} />
          <div
            role="menu"
            className="absolute right-0 top-full mt-1.5 z-20 w-[min(100vw-2rem,17.5rem)] rounded-xl shadow-xl border border-rose-100/80 bg-white p-1.5 space-y-0.5"
          >
            <button
              type="button"
              role="menuitem"
              onClick={() => handleExportCSV('summary')}
              className="w-full flex gap-2.5 px-2.5 py-2.5 text-left rounded-lg hover:bg-rose-50/90 transition-colors"
            >
              <FileSpreadsheet className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" aria-hidden />
              <span>
                <span className="block text-sm font-medium text-gray-800">主题摘要</span>
                <span className="block text-[11px] text-gray-500 mt-0.5 leading-snug">已选主题的笔记数、四维均值、置信度与关键词</span>
              </span>
            </button>
            <button
              type="button"
              role="menuitem"
              onClick={() => handleExportCSV('intervalMatrix')}
              className="w-full flex gap-2.5 px-2.5 py-2.5 text-left rounded-lg hover:bg-rose-50/90 transition-colors"
            >
              <Table2 className="w-4 h-4 text-violet-500 shrink-0 mt-0.5" aria-hidden />
              <span>
                <span className="block text-sm font-medium text-gray-800">互动区间汇总</span>
                <span className="block text-[11px] text-gray-500 mt-0.5 leading-snug">每主题 × 赞/评/藏/享 × 各数值分段篇数（与对比图一致）</span>
              </span>
            </button>
            <button
              type="button"
              role="menuitem"
              onClick={() => handleExportCSV('detail')}
              className="w-full flex gap-2.5 px-2.5 py-2.5 text-left rounded-lg hover:bg-rose-50/90 transition-colors"
            >
              <FileText className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" aria-hidden />
              <span>
                <span className="block text-sm font-medium text-gray-800">笔记明细</span>
                <span className="block text-[11px] text-gray-500 mt-0.5 leading-snug">已选主题下全部笔记的行级互动与置信度</span>
              </span>
            </button>
          </div>
        </>
      )}
    </div>
  )
}

// ================================================================
// 带动画的柱状图（interactive OverviewView bar）
// ================================================================

interface AnimatedBarChartProps {
  topics: Topic[]
  metric: 'avgLikes' | 'avgComments' | 'avgCollects' | 'avgShares'
  metricLabel: string
  color?: string
  onTopicClick?: (t: Topic) => void
}

export function AnimatedBarChart({ topics, metric, metricLabel, color = '#f43f5e', onTopicClick }: AnimatedBarChartProps) {
  const [hoveredId, setHoveredId] = useState<number | null>(null)
  const sorted = useMemo(() => [...topics].sort((a, b) => b[metric] - a[metric]).slice(0, 12), [topics, metric])
  const max = sorted[0]?.[metric] || 1

  return (
    <div className="space-y-1.5">
      {sorted.map((t, i) => {
        const val = t[metric]
        const pct = (val / max) * 100
        const isHovered = hoveredId === t.id
        return (
          <div
            key={t.id}
            className="group relative cursor-pointer"
            onMouseEnter={() => setHoveredId(t.id)}
            onMouseLeave={() => setHoveredId(null)}
            onClick={() => onTopicClick?.(t)}
          >
            {/* 标签行 */}
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-xs text-gray-400 w-4 text-right shrink-0">{i + 1}</span>
              <span className={`text-xs truncate transition-colors ${isHovered ? 'text-rose-600 font-medium' : 'text-gray-600'}`}>
                {t.name}
              </span>
              <span className="text-xs text-gray-400 ml-auto shrink-0">{formatNumber(val)}</span>
            </div>
            {/* 柱子 */}
            <div className="flex items-center gap-2">
              <div className="w-4 shrink-0" />
              <div className="flex-1 h-5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500 group-hover:brightness-110"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: color,
                    transitionDelay: `${i * 30}ms`,
                  }}
                />
              </div>
            </div>
            {/* 悬浮提示 */}
            {isHovered && (
              <div className="absolute left-full ml-2 top-0 z-10 bg-white/95 backdrop-blur rounded-xl shadow-lg border border-gray-100 p-3 pointer-events-none min-w-[180px]">
                <p className="font-semibold text-gray-700 text-sm mb-1">{t.name}</p>
                <div className="space-y-0.5 text-xs text-gray-600">
                  <div>❤️ {metricLabel}：{formatNumber(val)}</div>
                  <div>📄 笔记数：{t.noteCount}</div>
                  <div>🎯 置信度：{(t.avgConfidence * 100).toFixed(1)}%</div>
                </div>
                <p className="text-xs text-rose-400 mt-1.5">点击查看详情 →</p>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
