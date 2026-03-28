/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useState, useMemo, useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer
} from 'recharts'
import { Download, X, CheckSquare, Square, ChevronDown, ChevronUp, SlidersHorizontal, Info } from 'lucide-react'
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
const RadarChartAny = RadarChart as any
const RadarAny = Radar as any
const PolarGridAny = PolarGrid as any
const PolarAngleAxisAny = PolarAngleAxis as any
const PolarRadiusAxisAny = PolarRadiusAxis as any
const ResponsiveContainerAny = ResponsiveContainer as any

// ================================================================
// 类型定义
// ================================================================

interface CompareViewProps {
  topics: Topic[]
  allRecords: Topic[]
  onBack: () => void
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any
}

interface MetricFilterProps {
  minLikes: number
  maxLikes: number
  minComments: number
  maxComments: number
  minCollects: number
  maxCollects: number
  onChange: (key: string, value: [number, number]) => void
  maxValues: { likes: number; comments: number; collects: number }
}

interface ExportButtonProps {
  topics: Topic[]
  notes?: Note[]
  filename?: string
}

// ================================================================
// 工具函数
// ================================================================

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
// 指标区间过滤器
// ================================================================

export function MetricFilterPanel({ minLikes, maxLikes, minComments, maxComments, minCollects, maxCollects, onChange, maxValues }: MetricFilterProps) {
  const [expanded, setExpanded] = useState(false)

  const metrics = [
    { key: 'likes', label: '平均点赞', min: minLikes, max: maxLikes, step: Math.max(1, Math.floor(maxValues.likes / 50)) },
    { key: 'comments', label: '平均评论', min: minComments, max: maxComments, step: Math.max(1, Math.floor(maxValues.comments / 50)) },
    { key: 'collects', label: '平均收藏', min: minCollects, max: maxCollects, step: Math.max(1, Math.floor(maxValues.collects / 50)) },
  ]

  return (
    <div className="bg-white rounded-2xl shadow-md border border-gray-100 overflow-hidden">
      {/* 折叠头部 */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="w-4 h-4 text-rose-500" />
          <span className="font-medium text-gray-700 text-sm">指标区间过滤</span>
          {(minLikes > 0 || maxLikes < maxValues.likes || minComments > 0 || maxComments < maxValues.comments || minCollects > 0 || maxCollects < maxValues.collects) && (
            <span className="px-1.5 py-0.5 bg-rose-100 text-rose-600 text-xs rounded-full font-medium">已设置</span>
          )}
        </div>
        {expanded ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
      </button>

      {/* 展开内容 */}
      {expanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-gray-100 pt-3">
          {metrics.map(m => (
            <div key={m.key}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium text-gray-600">{m.label}</span>
                <span className="text-xs text-rose-500 font-medium">
                  {formatNumber(m.min)} ~ {formatNumber(m.max)}
                </span>
              </div>
              {/* 双滑块 */}
              <div className="relative h-6 flex items-center">
                <input
                  type="range"
                  min={0}
                  max={maxValues[m.key as 'likes' | 'comments' | 'collects']}
                  step={m.step}
                  value={m.min}
                  onChange={e => {
                    const val = Math.min(parseInt(e.target.value), m.max - m.step)
                    onChange(`${m.key}Min` as any, [val, m.max])
                  }}
                  className="absolute w-full appearance-none bg-transparent pointer-events-none z-10
                    [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none
                    [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
                    [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-rose-500
                    [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:cursor-pointer
                    [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-125"
                  style={{ accentColor: '#f43f5e' }}
                />
                <input
                  type="range"
                  min={0}
                  max={maxValues[m.key as 'likes' | 'comments' | 'collects']}
                  step={m.step}
                  value={m.max}
                  onChange={e => {
                    const val = Math.max(parseInt(e.target.value), m.min + m.step)
                    onChange(`${m.key}Max` as any, [m.min, val])
                  }}
                  className="absolute w-full appearance-none bg-transparent pointer-events-none z-20
                    [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none
                    [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
                    [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-rose-500
                    [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:cursor-pointer
                    [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-125"
                  style={{ accentColor: '#f43f5e' }}
                />
                {/* 轨道背景 */}
                <div className="absolute w-full h-1.5 bg-gray-200 rounded-full" />
                {/* 已选轨道 */}
                <div
                  className="absolute h-1.5 bg-rose-400 rounded-full"
                  style={{
                    left: `${(m.min / maxValues[m.key as 'likes' | 'comments' | 'collects']) * 100}%`,
                    right: `${100 - (m.max / maxValues[m.key as 'likes' | 'comments' | 'collects']) * 100}%`,
                  }}
                />
              </div>
            </div>
          ))}

          {/* 重置按钮 */}
          <button
            onClick={() => {
              onChange('likesMin' as any, [0, maxValues.likes])
              onChange('commentsMin' as any, [0, maxValues.comments])
              onChange('collectsMin' as any, [0, maxValues.collects])
            }}
            className="w-full py-1.5 text-xs text-gray-400 hover:text-rose-500 border border-gray-200 rounded-lg transition-colors"
          >
            重置所有过滤条件
          </button>
        </div>
      )}
    </div>
  )
}

// ================================================================
// 悬停详情气泡
// ================================================================

export function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: any[]; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white/95 backdrop-blur-md rounded-xl shadow-xl border border-gray-100 p-3 min-w-[160px]">
      <p className="font-semibold text-gray-700 text-sm mb-2">{label}</p>
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

export function CompareView({ topics, allRecords, onBack }: CompareViewProps) {
  const [selected, setSelected] = useState<Set<number>>(new Set(topics.slice(0, Math.min(4, topics.length)).map(t => t.id)))
  const [compareMode, setCompareMode] = useState<'radar' | 'bar' | 'trend'>('radar')
  const [showNoteModal, setShowNoteModal] = useState<{ topic: Topic; notes: Note[] } | null>(null)

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

  // ── 雷达图数据 ──
  const radarData = useMemo(() => {
    const metrics = ['平均点赞', '平均评论', '平均收藏', '平均分享', '置信度']
    return metrics.map(metric => {
      const entry: Record<string, any> = { metric }
      selectedTopics.forEach(t => {
        switch (metric) {
          case '平均点赞': entry[t.name] = Math.round(t.avgLikes); break
          case '平均评论': entry[t.name] = Math.round(t.avgComments); break
          case '平均收藏': entry[t.name] = Math.round(t.avgCollects); break
          case '平均分享': entry[t.name] = Math.round(t.avgShares); break
          case '置信度': entry[t.name] = parseFloat((t.avgConfidence * 100).toFixed(1)); break
        }
      })
      return entry
    })
  }, [selectedTopics])

  // ── 柱状图数据 ──
  const barData = useMemo(() => {
    return selectedTopics.map(t => ({
      name: t.name.length > 8 ? t.name.slice(0, 8) + '…' : t.name,
      fullName: t.name,
      点赞: Math.round(t.avgLikes),
      评论: Math.round(t.avgComments),
      收藏: Math.round(t.avgCollects),
      分享: Math.round(t.avgShares),
      笔记数: t.noteCount,
    }))
  }, [selectedTopics])

  // ── 笔记级对比数据（按点赞分段）──
  const notesDistributionData = useMemo(() => {
    if (selectedTopics.length < 2) return []
    const buckets = ['0-100', '101-500', '501-1k', '1k-5k', '5k+']
    return buckets.map(bucket => {
      const entry: Record<string, any> = { bucket }
      selectedTopics.forEach(t => {
        const notes = toNotes(t)
        let count = 0
        switch (bucket) {
          case '0-100': count = notes.filter(n => n.likes <= 100).length; break
          case '101-500': count = notes.filter(n => n.likes > 100 && n.likes <= 500).length; break
          case '501-1k': count = notes.filter(n => n.likes > 500 && n.likes <= 1000).length; break
          case '1k-5k': count = notes.filter(n => n.likes > 1000 && n.likes <= 5000).length; break
          case '5k+': count = notes.filter(n => n.likes > 5000).length; break
        }
        entry[t.name] = count
      })
      return entry
    })
  }, [selectedTopics])

  return (
    <div className="space-y-6">
      {/* ── 顶部操作栏 ── */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
            <span className="text-2xl">⚡</span>
            主题对比模式
          </h3>
          <p className="text-xs text-gray-400 mt-0.5">勾选 2~6 个主题进行横向对比，按 Ctrl/Cmd 可多选</p>
        </div>
        <div className="flex items-center gap-2">
          {/* 图表类型切换 */}
          <div className="flex bg-gray-100 rounded-lg p-0.5">
            {[
              { id: 'radar', label: '雷达图', icon: '🕸️' },
              { id: 'bar', label: '柱状图', icon: '📊' },
              { id: 'trend', label: '笔记分布', icon: '📈' },
            ].map(btn => (
              <button
                key={btn.id}
                onClick={() => setCompareMode(btn.id as typeof compareMode)}
                className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  compareMode === btn.id ? 'bg-white shadow-sm text-rose-600' : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <span>{btn.icon}</span>
                <span className="hidden sm:inline">{btn.label}</span>
              </button>
            ))}
          </div>

          {/* 导出按钮 */}
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
      <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100">
        <p className="text-xs font-medium text-gray-500 mb-3 flex items-center gap-1">
          <Info className="w-3.5 h-3.5" />
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
                    ? 'bg-rose-50 border-rose-400 text-rose-600 shadow-sm'
                    : 'bg-gray-50 border-transparent text-gray-500 hover:bg-gray-100'
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
            className="rounded-xl p-3 border-2 cursor-pointer transition-all hover:shadow-md"
            style={{ borderColor: COLORS[i % COLORS.length], backgroundColor: `${COLORS[i % COLORS.length]}08` }}
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

      {/* ── 图表区域 ── */}
      <div className="bg-white rounded-2xl p-4 sm:p-6 shadow-sm border border-gray-100">
        {compareMode === 'radar' && (
          <div>
            <h4 className="font-semibold text-gray-700 mb-4 text-sm">多维度能力雷达图（归一化显示）</h4>
            <ResponsiveContainerAny width="100%" height={360}>
              <RadarChartAny data={radarData}>
                <PolarGridAny stroke="#e5e7eb" />
                <PolarAngleAxisAny dataKey="metric" tick={{ fontSize: 12, fill: '#6b7280' }} />
                <PolarRadiusAxisAny tick={{ fontSize: 10, fill: '#9ca3af' }} />
                {selectedTopics.map((t, i) => (
                  <RadarAny
                    key={t.id}
                    name={t.name}
                    dataKey={t.name}
                    stroke={COLORS[i % COLORS.length]}
                    fill={COLORS[i % COLORS.length]}
                    fillOpacity={0.12 + i * 0.04}
                    strokeWidth={2}
                  />
                ))}
                <LegendAny
                  formatter={(value) => <span className="text-xs text-gray-600">{value}</span>}
                 />
                <TooltipAny content={<ChartTooltip />} />
              </RadarChartAny>
            </ResponsiveContainerAny>
          </div>
        )}

        {compareMode === 'bar' && (
          <div>
            <h4 className="font-semibold text-gray-700 mb-4 text-sm">互动指标柱状对比</h4>
            <ResponsiveContainerAny width="100%" height={360}>
              <BarChartAny data={barData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGridAny strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxisAny dataKey="name" tick={{ fontSize: 11, fill: '#6b7280' }} />
                <YAxisAny tick={{ fontSize: 11, fill: '#9ca3af' }}  />
                <TooltipAny content={<ChartTooltip />}  />
                <LegendAny formatter={(value) => <span className="text-xs text-gray-600">{value}</span>}  />
                {['点赞', '评论', '收藏', '分享'].map((key, i) => (
                  <BarAny
                    key={key}
                    dataKey={key}
                    fill={COLORS[i % COLORS.length]}
                    radius={[4, 4, 0, 0]}
                    maxBarSize={40}
                  />
                ))}
              </BarChartAny>
            </ResponsiveContainerAny>
          </div>
        )}

        {compareMode === 'trend' && (
          <div>
            <h4 className="font-semibold text-gray-700 mb-4 text-sm">笔记互动区间分布对比</h4>
            <p className="text-xs text-gray-400 mb-4">各主题笔记按点赞数分段统计</p>
            <ResponsiveContainerAny width="100%" height={320}>
              <BarChartAny data={notesDistributionData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGridAny strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxisAny dataKey="bucket" tick={{ fontSize: 11, fill: '#6b7280' }} />
                <YAxisAny tick={{ fontSize: 11, fill: '#9ca3af' }}  />
                <TooltipAny content={<ChartTooltip />}  />
                <LegendAny formatter={(value) => <span className="text-xs text-gray-600">{value}</span>}  />
                {selectedTopics.map((t, i) => (
                  <BarAny
                    key={t.id}
                    dataKey={t.name}
                    fill={COLORS[i % COLORS.length]}
                    radius={[4, 4, 0, 0]}
                    maxBarSize={30}
                  />
                ))}
              </BarChartAny>
            </ResponsiveContainerAny>
          </div>
        )}
      </div>

      {/* ── 详细数据表格 ── */}
      <div className="bg-white rounded-2xl p-4 sm:p-6 shadow-sm border border-gray-100 overflow-x-auto">
        <h4 className="font-semibold text-gray-700 mb-4 text-sm">对比数据明细</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100">
              <th className="text-left py-2 pr-4 text-gray-500 font-medium text-xs">主题</th>
              <th className="text-right py-2 px-2 text-gray-500 font-medium text-xs">笔记数</th>
              <th className="text-right py-2 px-2 text-gray-500 font-medium text-xs">均赞</th>
              <th className="text-right py-2 px-2 text-gray-500 font-medium text-xs">均评</th>
              <th className="text-right py-2 px-2 text-gray-500 font-medium text-xs">均藏</th>
              <th className="text-right py-2 px-2 text-gray-500 font-medium text-xs">均分</th>
              <th className="text-right py-2 pl-2 text-gray-500 font-medium text-xs">置信度</th>
            </tr>
          </thead>
          <tbody>
            {selectedTopics.map((t, i) => (
              <tr key={t.id} className="border-b border-gray-50 hover:bg-rose-50/30 transition-colors">
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                    <span className="font-medium text-gray-700">{t.name}</span>
                  </div>
                </td>
                <td className="text-right py-2.5 px-2 text-gray-600">{formatNumber(t.noteCount)}</td>
                <td className="text-right py-2.5 px-2 text-gray-600">{formatNumber(t.avgLikes)}</td>
                <td className="text-right py-2.5 px-2 text-gray-600">{formatNumber(t.avgComments)}</td>
                <td className="text-right py-2.5 px-2 text-gray-600">{formatNumber(t.avgCollects)}</td>
                <td className="text-right py-2.5 pl-2 text-gray-600">{formatNumber(t.avgShares)}</td>
                <td className="text-right py-2.5 pl-2">
                  <span className={`font-medium ${
                    t.avgConfidence > 0.7 ? 'text-green-600' : t.avgConfidence > 0.5 ? 'text-yellow-600' : 'text-red-500'
                  }`}>
                    {(t.avgConfidence * 100).toFixed(1)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
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

  const handleExportCSV = (type: 'summary' | 'detail') => {
    if (type === 'summary') {
      const data = topics.map(t => ({
        主题: t.name,
        笔记数: t.noteCount,
        平均点赞: Math.round(t.avgLikes),
        平均评论: Math.round(t.avgComments),
        平均收藏: Math.round(t.avgCollects),
        平均分享: Math.round(t.avgShares),
        置信度: (t.avgConfidence * 100).toFixed(1) + '%',
        关键词: t.keywords.join(' | '),
      }))
      downloadCSV(data, `${filename}-摘要.csv`)
    } else {
      const allNotes = topics.flatMap(t => toNotes(t).map(n => ({
        主题: t.name,
        笔记ID: n.id,
        标题: n.title,
        点赞: n.likes,
        评论: n.comments,
        收藏: n.collects,
        分享: n.shares,
        置信度: (n.confidence * 100).toFixed(1) + '%',
        IP属地: n.ipLocation,
      })))
      downloadCSV(allNotes, `${filename}-笔记明细.csv`)
    }
    setOpen(false)
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-500 hover:text-rose-500 border border-gray-200 hover:border-rose-300 rounded-lg transition-colors"
      >
        <Download className="w-4 h-4" />
        导出
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-full mt-1 z-20 bg-white rounded-xl shadow-xl border border-gray-100 p-1 min-w-[160px]">
            <button
              onClick={() => handleExportCSV('summary')}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:bg-rose-50 hover:text-rose-600 rounded-lg transition-colors"
            >
              <span>📋</span> 导出主题摘要
            </button>
            <button
              onClick={() => handleExportCSV('detail')}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:bg-rose-50 hover:text-rose-600 rounded-lg transition-colors"
            >
              <span>📄</span> 导出笔记明细
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
