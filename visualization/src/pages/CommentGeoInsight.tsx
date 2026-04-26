import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeft, Loader2, MapPin } from 'lucide-react'
import { loadLiveCommentBundle, type LiveCommentBundle } from '../data/commentLiveData'
import { CommentGeoInsightSections } from '../components/CommentGeoInsightSections'

export default function CommentGeoInsight() {
  const [bundle, setBundle] = useState<LiveCommentBundle | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let c = false
    setLoading(true)
    loadLiveCommentBundle()
      .then(b => {
        if (!c) {
          setBundle(b)
          setErr(null)
        }
      })
      .catch(e => {
        if (!c) setErr(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!c) setLoading(false)
      })
    return () => {
      c = true
    }
  }, [])

  if (loading) {
    return (
      <div className="min-h-[50vh] flex flex-col items-center justify-center gap-3 text-slate-600">
        <Loader2 className="w-10 h-10 animate-spin text-cyan-500" />
        <p className="text-sm">加载评论与地域数据…</p>
      </div>
    )
  }

  if (err || !bundle) {
    return (
      <div className="max-w-lg mx-auto mt-16 rounded-2xl border border-amber-200 bg-amber-50/80 p-6 text-amber-950 text-sm">
        {err || '无数据'}
      </div>
    )
  }

  return (
    <div className="min-h-screen text-slate-900 pb-16 px-4 sm:px-6">
      <header className="max-w-7xl mx-auto pt-4 pb-6 flex flex-wrap items-center gap-4">
        <Link
          to="/comments"
          className="inline-flex items-center gap-2 text-sm font-medium text-cyan-700 hover:text-cyan-900"
        >
          <ArrowLeft className="w-4 h-4" />
          返回评论分析
        </Link>
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shrink-0">
            <MapPin className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900">地域 · 情感 · 主题</h1>
            <p className="text-xs text-slate-500 mt-0.5">
              IP 分布与极性来自全量预测子集；主题交叉为 IP × content 宏观主题（note_id 对齐）
            </p>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto space-y-10">
        <CommentGeoInsightSections bundle={bundle} />

        <p className="text-[11px] text-slate-500 text-center max-w-2xl mx-auto">
          数据与主评论分析页同源（<span className="font-mono">loadLiveCommentBundle</span>）。若需省/直辖市下钻，可在下游对 IP 文本做归一化后再聚合。
        </p>
      </div>
    </div>
  )
}
