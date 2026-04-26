import { CONTENT_SERVER_BASE, loadTopicCSV, type TopicRecord } from './topicData'
import type {
  CommentAnalysis,
  CommentMetrics,
  CommentTopic,
  NoteCommentLine,
  TopComment,
} from './commentData'
import type { EvaluationMetrics, RoBERTaConfig } from './commentData'

const BASE = CONTENT_SERVER_BASE.replace(/\/$/, '')

export interface PredictedCommentRow {
  note_id: string
  nickname: string
  content: string
  ip_location: string | null
  sentiment_polarity: number
  sentiment_text: string
}

export interface LivePolaritySummary {
  negative: number
  neutral: number
  positive: number
  total: number
}

export interface WordFreqItem {
  text: string
  weight: number
}

export interface LengthBucketRow {
  bucket: string
  negative: number
  neutral: number
  positive: number
}

export interface IpBucketRow {
  name: string
  count: number
  positive: number
  neutral: number
  negative: number
}

export interface ParsedEvaluation {
  metrics: EvaluationMetrics
  classData: { name: string; precision: number; recall: number; f1: number; support: number }[]
}

/** 与 evaluate_model.py 写出的 eval_viz_payload.json 一致，用于前端复现 04–06 等图 */
export interface EvalVizPayload {
  true_labels: number[]
  pred_labels: number[]
  probs: [number, number, number][]
}

/** 预测评论按 IP × content 宏观主题 聚合（用于地域洞察页） */
export interface GeoMacroRow {
  region: string
  macroTopic: string
  count: number
}

export interface LiveCommentBundle {
  polarity: LivePolaritySummary | null
  metrics: CommentMetrics
  noteRank: { noteId: string; commentCount: number }[]
  topics: CommentTopic[]
  noteDetails: CommentAnalysis[]
  wordCloud: WordFreqItem[]
  lengthBySentiment: LengthBucketRow[]
  ipDistribution: IpBucketRow[]
  /** 全量预测子集中 IP 与 BERTopic 宏观主题的交叉计数 */
  geoMacroMix: GeoMacroRow[]
  predictionsMeta: { total: number; sampleSize: number }
  evalParse: ParsedEvaluation | null
  /** 存在且非空时与 matplotlib 导出图同源，用于置信度/ROC 等 */
  evalViz: EvalVizPayload | null
  robertaConfig: RoBERTaConfig
  trainingHistory: { epoch: number; trainLoss: number; valLoss: number; valAcc: number; valF1: number }[]
  /** 训练曲线来源（开发服从 checkpoint_temp 下最新 checkpoint 解析） */
  trainingHistoryMeta: { checkpointDir: string; relativePath: string } | null
  /** comment note_id 与 content/BERTopic 关联摘要 */
  contentJoinSummary: {
    contentTopicRows: number
    highCommentNotesMatched: number
    highCommentNotesTotal: number
  }
}

function contentNoteMap(records: TopicRecord[]): Map<string, TopicRecord> {
  const m = new Map<string, TopicRecord>()
  for (const r of records) {
    const id = (r.note_id || '').trim()
    if (id) m.set(id, r)
  }
  return m
}

function displayTitleFromContent(r: TopicRecord | undefined, noteId: string): string {
  if (!r) return `笔记 ${noteId.slice(0, 12)}…`
  const t = (r.title || '').trim()
  if (t) return t.length > 52 ? `${t.slice(0, 52)}…` : t
  const c = (r.content || '').trim()
  if (c) return c.length > 52 ? `${c.slice(0, 52)}…` : c
  return `笔记 ${noteId.slice(0, 12)}…`
}

function polarityToSentiment(
  p: number
): 'positive' | 'neutral' | 'negative' {
  if (p === 1) return 'positive'
  if (p === -1) return 'negative'
  return 'neutral'
}

function sentimentScore01(p: number): number {
  return (p + 1) / 2
}

function parseSimpleCsv(text: string): string[][] {
  const rows: string[][] = []
  let cur = ''
  let row: string[] = []
  let inQ = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (inQ) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cur += '"'
          i++
        } else {
          inQ = false
        }
      } else cur += ch
      continue
    }
    if (ch === '"') {
      inQ = true
      continue
    }
    if (ch === ',') {
      row.push(cur)
      cur = ''
      continue
    }
    if (ch === '\r') continue
    if (ch === '\n') {
      row.push(cur)
      cur = ''
      if (row.some(c => c.length)) rows.push(row)
      row = []
      continue
    }
    cur += ch
  }
  row.push(cur)
  if (row.some(c => c.length)) rows.push(row)
  return rows
}

export function parsePolarityCsv(text: string): LivePolaritySummary | null {
  const rows = parseSimpleCsv(text.trim())
  if (rows.length < 2) return null
  const neg = { negative: 0, neutral: 0, positive: 0, total: 0 }
  for (let i = 1; i < rows.length; i++) {
    const [pol, countStr] = rows[i]
    const c = parseInt(countStr?.replace(/"/g, '') || '0', 10)
    if (!Number.isFinite(c)) continue
    const p = parseInt(pol, 10)
    if (p === -1) neg.negative = c
    else if (p === 0) neg.neutral = c
    else if (p === 1) neg.positive = c
  }
  neg.total = neg.negative + neg.neutral + neg.positive
  return neg.total ? neg : null
}

export function parseNotesCsv(text: string): { noteId: string; commentCount: number }[] {
  const rows = parseSimpleCsv(text.trim())
  const out: { noteId: string; commentCount: number }[] = []
  for (let i = 1; i < rows.length; i++) {
    const [id, c] = rows[i]
    if (!id) continue
    const n = parseInt(c, 10)
    if (!Number.isFinite(n)) continue
    out.push({ noteId: id.trim(), commentCount: n })
  }
  return out.sort((a, b) => b.commentCount - a.commentCount)
}

export function parseMeaninglessWords(text: string): Set<string> {
  const s = new Set<string>()
  for (const line of text.split(/\r?\n/)) {
    const t = line.replace(/\ufeff/g, '').trim()
    if (!t || t.startsWith('#')) continue
    const w = t.toLowerCase()
    if (w.length) s.add(w)
  }
  return s
}

function tokenizeForFreq(content: string): string[] {
  return content
    .split(/[^\u4e00-\u9fa5a-zA-Z0-9]+/)
    .map(t => t.trim())
    .filter(t => t.length >= 2)
}

function buildWordFreq(
  rows: PredictedCommentRow[],
  stop: Set<string>,
  topN = 36
): WordFreqItem[] {
  const m = new Map<string, number>()
  for (const r of rows) {
    for (const t of tokenizeForFreq(r.content || '')) {
      const k = t.toLowerCase()
      if (stop.has(k)) continue
      m.set(t, (m.get(t) || 0) + 1)
    }
  }
  return [...m.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([text, weight]) => ({ text, weight }))
}

function lengthBucket(len: number): string {
  if (len < 8) return '1–7 字'
  if (len < 20) return '8–19 字'
  if (len < 50) return '20–49 字'
  return '50 字及以上'
}

function buildLengthBuckets(rows: PredictedCommentRow[]): LengthBucketRow[] {
  const keys = ['1–7 字', '8–19 字', '20–49 字', '50 字及以上'] as const
  const acc: Record<string, { negative: number; neutral: number; positive: number }> =
    {}
  for (const k of keys) acc[k] = { negative: 0, neutral: 0, positive: 0 }
  for (const r of rows) {
    const len = [...(r.content || '')].length
    const b = lengthBucket(len)
    if (r.sentiment_polarity === 1) acc[b].positive++
    else if (r.sentiment_polarity === -1) acc[b].negative++
    else acc[b].neutral++
  }
  return keys.map(bucket => ({
    bucket,
    negative: acc[bucket].negative,
    neutral: acc[bucket].neutral,
    positive: acc[bucket].positive,
  }))
}

function buildIpDistribution(
  rows: PredictedCommentRow[],
  topN = 12
): IpBucketRow[] {
  const m = new Map<
    string,
    { count: number; positive: number; neutral: number; negative: number }
  >()
  for (const r of rows) {
    const name = r.ip_location?.trim() || '未知'
    if (!m.has(name))
      m.set(name, { count: 0, positive: 0, neutral: 0, negative: 0 })
    const o = m.get(name)!
    o.count++
    if (r.sentiment_polarity === 1) o.positive++
    else if (r.sentiment_polarity === -1) o.negative++
    else o.neutral++
  }
  return [...m.entries()]
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, topN)
    .map(([name, v]) => ({ name, ...v }))
}

interface NoteAgg {
  noteId: string
  comments: PredictedCommentRow[]
}

function buildNoteAggregates(
  sample: PredictedCommentRow[],
  topNoteIds: Set<string>
): NoteAgg[] {
  const map = new Map<string, PredictedCommentRow[]>()
  for (const r of sample) {
    const nid = String(r.note_id ?? '').trim()
    if (!nid || !topNoteIds.has(nid)) continue
    if (!map.has(nid)) map.set(nid, [])
    map.get(nid)!.push(r)
  }
  return [...map.entries()].map(([noteId, comments]) => ({ noteId, comments }))
}

function topKeywordsForNote(
  comments: PredictedCommentRow[],
  stop: Set<string>,
  k = 6
): string[] {
  const m = new Map<string, number>()
  for (const c of comments) {
    for (const t of tokenizeForFreq(c.content || '')) {
      const key = t.toLowerCase()
      if (stop.has(key)) continue
      m.set(t, (m.get(t) || 0) + 1)
    }
  }
  return [...m.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([w]) => w)
}

function ratio3(comments: PredictedCommentRow[]) {
  let pos = 0,
    neu = 0,
    neg = 0
  for (const c of comments) {
    if (c.sentiment_polarity === 1) pos++
    else if (c.sentiment_polarity === -1) neg++
    else neu++
  }
  const t = comments.length || 1
  return {
    positiveRatio: pos / t,
    neutralRatio: neu / t,
    negativeRatio: neg / t,
    avgSentimentScore:
      comments.reduce((s, c) => s + c.sentiment_polarity, 0) / t,
  }
}

function topicContextLine(cr: TopicRecord | undefined): string | undefined {
  if (!cr) return undefined
  const macro = cr.macro_topic?.trim()
  const id = cr.micro_topic_id
  const kw = (cr.micro_topic_keywords || '').trim()
  const kwShort = kw.length > 72 ? `${kw.slice(0, 72)}…` : kw
  const parts: string[] = []
  if (macro) parts.push(`宏观 ${macro}`)
  if (id != null && !Number.isNaN(id)) parts.push(`微观#${id}`)
  if (kwShort) parts.push(kwShort)
  return parts.length ? parts.join(' · ') : undefined
}

function rowsToNoteCommentLines(rows: PredictedCommentRow[]): NoteCommentLine[] {
  return rows.map((r, i) => ({
    id: `${r.note_id}_${i}`,
    user: r.nickname || '用户',
    content: r.content || '',
    sentiment: polarityToSentiment(r.sentiment_polarity),
  }))
}

function rowsToTopComments(
  rows: PredictedCommentRow[],
  contentRec: TopicRecord | undefined,
  max?: number,
): TopComment[] {
  const ctx = topicContextLine(contentRec)
  const list = max != null ? rows.slice(0, max) : rows
  return list.map((r, i) => ({
    id: `${r.note_id}_${i}`,
    user: r.nickname || '用户',
    content: r.content || '',
    likes: 0,
    sentiment: polarityToSentiment(r.sentiment_polarity),
    createdAt: '',
    noteTopicContext: i === 0 ? ctx : undefined,
  }))
}

export function parseEvaluationReport(text: string): ParsedEvaluation | null {
  const accM = text.match(/Accuracy:\s*([\d.]+)/i)
  const f1M = text.match(/Macro-F1:\s*([\d.]+)/i)
  if (!accM || !f1M) return null
  const accuracy = parseFloat(accM[1])
  const macroF1 = parseFloat(f1M[1])
  const classData: ParsedEvaluation['classData'] = []
  const lineRe =
    /^\s*([^\d].+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$/gm
  let m: RegExpExecArray | null
  while ((m = lineRe.exec(text)) !== null) {
    const name = m[1].trim()
    if (!name.includes('情感') || /avg/i.test(name)) continue
    classData.push({
      name,
      precision: parseFloat(m[2]),
      recall: parseFloat(m[3]),
      f1: parseFloat(m[4]),
      support: parseInt(m[5], 10),
    })
  }
  const metrics: EvaluationMetrics = {
    accuracy,
    macroF1,
    macroAuc: 0,
    precision: classData.map(c => c.precision),
    recall: classData.map(c => c.recall),
    f1Score: classData.map(c => c.f1),
  }
  return { metrics, classData }
}

/**
 * 与 comment/train_roberta.py 默认一致：DEFAULT_MODEL_NAME（MacBERT-large）、
 * DEFAULT_MAX_LENGTH、Large 骨干下默认 train_batch×grad_accum 与学习率。
 * 若实际训练时传入 --model_name 其它值，请以 saved_model/config.json 为准并同步此处。
 */
const REAL_ROBERTA_CONFIG: RoBERTaConfig = {
  modelName:
    'Chinese MacBERT-large（哈工大·讯飞，MLM 修正 MacBERT 中文 Large；train_roberta.py 默认基座）',
  baseModel: 'hfl/chinese-macbert-large',
  numLabels: 3,
  maxLength: 512,
  batchSize: 4,
  learningRate: 1e-5,
  epochs: 20,
  trainingDate: 'checkpoint_temp · export_dir 默认 saved_model',
  trainSize: 6401,
  valSize: 800,
  testSize: 800,
}

function metricsFromPolarity(p: LivePolaritySummary): CommentMetrics {
  const t = p.total || 1
  return {
    totalComments: p.total,
    avgSentimentScore: (p.positive - p.negative) / t,
    positiveRatio: p.positive / t,
    negativeRatio: p.negative / t,
    neutralRatio: p.neutral / t,
    avgCommentLength: 0,
    replyRate: NaN,
  }
}

/** 与 vite 中 /comment/predictions 的 PREDICTIONS_API_MAX 协调；单页尽量大以减少往返，仍分页直至全量 */
const DEFAULT_PRED_PAGE_SIZE = 120_000

function resolvedPageSize(explicit?: number): number {
  if (explicit !== undefined && explicit > 0) return explicit
  const fromEnv = parseInt(String(import.meta.env.VITE_COMMENT_PRED_SAMPLE_LIMIT || ''), 10)
  if (Number.isFinite(fromEnv) && fromEnv > 0) return fromEnv
  return DEFAULT_PRED_PAGE_SIZE
}

function parseEvalVizPayload(json: unknown): EvalVizPayload | null {
  if (!json || typeof json !== 'object') return null
  const o = json as Record<string, unknown>
  const tl = o.true_labels
  const pl = o.pred_labels
  const pr = o.probs
  if (!Array.isArray(tl) || !Array.isArray(pl) || !Array.isArray(pr)) return null
  if (tl.length === 0 || tl.length !== pl.length || pr.length !== tl.length) return null
  for (const row of pr) {
    if (!Array.isArray(row) || row.length !== 3) return null
  }
  return {
    true_labels: tl.map(x => Number(x)),
    pred_labels: pl.map(x => Number(x)),
    probs: pr as [number, number, number][],
  }
}

async function fetchAllPredictions(pageSize: number): Promise<{ total: number; rows: PredictedCommentRow[] }> {
  let offset = 0
  let total = 0
  const out: PredictedCommentRow[] = []
  for (;;) {
    const resp = (await fetch(`${BASE}/comment/predictions?limit=${pageSize}&offset=${offset}`).then(r => r.json())) as {
      total?: number
      offset?: number
      limit?: number
      sample?: PredictedCommentRow[]
      error?: string
    }
    if (resp.error) throw new Error(resp.error)
    const sample = resp.sample ?? []
    total = resp.total ?? total
    out.push(...sample)
    offset += sample.length
    if (!sample.length) break
    if (total && offset >= total) break
  }
  return { total: total || out.length, rows: out }
}

export async function loadLiveCommentBundle(
  pageSize?: number,
): Promise<LiveCommentBundle> {
  const limit = resolvedPageSize(pageSize)
  const [
    polText,
    notesText,
    stopText,
    predAll,
    trainRes,
    evalText,
    evalVizRaw,
    topicPack,
  ] = await Promise.all([
    fetch(`${BASE}/comment/comment_results/prediction_stats_polarity.csv`).then(
      r => (r.ok ? r.text() : '')
    ),
    fetch(`${BASE}/comment/comment_results/prediction_stats_notes.csv`).then(r =>
      r.ok ? r.text() : ''
    ),
    fetch(`${BASE}/comment/meaningless_word.txt`).then(r =>
      r.ok ? r.text() : ''
    ),
    fetchAllPredictions(limit),
    fetch(`${BASE}/comment/training_history`).then(async (r) => {
      if (!r.ok) return { history: [], meta: null }
      try {
        const j = (await r.json()) as {
          history?: LiveCommentBundle['trainingHistory']
          meta?: LiveCommentBundle['trainingHistoryMeta']
        }
        return { history: j.history ?? [], meta: j.meta ?? null }
      } catch {
        return { history: [], meta: null }
      }
    }) as Promise<{
      history: LiveCommentBundle['trainingHistory']
      meta: LiveCommentBundle['trainingHistoryMeta']
    }>,
    fetch(`${BASE}/comment/results/evaluation_report.txt`).then(r =>
      r.ok ? r.text() : ''
    ),
    fetch(`${BASE}/comment/results/eval_viz_payload.json`).then(async r => {
      if (!r.ok) return null
      try {
        return parseEvalVizPayload(await r.json())
      } catch {
        return null
      }
    }),
    loadTopicCSV().catch((e: unknown) => {
      console.warn('[commentLiveData] loadTopicCSV 失败，评论页无法关联 content:', e)
      return { records: [] as TopicRecord[], meta: null }
    }),
  ])

  const polarity = parsePolarityCsv(polText)
  const metrics = polarity
    ? metricsFromPolarity(polarity)
    : {
        totalComments: 0,
        avgSentimentScore: 0,
        positiveRatio: 0,
        negativeRatio: 0,
        neutralRatio: 0,
        avgCommentLength: 0,
        replyRate: NaN,
      }

  const noteRank = parseNotesCsv(notesText)
  const stop = parseMeaninglessWords(stopText || '')
  const contentMap = contentNoteMap(topicPack.records)

  /** 与排名表、BERTopic CSV、rawdata 建联时统一 trim，避免隐性空白导致匹配失败 */
  const sample: PredictedCommentRow[] = predAll.rows.map(r => ({
    ...r,
    note_id: String(r.note_id ?? '').trim(),
  }))
  const totalPred = predAll.total

  let avgLen = 0
  if (sample.length) {
    avgLen =
      sample.reduce((s, r) => s + [...(r.content || '')].length, 0) /
      sample.length
  }
  metrics.avgCommentLength = Math.round(avgLen)

  const allIds = new Set(noteRank.map(n => n.noteId.trim()))
  const aggs = buildNoteAggregates(sample, allIds)

  const noteOrder = new Map(noteRank.map((n, i) => [n.noteId, i]))
  aggs.sort(
    (a, b) =>
      (noteOrder.get(a.noteId) ?? 99999) - (noteOrder.get(b.noteId) ?? 99999)
  )

  const rankMap = new Map(noteRank.map(n => [n.noteId, n.commentCount]))
  const topics: CommentTopic[] = aggs.map((a, idx) => {
    const { positiveRatio, neutralRatio, negativeRatio, avgSentimentScore } =
      ratio3(a.comments)
    const fullCount = rankMap.get(a.noteId) ?? a.comments.length
    const cr = contentMap.get(a.noteId)
    const macro = cr?.macro_topic?.trim() || ''
    const contentFull = (cr?.content || '').trim()
    const descFull = (cr?.desc || '').trim()
    const predN = a.comments.length
    return {
      id: idx + 1,
      name: displayTitleFromContent(cr, a.noteId),
      noteId: a.noteId,
      noteUrl: cr?.note_url?.trim() || undefined,
      noteContent: contentFull || undefined,
      noteDesc: descFull || undefined,
      noteComments: rowsToNoteCommentLines(a.comments),
      contentMacroTopic: macro || undefined,
      contentMatched: Boolean(cr),
      contentMicroTopicId: cr?.micro_topic_id,
      contentMicroKeywords: cr?.micro_topic_keywords?.trim() || undefined,
      contentMappingConfidence: cr?.confidence,
      commentCount: fullCount,
      sentimentFromCount: predN,
      keywords: topKeywordsForNote(a.comments, stop),
      positiveRatio,
      neutralRatio,
      negativeRatio,
      avgSentimentScore: sentimentScore01(avgSentimentScore),
      trend: [],
    }
  })

  const noteDetails: CommentAnalysis[] = aggs.map((a, idx) => {
    const { positiveRatio, neutralRatio, negativeRatio } = ratio3(a.comments)
    const maj =
      positiveRatio >= neutralRatio && positiveRatio >= negativeRatio
        ? 'positive'
        : negativeRatio >= neutralRatio
          ? 'negative'
          : 'neutral'
    const fullCount = rankMap.get(a.noteId) ?? a.comments.length
    const cr = contentMap.get(a.noteId)
    const macro = cr?.macro_topic?.trim()
    const contentFull = (cr?.content || '').trim()
    const descFull = (cr?.desc || '').trim()
    const predN = a.comments.length
    return {
      id: `n_${a.noteId}`,
      noteId: a.noteId,
      noteUrl: cr?.note_url?.trim() || undefined,
      noteTitle: displayTitleFromContent(cr, a.noteId),
      topicId: idx,
      topicName: macro || '（content 主题表中无此笔记）',
      contentMatched: Boolean(cr),
      contentMicroTopicId: cr?.micro_topic_id,
      contentMicroKeywords: cr?.micro_topic_keywords?.trim() || undefined,
      contentTopicKeywords: cr?.keywords?.trim() || undefined,
      contentMappingConfidence: cr?.confidence,
      noteContent: contentFull || undefined,
      noteDesc: descFull || undefined,
      sentiment: maj,
      sentimentScore: sentimentScore01(
        a.comments.reduce((s, c) => s + c.sentiment_polarity, 0) /
          (a.comments.length || 1)
      ),
      keywords: topKeywordsForNote(a.comments, stop, 8),
      commentCount: fullCount,
      sentimentSampleSize: predN,
      avgCommentLikes: 0,
      topComments: rowsToTopComments(a.comments, cr),
    }
  })

  let highCommentNotesMatched = 0
  for (const a of aggs) {
    if (contentMap.has(a.noteId)) highCommentNotesMatched += 1
  }

  const wordCloud = buildWordFreq(sample, stop, 40)
  const lengthBySentiment = buildLengthBuckets(sample)
  const ipDistribution = buildIpDistribution(sample)

  const geoMacroAcc = new Map<string, number>()
  for (const r of sample) {
    const nid = String(r.note_id ?? '').trim()
    if (!nid) continue
    const cr = contentMap.get(nid)
    const macro = (cr?.macro_topic || '（未关联 content）').trim() || '（未关联 content）'
    const region = r.ip_location?.trim() || '未知'
    const key = `${region}\u0001${macro}`
    geoMacroAcc.set(key, (geoMacroAcc.get(key) || 0) + 1)
  }
  const geoMacroMix: GeoMacroRow[] = [...geoMacroAcc.entries()]
    .map(([k, count]) => {
      const [region, macroTopic] = k.split('\u0001')
      return { region, macroTopic, count }
    })
    .sort((a, b) => b.count - a.count)
  const evalParse = parseEvaluationReport(evalText)
  const trainingHistory = trainRes.history ?? []
  const trainingHistoryMeta = trainRes.meta ?? null
  const evalViz = evalVizRaw

  if (topicPack.records.length > 0) {
    console.log(
      `[commentLiveData] content 关联：高评论笔记 ${highCommentNotesMatched}/${aggs.length} 条命中 BERTopic（content 共 ${topicPack.records.length} 行）`,
    )
  }

  return {
    polarity,
    metrics,
    noteRank,
    topics,
    noteDetails,
    wordCloud,
    lengthBySentiment,
    ipDistribution,
    geoMacroMix,
    predictionsMeta: { total: totalPred, sampleSize: sample.length },
    evalParse,
    evalViz,
    robertaConfig: REAL_ROBERTA_CONFIG,
    trainingHistory,
    trainingHistoryMeta,
    contentJoinSummary: {
      contentTopicRows: topicPack.records.length,
      highCommentNotesMatched,
      highCommentNotesTotal: aggs.length,
    },
  }
}
