import { CONTENT_SERVER_BASE, loadTopicCSV, type TopicRecord } from './topicData'
import type {
  CommentAnalysis,
  CommentMetrics,
  CommentTopic,
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

export interface MisclassifiedSample {
  id: number
  content: string
  trueLabel: string
  predLabel: string
  confidence: number
  correct: boolean
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
  predictionsMeta: { total: number; sampleSize: number }
  evalParse: ParsedEvaluation | null
  robertaConfig: RoBERTaConfig
  trainingHistory: { epoch: number; trainLoss: number; valLoss: number; valAcc: number; valF1: number }[]
  misclassifiedSamples: MisclassifiedSample[]
  /** comment note_id 与 content/BERTopic 关联摘要 */
  contentJoinSummary: {
    contentTopicRows: number
    highCommentNotesMatched: number
    highCommentNotesTotal: number
  }
}

const labelZh: Record<number, string> = {
  [-1]: '负向',
  0: '中性',
  1: '正向',
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
    if (!topNoteIds.has(r.note_id)) continue
    if (!map.has(r.note_id)) map.set(r.note_id, [])
    map.get(r.note_id)!.push(r)
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

function rowsToTopComments(
  rows: PredictedCommentRow[],
  contentRec: TopicRecord | undefined,
  max = 5,
): TopComment[] {
  const ctx = topicContextLine(contentRec)
  const sorted = [...rows].slice(0, 80)
  return sorted.slice(0, max).map((r, i) => ({
    id: `${r.note_id}_${i}`,
    user: r.nickname || '用户',
    content: r.content?.slice(0, 280) || '',
    likes: 0,
    sentiment: polarityToSentiment(r.sentiment_polarity),
    createdAt: '',
    noteTopicContext: ctx,
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

function parseMisclassifiedCsv(text: string, max = 12): MisclassifiedSample[] {
  const rows = parseSimpleCsv(text.trim())
  const out: MisclassifiedSample[] = []
  for (let i = 1; i < rows.length && out.length < max; i++) {
    const r = rows[i]
    if (r.length < 9) continue
    const trueLab = r[2]
    const predLab = r[3]
    const conf = r[4]
    const content = r.slice(8).join(',').replace(/^"|"$/g, '')
    const tl = parseInt(trueLab, 10)
    const pl = parseInt(predLab, 10)
    out.push({
      id: i,
      content: content.slice(0, 400),
      trueLabel: labelZh[tl] ?? String(tl),
      predLabel: labelZh[pl] ?? String(pl),
      confidence: parseFloat(conf) || 0,
      correct: false,
    })
  }
  return out
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

/** 与 vite 中 /comment/predictions 的 PREDICTIONS_API_MAX 协调；默认尽量一次拉全量做词频/分桶等 */
const DEFAULT_PRED_SAMPLE_LIMIT = 120_000

function resolvedSampleLimit(explicit?: number): number {
  if (explicit !== undefined && explicit > 0) return explicit
  const fromEnv = parseInt(
    String(import.meta.env.VITE_COMMENT_PRED_SAMPLE_LIMIT || ''),
    10,
  )
  if (Number.isFinite(fromEnv) && fromEnv > 0) return fromEnv
  return DEFAULT_PRED_SAMPLE_LIMIT
}

export async function loadLiveCommentBundle(
  sampleLimit?: number,
): Promise<LiveCommentBundle> {
  const limit = resolvedSampleLimit(sampleLimit)
  const [
    polText,
    notesText,
    stopText,
    predRes,
    trainRes,
    evalText,
    misText,
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
    fetch(
      `${BASE}/comment/predictions?limit=${limit}&offset=0`,
    ).then(r => r.json()) as Promise<{
      total?: number
      sample?: PredictedCommentRow[]
      error?: string
    }>,
    fetch(`${BASE}/comment/training_history`).then(r =>
      r.ok ? r.json() : { history: [] }
    ) as Promise<{ history?: LiveCommentBundle['trainingHistory'] }>,
    fetch(`${BASE}/comment/results/evaluation_report.txt`).then(r =>
      r.ok ? r.text() : ''
    ),
    fetch(`${BASE}/comment/results/misclassified_test.csv`).then(r =>
      r.ok ? r.text() : ''
    ),
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

  if (predRes.error) {
    throw new Error(predRes.error)
  }
  const sample = predRes.sample ?? []
  const totalPred = predRes.total ?? sample.length

  let avgLen = 0
  if (sample.length) {
    avgLen =
      sample.reduce((s, r) => s + [...(r.content || '')].length, 0) /
      sample.length
  }
  metrics.avgCommentLength = Math.round(avgLen)

  const topIds = new Set(
    noteRank.slice(0, 24).map(n => n.noteId)
  )
  const aggs = buildNoteAggregates(sample, topIds)

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
    return {
      id: idx + 1,
      name: displayTitleFromContent(cr, a.noteId),
      noteId: a.noteId,
      noteUrl: cr?.note_url?.trim() || undefined,
      contentMacroTopic: macro || undefined,
      contentMatched: Boolean(cr),
      contentMicroTopicId: cr?.micro_topic_id,
      contentMicroKeywords: cr?.micro_topic_keywords?.trim() || undefined,
      contentMappingConfidence: cr?.confidence,
      commentCount: fullCount,
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
      sentiment: maj,
      sentimentScore: sentimentScore01(
        a.comments.reduce((s, c) => s + c.sentiment_polarity, 0) /
          (a.comments.length || 1)
      ),
      keywords: topKeywordsForNote(a.comments, stop, 8),
      commentCount: fullCount,
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
  const evalParse = parseEvaluationReport(evalText)
  const trainingHistory = trainRes.history ?? []
  const misclassifiedSamples = parseMisclassifiedCsv(misText, 14)

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
    predictionsMeta: { total: totalPred, sampleSize: sample.length },
    evalParse,
    robertaConfig: REAL_ROBERTA_CONFIG,
    trainingHistory,
    misclassifiedSamples,
    contentJoinSummary: {
      contentTopicRows: topicPack.records.length,
      highCommentNotesMatched,
      highCommentNotesTotal: aggs.length,
    },
  }
}
