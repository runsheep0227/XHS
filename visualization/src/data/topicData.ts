// ============================================================
// 数据加载层
// 笔记主题页数据**仅**来自仓库 content/，开发态由 Vite 同源读盘提供：
//   - content/bertopic_results_optimized/*.csv（主表 final_pro_topics）
//   - content/rawdata/*.json（按 note_id 合并，补互动/标签/IP/类型）
// 开发态：content / comment 均由 vite.config 挂在与前端相同的端口（默认 1306），请求使用同源 ''。
// 生产构建若前后端不同源，可单独设置 VITE_CONTENT_SERVER 为数据接口根 URL（须自行托管相同路径）。
// loadTopicCSV() 在单次页面会话内缓存结果，主题页与评论页共用，避免重复拉取与合并。
// ============================================================

import { MACRO_ANCHOR_ORDER } from '../theme/macroTopicColors'

function resolveContentServerBase(): string {
  const v = import.meta.env.VITE_CONTENT_SERVER as string | undefined
  if (v !== undefined && v.trim() !== '') {
    return v.replace(/\/$/, '')
  }
  return ''
}

export const CONTENT_SERVER_BASE = resolveContentServerBase()

const CONTENT_SERVER = CONTENT_SERVER_BASE

/** 最近一次成功加载笔记主题数据时的来源说明（便于界面展示） */
export interface TopicDataLoadMeta {
  /** 实际请求的 BERTopic CSV 文件名 */
  bertopicCsv: string
  /** BERTopic CSV 在磁盘上的相对 content 根的路径 */
  bertopicRelativePath: string
  /** 扫描到的 rawdata JSON 文件个数 */
  rawdataJsonFileCount: number
  /** 合并去重后的原始笔记条数 */
  rawNotesMerged: number
  /** CSV 解析后的行数（与笔记条数一致） */
  rowCount: number
  /** 与 raw 成功拼上的笔记条数 */
  rowsJoinedWithRaw: number
  contentServer: string
}

// ============================================================
// 接口定义
// ============================================================

export interface RawNote {
  note_id: string
  type: string
  title: string
  desc: string
  liked_count: string
  collected_count: string
  comment_count: string
  share_count: string
  ip_location: string
  tag_list: string
  time: number
  note_url: string
  nickname: string
  // ── 新增字段（来自原始 JSON）──
  video_url?: string
  source_keyword?: string
  image_list?: string
}

export interface TopicRecord {
  note_id: string
  content: string
  segmented_text: string
  micro_topic_id: number
  micro_topic_keywords: string
  micro_topic_full_keywords: string
  macro_topic: string
  keywords: string
  note_count: number
  confidence: number
  liked_count?: number
  collected_count?: number
  comment_count?: number
  share_count?: number
  title?: string
  desc?: string
  time?: number
  ip_location?: string
  tag_list?: string
  note_url?: string
  nickname?: string
  // ── 新增字段（来自原始 JSON 关联合并）──
  type?: 'normal' | 'video'   // 图文 / 视频
  source_keyword?: string      // 采集来源关键词
  image_count?: number         // 图片数量
  /** BERTopic / CSV 噪声样本标记（final_pro_topics 等含 is_noise 列时） */
  is_noise?: boolean
}

export interface Topic {
  id: number
  name: string
  noteCount: number
  keywords: string[]
  microTopics: MicroTopic[]
  avgLikes: number
  avgComments: number
  avgCollects: number
  avgShares: number
  avgConfidence: number
  confidenceRate: number
  macroTopic: string
  rawRecords: TopicRecord[]
}

export interface MicroTopic {
  id: number
  name: string
  keywords: string[]
  noteCount: number
  avgLikes: number
  avgComments: number
  avgCollects: number
  avgShares: number
  avgConfidence: number
}

export interface Note {
  id: string
  title: string
  content: string
  likes: number
  comments: number
  shares: number
  collects: number
  discusses: number
  topicId: number
  topicName: string
  macroTopicName: string
  microTopicId: number
  keywords: string[]
  createdAt: string
  ipLocation: string
  noteUrl: string
  confidence: number
  commentList: Comment[]
}

export interface Comment {
  id: string
  user: string
  content: string
  likes: number
  createdAt: string
}

// ============================================================
// CSV 解析（处理引号包围字段）
// ============================================================

function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') { current += '"'; i++ }
      else inQuotes = !inQuotes
    } else if (ch === ',' && !inQuotes) {
      result.push(current); current = ''
    } else {
      current += ch
    }
  }
  result.push(current)
  return result
}

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split('\n')
  if (lines.length < 2) return []
  const headers = parseCSVLine(lines[0]).map(h => h.replace(/^\ufeff/g, '').trim())
  return lines.slice(1).map(line => {
    const values = parseCSVLine(line)
    const row: Record<string, string> = {}
    headers.forEach((h, i) => {
      if (!h) return
      row[h] = (values[i] || '').trim()
    })
    return row
  })
}

function parseNum(val: string | undefined, fallback = 0): number {
  if (!val || val === '') return fallback
  const n = parseFloat(val.replace(/,/g, '').trim())
  return isNaN(n) ? fallback : n
}

function parseNoiseCell(val: string | undefined): boolean | undefined {
  if (val === undefined || val === '') return undefined
  const s = val.trim().toLowerCase()
  if (['true', '1', 'yes'].includes(s)) return true
  if (['false', '0', 'no'].includes(s)) return false
  return undefined
}

/** 将 raw 的 time（秒/毫秒时间戳或 ISO 字符串）统一为 UTC 毫秒；无效则 undefined */
export function normalizeTimeToMs(v: unknown): number | undefined {
  if (v === undefined || v === null || v === '') return undefined
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return undefined
    const ms = v < 1e12 ? Math.round(v * 1000) : Math.round(v)
    const d = new Date(ms)
    return Number.isNaN(d.getTime()) ? undefined : ms
  }
  if (typeof v === 'string') {
    const s = v.trim()
    if (!s) return undefined
    if (/^\d+(\.\d+)?$/.test(s)) return normalizeTimeToMs(Number(s))
    const d = new Date(s)
    return Number.isNaN(d.getTime()) ? undefined : d.getTime()
  }
  return undefined
}

/** 从已合并的 TopicRecord 取发布时间（毫秒） */
export function getRecordTimeMs(r: TopicRecord): number | undefined {
  return normalizeTimeToMs(r.time as unknown)
}

export function formatTopicTimestamp(ms: number | undefined): string {
  if (ms === undefined || !Number.isFinite(ms)) return ''
  const d = new Date(ms)
  if (Number.isNaN(d.getTime())) return ''
  try {
    return d.toLocaleString('zh-CN', { dateStyle: 'medium', timeStyle: 'short' })
  } catch {
    return d.toISOString()
  }
}

function macroKeyForTemporal(r: TopicRecord): string {
  const m = (r.macro_topic || '未分类').trim() || '未分类'
  const low = m.toLowerCase()
  if (r.is_noise === true || low.includes('噪声') || low.includes('outlier')) return '噪声数据'
  return m
}

function monthPeriodKey(d: Date): string {
  const y = d.getFullYear()
  const mo = d.getMonth() + 1
  return `${y}-${String(mo).padStart(2, '0')}`
}

function monthLabel(period: string): string {
  const [ys, ms] = period.split('-')
  return `${ys}年${parseInt(ms, 10)}月`
}

/** 本地时区：周一为一周起点，键为 yyyy-MM-dd（周一日期） */
function weekPeriodKey(d: Date): string {
  const c = new Date(d.getFullYear(), d.getMonth(), d.getDate())
  const day = c.getDay()
  const diff = day === 0 ? -6 : 1 - day
  c.setDate(c.getDate() + diff)
  const y = c.getFullYear()
  const m = c.getMonth() + 1
  const dayM = c.getDate()
  return `${y}-${String(m).padStart(2, '0')}-${String(dayM).padStart(2, '0')}`
}

function weekLabel(period: string): string {
  return `周起 ${period}`
}

export interface TemporalMacroBucketRow {
  period: string
  label: string
  total: number
  [macro: string]: string | number
}

/**
 * 按「月」或「周」聚合各宏观主题的笔记篇数，用于时序演变图。
 * 依赖 rawdata JSON 合并后的 `time` 字段；秒/毫秒时间戳与 ISO 字符串均可。
 */
export function buildTemporalMacroSeries(
  records: TopicRecord[],
  bucket: 'month' | 'week' = 'month',
): {
  rows: TemporalMacroBucketRow[]
  macroKeys: string[]
  withTimeCount: number
  missingTimeCount: number
  totalCount: number
  coveragePct: number
} {
  const totalCount = records.length
  let withTimeCount = 0
  const periodMacro = new Map<string, Map<string, number>>()

  for (const r of records) {
    const ms = getRecordTimeMs(r)
    if (ms === undefined) continue
    withTimeCount++
    const d = new Date(ms)
    const period = bucket === 'month' ? monthPeriodKey(d) : weekPeriodKey(d)
    const macro = macroKeyForTemporal(r)
    if (!periodMacro.has(period)) periodMacro.set(period, new Map())
    const inner = periodMacro.get(period)!
    inner.set(macro, (inner.get(macro) || 0) + 1)
  }

  const macroTotals = new Map<string, number>()
  for (const inner of periodMacro.values()) {
    for (const [m, c] of inner) macroTotals.set(m, (macroTotals.get(m) || 0) + c)
  }

  const orderedMacros: string[] = []
  for (const name of MACRO_ANCHOR_ORDER) {
    if (macroTotals.has(name)) orderedMacros.push(name)
  }
  const rest = [...macroTotals.keys()]
    .filter((k) => !orderedMacros.includes(k))
    .sort((a, b) => (macroTotals.get(b) || 0) - (macroTotals.get(a) || 0))
  orderedMacros.push(...rest)

  const periods = [...periodMacro.keys()].sort()
  const rows: TemporalMacroBucketRow[] = periods.map((period) => {
    const inner = periodMacro.get(period)!
    let total = 0
    const row: TemporalMacroBucketRow = {
      period,
      label: bucket === 'month' ? monthLabel(period) : weekLabel(period),
      total: 0,
    }
    for (const m of orderedMacros) {
      const c = inner.get(m) || 0
      row[m] = c
      total += c
    }
    row.total = total
    return row
  })

  const missingTimeCount = totalCount - withTimeCount
  const coveragePct = totalCount > 0 ? (withTimeCount / totalCount) * 100 : 0

  return {
    rows,
    macroKeys: orderedMacros,
    withTimeCount,
    missingTimeCount,
    totalCount,
    coveragePct,
  }
}

// ============================================================
// 文件清单获取
// ============================================================

async function fetchManifest() {
  try {
    const resp = await fetch(`${CONTENT_SERVER}/__manifest__`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    return await resp.json() as { rawdata: string[]; bertopic: string[] }
  } catch {
    return null
  }
}

type Manifest = { rawdata: string[]; bertopic: string[] } | null

/**
 * 合并 content/rawdata 下全部 JSON（按 note_id 去重，后出现的文件覆盖先前的同 id，便于取较新抓取）
 * @param manifest 若已拉过 __manifest__ 可传入，避免重复请求
 */
async function fetchAllRawNotesMap(manifest?: Manifest): Promise<{
  map: Map<string, RawNote>
  fileCount: number
}> {
  const m = manifest ?? (await fetchManifest())
  const rawFiles = (m?.rawdata || []).filter((f) => f.endsWith('.json'))
  const byId = new Map<string, RawNote>()
  const chunks = await Promise.all(
    rawFiles.map(async (file) => {
      try {
        const resp = await fetch(`${CONTENT_SERVER}/rawdata/${encodeURIComponent(file)}`)
        if (!resp.ok) return [] as unknown[]
        const chunk: unknown = await resp.json()
        return Array.isArray(chunk) ? chunk : []
      } catch {
        console.warn(`[DataLoader] 跳过 raw 文件: ${file}`)
        return [] as unknown[]
      }
    }),
  )
  for (let i = 0; i < rawFiles.length; i++) {
    for (const item of chunks[i] || []) {
      const n = item as RawNote
      if (n?.note_id) byId.set(String(n.note_id).trim(), n)
    }
  }
  return { map: byId, fileCount: rawFiles.length }
}

// ============================================================
// 数据加载（关联合并）
// ============================================================

let topicCsvCache: {
  records: TopicRecord[]
  meta: TopicDataLoadMeta | null
} | null = null
let topicCsvInflight: Promise<{
  records: TopicRecord[]
  meta: TopicDataLoadMeta | null
}> | null = null

/** 强制下次重新拉取主题表（一般无需调用；热替换数据文件时可调试用） */
export function invalidateTopicCsvCache(): void {
  topicCsvCache = null
  topicCsvInflight = null
}

/**
 * 加载 final_pro_topics.csv 并关联合并原始笔记互动数据
 * - 字段映射：macro_topic_name → macro_topic
 * - 字段映射：mapping_confidence → confidence
 * - 按 note_id 与 rawdata JSON 关联合并（补全 liked_count 等字段）
 * - 全应用共享同一份内存结果，避免主题页与评论页重复 IO/解析
 */
export async function loadTopicCSV(): Promise<{
  records: TopicRecord[]
  meta: TopicDataLoadMeta | null
}> {
  if (topicCsvCache) return topicCsvCache
  if (topicCsvInflight) return topicCsvInflight
  topicCsvInflight = loadTopicCSVOnce()
  try {
    const r = await topicCsvInflight
    topicCsvCache = r
    return r
  } finally {
    topicCsvInflight = null
  }
}

async function loadTopicCSVOnce(): Promise<{
  records: TopicRecord[]
  meta: TopicDataLoadMeta | null
}> {
  try {
    const manifest = await fetchManifest()
    const bertFiles = manifest?.bertopic || []
    // 优先加载 final_pro_topics.csv
    const targetCsv = bertFiles.find(f => f.includes('final_pro_topics')) || bertFiles[0]
    if (!targetCsv){
      console.error('[DataLoader] manifest 中无 BERTopic CSV，请确认 content/bertopic_results_optimized 下存在 csv 且已 npm run dev')
      return { records: [], meta: null }
    }

    const resp = await fetch(`${CONTENT_SERVER}/bertopic/${encodeURIComponent(targetCsv)}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const csvText = await resp.text()
    const rows = parseCSV(csvText)

    const { map: rawMap, fileCount: rawdataJsonFileCount } = await fetchAllRawNotesMap(manifest)
    if (rawMap.size > 0) {
      console.log(`[DataLoader] 已从 rawdata 合并 ${rawMap.size} 条原始笔记（${rawdataJsonFileCount} 个 JSON）`)
    } else {
      console.warn('[DataLoader] 未从 rawdata 加载到笔记，互动/标签/地理等依赖原始 JSON 的图表将为空')
    }

    let rowsJoinedWithRaw = 0
    const records: TopicRecord[] = rows.map(row => {
      // 从 CSV 读取 BERTopic 分类结果（note_id 与 comment/rawdata 对齐须 trim，且兼容 UTF-8 BOM 表头）
      const noteId: string = (row['note_id'] || '').trim()
      const raw = rawMap.get(noteId)
      if (raw) rowsJoinedWithRaw += 1

      // 字段名映射
      const macro_topic: string = row['macro_topic_name'] || row['macro_topic'] || '未分类'
      const confidence: number = parseFloat(row['mapping_confidence'] || row['confidence'] || '0')

      return {
        note_id: noteId,
        content: row['content'] || '',
        segmented_text: row['segmented_text'] || '',
        micro_topic_id: parseInt(row['micro_topic_id'] || '0', 10),
        micro_topic_keywords: row['micro_topic_keywords'] || '',
        micro_topic_full_keywords: row['micro_topic_full_keywords'] || '',
        macro_topic,                    // 映射：macro_topic_name → macro_topic
        keywords: row['keywords'] || '',
        note_count: parseNum(row['note_count']),
        confidence,                     // 映射：mapping_confidence → confidence
        // 以下字段从 raw JSON 关联合并（若缺失则为空）
        liked_count: raw ? parseNum(raw.liked_count) : undefined,
        collected_count: raw ? parseNum(raw.collected_count) : undefined,
        comment_count: raw ? parseNum(raw.comment_count) : undefined,
        share_count: raw ? parseNum(raw.share_count) : undefined,
        title: raw?.title,
        desc: raw?.desc,
        time: raw ? normalizeTimeToMs(raw.time) : undefined,
        ip_location: raw?.ip_location,
        tag_list: raw?.tag_list,
        note_url: raw?.note_url,
        nickname: raw?.nickname,
        source_keyword: raw?.source_keyword,
        image_count: raw?.image_list ? raw.image_list.split(',').filter(Boolean).length : 0,
        type: raw
          ? raw.video_url || String(raw.type || '') === 'video'
            ? 'video'
            : 'normal'
          : undefined,
        is_noise: parseNoiseCell(row['is_noise']),
      }
    })

    const meta: TopicDataLoadMeta = {
      bertopicCsv: targetCsv,
      bertopicRelativePath: `bertopic_results_optimized/${targetCsv}`,
      rawdataJsonFileCount,
      rawNotesMerged: rawMap.size,
      rowCount: records.length,
      rowsJoinedWithRaw,
      contentServer: CONTENT_SERVER,
    }
    console.log(
      `[DataLoader] 主题数据仅来自 content/: ${meta.bertopicRelativePath} + rawdata/（${meta.rawdataJsonFileCount} 文件，${meta.rowsJoinedWithRaw}/${meta.rowCount} 行已关联合并）`,
    )
    return { records, meta }

  } catch (e: unknown) {
    console.error('[DataLoader] loadTopicCSV 失败:', e)
    return { records: [], meta: null }
  }
}

export async function loadRawNotes(): Promise<RawNote[]> {
  try {
    const { map } = await fetchAllRawNotesMap()
    return Array.from(map.values())
  } catch {
    return []
  }
}

// ============================================================
// topic_distribution_stats.csv 加载（宏观主题关键词）
// 字段：macro_topic, micro_topic_id, note_count, keywords, confidence
// ============================================================

export interface TopicStats {
  macro_topic: string
  micro_topic_id: number
  note_count: number
  keywords: string[]
  confidence: number
}

/**
 * 加载 topic_distribution_stats.csv，返回宏观主题关键词数据
 */
export async function loadTopicStats(): Promise<TopicStats[]> {
  try {
    const manifest = await fetchManifest()
    const bertFiles = manifest?.bertopic || []
    const statsFile = bertFiles.find((f) => f.includes('topic_distribution_stats'))
    if (!statsFile) return []

    const resp = await fetch(`${CONTENT_SERVER}/bertopic/${encodeURIComponent(statsFile)}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const csvText = await resp.text()
    const rows = parseCSV(csvText)

    const stats: TopicStats[] = rows.map(row => ({
      macro_topic: row['macro_topic'] || row['macro_topic_name'] || '',
      micro_topic_id: parseInt(row['micro_topic_id'] || '0', 10),
      note_count: parseNum(row['note_count']),
      keywords: (row['keywords'] || '')
        .split(',')
        .map((k: string) => k.trim())
        .filter(Boolean),
      confidence: parseFloat(row['confidence'] || '0'),
    })).filter(s => s.macro_topic !== '')

    console.log(`[DataLoader] 已加载 ${stats.length} 条主题统计数据`)
    return stats
  } catch (e) {
    console.warn('[DataLoader] loadTopicStats 失败:', e)
    return []
  }
}

/** 从主题统计构建全局关键词词云（带权重） */
export function buildKeywordCloud(stats: TopicStats[]) {
  if (!stats || stats.length === 0) return []
  // 权重 = note_count * confidence，关键词按 macro_topic 分组
  return stats
    .filter(s => s.keywords.length > 0)
    .flatMap(s =>
      s.keywords.map(keyword => ({
        keyword,
        macro_topic: s.macro_topic,
        note_count: s.note_count,
        confidence: s.confidence,
        // 权重：综合考虑笔记数量和置信度
        weight: Math.round(s.note_count * s.confidence),
      }))
    )
    .sort((a, b) => b.weight - a.weight)
}

/** 无 topic_distribution_stats.csv 时，用 final_pro_topics 行内的微观主题词生成词云权重 */
export function buildKeywordCloudFromRecords(records: TopicRecord[]) {
  const freq = new Map<string, { weight: number; macro_topic: string }>()
  for (const r of records) {
    const macro = r.macro_topic || '未分类'
    const conf = r.confidence || 0.5
    const parts = (r.micro_topic_keywords || r.keywords || '')
      .split(',')
      .map((k) => k.trim())
      .filter(Boolean)
    for (const kw of parts) {
      const w = Math.max(1, Math.round(conf * 10))
      const prev = freq.get(kw) || { weight: 0, macro_topic: macro }
      prev.weight += w
      prev.macro_topic = macro
      freq.set(kw, prev)
    }
  }
  return [...freq.entries()]
    .map(([keyword, { weight, macro_topic }]) => ({
      keyword,
      macro_topic,
      note_count: 1,
      confidence: 0.5,
      weight,
    }))
    .sort((a, b) => b.weight - a.weight)
}

// ============================================================
// 数据聚合
// ============================================================

export function aggregateTopics(records?: TopicRecord[]): Topic[] {
  if (!records || records.length === 0) return []

  const data = records

  const macroMap = new Map<string, TopicRecord[]>()
  for (const record of data) {
    const key = record.macro_topic || '未分类'
    if (!macroMap.has(key)) macroMap.set(key, [])
    macroMap.get(key)!.push(record)
  }

  const topics: Topic[] = []
  let globalId = 1

  for (const [macroName, group] of macroMap.entries()) {
    const microMap = new Map<number, TopicRecord[]>()
    for (const r of group) {
      if (!microMap.has(r.micro_topic_id)) microMap.set(r.micro_topic_id, [])
      microMap.get(r.micro_topic_id)!.push(r)
    }

    const microTopics: MicroTopic[] = []
    for (const [microId, microGroup] of microMap.entries()) {
      const kwStr = microGroup[0].micro_topic_full_keywords || microGroup[0].micro_topic_keywords
      const keywords = kwStr.split(',').map(k => k.trim()).filter(Boolean).slice(0, 8)
      microTopics.push({
        id: microId,
        name: keywords[0] || `主题${microId}`,
        keywords,
        noteCount: microGroup.length,
        avgLikes: Math.round(microGroup.reduce((s, r) => s + (r.liked_count || 0), 0) / Math.max(microGroup.length, 1)),
        avgComments: Math.round(microGroup.reduce((s, r) => s + (r.comment_count || 0), 0) / Math.max(microGroup.length, 1)),
        avgCollects: Math.round(microGroup.reduce((s, r) => s + (r.collected_count || 0), 0) / Math.max(microGroup.length, 1)),
        avgShares: Math.round(microGroup.reduce((s, r) => s + (r.share_count || 0), 0) / Math.max(microGroup.length, 1)),
        avgConfidence: parseFloat((microGroup.reduce((s, r) => s + r.confidence, 0) / Math.max(microGroup.length, 1)).toFixed(3)),
      })
    }

    const csvKwSet = new Set(
      group.map(r => r.keywords).filter(Boolean).join(',').split(',').map(k => k.trim()).filter(Boolean)
    )
    const microKwStr = group.map(r => r.micro_topic_full_keywords || r.micro_topic_keywords).join(',')
    const allKeywords = [
      ...new Set([...csvKwSet, ...microKwStr.split(',').map(k => k.trim()).filter(Boolean)]),
    ].slice(0, 15)

    const confidences = group.map(r => r.confidence)
    const avgConfidence = parseFloat((confidences.reduce((s, v) => s + v, 0) / Math.max(confidences.length, 1)).toFixed(3))
    const highConfRate = confidences.filter(v => v > 0.7).length / Math.max(confidences.length, 1)

    topics.push({
      id: globalId++,
      name: macroName,
      noteCount: group.length,
      keywords: allKeywords.slice(0, 10),
      microTopics,
      avgLikes: Math.round(group.reduce((s, r) => s + (r.liked_count || 0), 0) / Math.max(group.length, 1)),
      avgComments: Math.round(group.reduce((s, r) => s + (r.comment_count || 0), 0) / Math.max(group.length, 1)),
      avgCollects: Math.round(group.reduce((s, r) => s + (r.collected_count || 0), 0) / Math.max(group.length, 1)),
      avgShares: Math.round(group.reduce((s, r) => s + (r.share_count || 0), 0) / Math.max(group.length, 1)),
      avgConfidence,
      confidenceRate: parseFloat(highConfRate.toFixed(3)),
      macroTopic: macroName,
      rawRecords: group,
    })
  }

  return topics.sort((a, b) => b.noteCount - a.noteCount)
}

export function buildInteractionMatrix(topics: Topic[]) {
  return topics.map(t => ({
    topic: t.name,
    likes: t.avgLikes,
    comments: t.avgComments,
    collects: t.avgCollects,
    shares: t.avgShares,
    discusses: 0,
  }))
}

/**
 * 为单个宏观主题生成词云用 (词, 权重) 列表。
 * 逻辑对齐 content/bertopic_visualize.py 中 plot_wordclouds：对每个微观主题取前列关键词并累加权重（此处无 c-TF-IDF 数值，用位次递减 + 笔记量平方根作为近似）。
 */
export function buildWordCloudEntriesForTopic(topic: Topic): { name: string; value: number }[] {
  const freq = new Map<string, number>()
  const macroTop = 12
  const microTop = 15

  topic.keywords.slice(0, macroTop).forEach((kw, i) => {
    const w = macroTop - i
    freq.set(kw, (freq.get(kw) || 0) + w * 2.5)
  })

  for (const mt of topic.microTopics) {
    const boost = Math.sqrt(Math.max(mt.noteCount, 1))
    mt.keywords.slice(0, microTop).forEach((kw, i) => {
      const w = (microTop - i) * boost
      freq.set(kw, (freq.get(kw) || 0) + w)
    })
  }

  return [...freq.entries()]
    .map(([name, value]) => ({ name, value: Math.round(value * 10) / 10 }))
    .filter((e) => e.name.trim().length > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 72)
}

export function toNotes(records: TopicRecord[]): Note[] {
  return records.map(r => {
    const kwStr = r.micro_topic_full_keywords || r.micro_topic_keywords
    return {
      id: r.note_id,
      title: r.title || '无标题',
      content: r.desc || r.content || '',
      likes: r.liked_count || 0,
      comments: r.comment_count || 0,
      shares: r.share_count || 0,
      collects: r.collected_count || 0,
      discusses: 0,
      topicId: r.micro_topic_id,
      topicName: r.micro_topic_keywords?.split(',')[0]?.trim() || `主题${r.micro_topic_id}`,
      macroTopicName: r.macro_topic,
      microTopicId: r.micro_topic_id,
      keywords: kwStr.split(',').map(k => k.trim()).filter(Boolean),
      createdAt: (() => {
        const ms = getRecordTimeMs(r)
        if (ms === undefined) return ''
        const d = new Date(ms)
        return Number.isNaN(d.getTime()) ? '' : d.toISOString().split('T')[0]
      })(),
      ipLocation: r.ip_location || '',
      noteUrl: r.note_url || '',
      confidence: r.confidence,
      commentList: [],
    }
  })
}

// ============================================================
// 新增聚合函数
// ============================================================

/** 图文 vs 视频 内容类型分布（无 raw 合并得到 type 时返回空数组） */
export function buildTypeDistribution(records: TopicRecord[]) {
  const valid = records.filter((r) => r.type !== undefined)
  if (valid.length === 0) return []
  const total = valid.length
  const normal = valid.filter((r) => r.type === 'normal').length
  const video = valid.filter((r) => r.type === 'video').length
  return [
    { id: 'normal', name: '图文笔记', value: normal, color: '#f43f5e', pct: normal / total },
    { id: 'video', name: '视频笔记', value: video, color: '#8b5cf6', pct: video / total },
  ]
}

/** 标签频率分析（全局 TOP N） */
export function buildTagFrequency(records: TopicRecord[], topN = 30) {
  const tagMap = new Map<string, { count: number; topics: Set<string> }>()
  for (const r of records) {
    if (!r.tag_list) continue
    const macro = r.macro_topic || '未分类'
    for (const tag of r.tag_list.split(',').map(t => t.trim()).filter(Boolean)) {
      if (!tagMap.has(tag)) tagMap.set(tag, { count: 0, topics: new Set() })
      const entry = tagMap.get(tag)!
      entry.count++
      entry.topics.add(macro)
    }
  }
  return Array.from(tagMap.entries())
    .map(([tag, { count, topics }]) => ({ tag, count, topics: Array.from(topics) }))
    .sort((a, b) => b.count - a.count)
    .slice(0, topN)
}

/** IP 属地地理分布 */
export function buildIPDistribution(records: TopicRecord[]) {
  const locMap = new Map<string, number>()
  let unknownCount = 0
  for (const r of records) {
    if (!r.ip_location || r.ip_location.trim() === '') {
      unknownCount++
      continue
    }
    locMap.set(r.ip_location, (locMap.get(r.ip_location) || 0) + 1)
  }
  const total = records.length || 1
  const knownTotal = total - unknownCount
  const items = Array.from(locMap.entries())
    .map(([location, count]) => ({
      location,
      count,
      pct: knownTotal > 0 ? count / knownTotal : 0, // 用已知IP数作为分母
    }))
    .sort((a, b) => b.count - a.count)

  if (unknownCount > 0) {
    items.push({
      location: '未知IP',
      count: unknownCount,
      pct: unknownCount / total,
    })
  }
  return items
}

/** 采集来源关键词分布 */
export function buildSourceKeywordDistribution(records: TopicRecord[]) {
  const kwMap = new Map<string, number>()
  for (const r of records) {
    if (!r.source_keyword) continue
    kwMap.set(r.source_keyword, (kwMap.get(r.source_keyword) || 0) + 1)
  }
  return Array.from(kwMap.entries())
    .map(([keyword, count]) => ({ keyword, count }))
    .sort((a, b) => b.count - a.count)
}

/** 综合互动指标统计（用于内容分析页） */
export function buildEngagementStats(records: TopicRecord[]) {
  const valid = records.filter(r => r.liked_count !== undefined)
  const getAvg = (field: 'liked_count' | 'collected_count' | 'comment_count' | 'share_count' | 'image_count') =>
    valid.length ? Math.round(valid.reduce((s, r) => s + (r[field] || 0), 0) / valid.length) : 0

  const allLikes = valid.map(r => r.liked_count || 0)
  const allCollects = valid.map(r => r.collected_count || 0)
  const allComments = valid.map(r => r.comment_count || 0)
  const allShares = valid.map(r => r.share_count || 0)

  const percentile = (arr: number[], p: number) => {
    const sorted = [...arr].sort((a, b) => a - b)
    return sorted[Math.floor(sorted.length * p)] || 0
  }

  return {
    totalNotes: records.length,
    totalImages: valid.reduce((s, r) => s + (r.image_count || 0), 0),
    avgLikes: getAvg('liked_count'),
    avgCollects: getAvg('collected_count'),
    avgComments: getAvg('comment_count'),
    avgShares: getAvg('share_count'),
    avgImageCount: getAvg('image_count'),
    p75Likes: percentile(allLikes, 0.75),
    p90Likes: percentile(allLikes, 0.90),
    p75Collects: percentile(allCollects, 0.75),
    p90Collects: percentile(allCollects, 0.90),
    normalCount: valid.filter(r => r.type === 'normal').length,
    videoCount: valid.filter(r => r.type === 'video').length,
    topLocations: buildIPDistribution(records).filter(d => d.location !== '未知IP').slice(0, 5),
    topKeywords: buildSourceKeywordDistribution(records).slice(0, 5),
  }
}
