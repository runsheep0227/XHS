// ============================================================
// 数据加载层
// 支持从 content 目录读取真实 BERTopic 结果 + 原始笔记数据
// 字段名映射：CSV 列名 → 可视化接口字段
// ============================================================

// 内容服务器地址（与 vite.config.ts 端口一致）
const CONTENT_SERVER = 'http://localhost:4173'

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
  trend: number[]
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
  const headers = parseCSVLine(lines[0])
  return lines.slice(1).map(line => {
    const values = parseCSVLine(line)
    const row: Record<string, string> = {}
    headers.forEach((h, i) => { row[h.trim()] = (values[i] || '').trim() })
    return row
  })
}

function parseNum(val: string | undefined, fallback = 0): number {
  if (!val || val === '') return fallback
  const n = parseFloat(val.replace(/,/g, '').trim())
  return isNaN(n) ? fallback : n
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

// ============================================================
// 数据加载（关联合并）
// ============================================================

/**
 * 加载 final_pro_topics.csv 并关联合并原始笔记互动数据
 * - 字段映射：macro_topic_name → macro_topic
 * - 字段映射：mapping_confidence → confidence
 * - 按 note_id 与 rawdata JSON 关联合并（补全 liked_count 等字段）
 */
export async function loadTopicCSV(): Promise<TopicRecord[]> {
  try {
    const manifest = await fetchManifest()
    const bertFiles = manifest?.bertopic || []
    // 优先加载 final_pro_topics.csv
    const targetCsv = bertFiles.find(f => f.includes('final_pro_topics')) || bertFiles[0]
    if (!targetCsv) throw new Error('未找到 BERTopic CSV 文件')

    const resp = await fetch(`${CONTENT_SERVER}/bertopic/${encodeURIComponent(targetCsv)}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const csvText = await resp.text()
    const rows = parseCSV(csvText)

    // 加载原始笔记（用于关联合并互动数据）
    const rawFiles = manifest?.rawdata || []
    const rawFile = rawFiles.find(f => f.includes('search_contents')) || rawFiles[0]
    let rawMap = new Map<string, RawNote>()

    if (rawFile) {
      try {
        const rawResp = await fetch(`${CONTENT_SERVER}/rawdata/${encodeURIComponent(rawFile)}`)
        if (rawResp.ok) {
          const rawData: RawNote[] = await rawResp.json()
          rawMap = new Map(rawData.map(n => [n.note_id, n]))
        }
      } catch (e) {
        console.warn('[DataLoader] 加载原始笔记失败，将仅使用 CSV 数据：', e)
      }
    }

    const records: TopicRecord[] = rows.map(row => {
      // 从 CSV 读取 BERTopic 分类结果
      const noteId: string = row['note_id'] || ''
      const raw = rawMap.get(noteId)

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
        time: raw?.time ? Number(raw.time) : undefined,
        ip_location: raw?.ip_location,
        tag_list: raw?.tag_list,
        note_url: raw?.note_url,
        nickname: raw?.nickname,
      }
    })

    console.log(`[DataLoader] 已加载 ${records.length} 条 BERTopic 记录`)
    if (rawMap.size > 0) console.log(`[DataLoader] 已关联合并 ${rawMap.size} 条原始笔记数据`)
    return records

  } catch (e: any) {
    console.error('[DataLoader] loadTopicCSV 失败:', e)
    return []
  }
}

export async function loadRawNotes(): Promise<RawNote[]> {
  try {
    const manifest = await fetchManifest()
    const rawFiles = manifest?.rawdata || []
    const rawFile = rawFiles.find(f => f.includes('search_contents')) || rawFiles[0]
    if (!rawFile) return []

    const resp = await fetch(`${CONTENT_SERVER}/rawdata/${encodeURIComponent(rawFile)}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data: RawNote[] = await resp.json()
    return data
  } catch {
    return []
  }
}

// ============================================================
// 备用：当 content 目录不可用时使用内置主题配置
// ============================================================

const FALLBACK_MACRO_TOPICS = [
  {
    name: 'AI内容创作',
    microTopics: [
      { id: 1, name: 'AI绘画', keywords: ['绘画', 'Midjourney', 'Stable Diffusion', '即梦'], noteCount: 2520, avgLikes: 820, avgComments: 87, avgCollects: 210, avgShares: 65, avgConfidence: 0.65 },
      { id: 2, name: 'AI视频', keywords: ['视频', 'Sora', '剪辑', '生成'], noteCount: 1840, avgLikes: 756, avgComments: 79, avgCollects: 195, avgShares: 58, avgConfidence: 0.71 },
    ],
  },
  {
    name: 'AI效率办公',
    microTopics: [
      { id: 10, name: 'AI写作', keywords: ['写作', 'ChatGPT', '文案', '润色'], noteCount: 2150, avgLikes: 698, avgComments: 72, avgCollects: 180, avgShares: 48, avgConfidence: 0.73 },
      { id: 11, name: 'AI数据分析', keywords: ['Python', 'Excel', '数据', '分析'], noteCount: 1320, avgLikes: 612, avgComments: 65, avgCollects: 155, avgShares: 42, avgConfidence: 0.68 },
    ],
  },
  {
    name: 'AI学习教育',
    microTopics: [
      { id: 20, name: 'AI课程', keywords: ['教程', '课程', '学习', '免费'], noteCount: 1980, avgLikes: 534, avgComments: 58, avgCollects: 145, avgShares: 38, avgConfidence: 0.70 },
      { id: 21, name: 'AI提示词', keywords: ['提示词', 'Prompt', '技巧', '工程'], noteCount: 1560, avgLikes: 487, avgComments: 54, avgCollects: 130, avgShares: 35, avgConfidence: 0.66 },
    ],
  },
  {
    name: 'AI工具测评',
    microTopics: [
      { id: 30, name: 'AI工具对比', keywords: ['测评', '对比', '推荐', '避坑'], noteCount: 1750, avgLikes: 645, avgComments: 68, avgCollects: 168, avgShares: 45, avgConfidence: 0.69 },
    ],
  },
  {
    name: 'AI赋能工作生活',
    microTopics: [
      { id: 40, name: 'AI副业', keywords: ['副业', '变现', '赚钱', '接单'], noteCount: 2100, avgLikes: 920, avgComments: 95, avgCollects: 245, avgShares: 72, avgConfidence: 0.74 },
      { id: 41, name: 'AI职场', keywords: ['职场', '工作', '效率', '晋升'], noteCount: 1430, avgLikes: 578, avgComments: 62, avgCollects: 148, avgShares: 40, avgConfidence: 0.67 },
    ],
  },
]

function generateFallbackRecords(): TopicRecord[] {
  const records: TopicRecord[] = []
  const sampleTitles = [
    '亲测好用！AI写作神器让我效率翻倍', 'Midjourney保姆级教程，新手必看',
    'Stable Diffusion本地部署完整攻略', 'ChatGPT高级用法，效率提升10倍',
    'AI头像生成也太好看了吧！', 'Notion AI太香了！知识管理神器',
    'Copilot帮我写代码，太绝了', 'AI时代如何学习？这条路分享给你',
    '用AI做副业，月入过万真实经历', '这些AI工具我真的离不开',
  ]
  const sampleContents = [
    '最近开始用AI工具辅助工作，真的效率提升了好多！特别推荐这几个...',
    '今天分享一个超好用的AI绘图工具，新手也能快速上手，附详细教程...',
    '用了三个月ChatGPT，总结了一些高级用法和提示词技巧，收藏起来...',
  ]
  const ipLocations = ['北京', '上海', '广东', '浙江', '江苏', '四川', '湖北', '山东', '河南', '福建']
  const nicknames = ['爱分享的CC', 'AI探索家', '效率达人', '科技控', '学习小能手', '创意无限']

  let idx = 0
  for (const macro of FALLBACK_MACRO_TOPICS) {
    for (const micro of macro.microTopics) {
      const batchSize = Math.max(5, Math.floor(micro.noteCount * 0.05))
      for (let i = 0; i < batchSize; i++) {
        const rid = `note_${String(idx++).padStart(6, '0')}`
        records.push({
          note_id: rid,
          content: sampleContents[idx % sampleContents.length],
          segmented_text: sampleContents[idx % sampleContents.length],
          micro_topic_id: micro.id,
          micro_topic_keywords: micro.keywords.join(','),
          micro_topic_full_keywords: micro.keywords.join(','),
          macro_topic: macro.name,
          keywords: micro.keywords.join(','),
          note_count: micro.noteCount,
          confidence: micro.avgConfidence + (Math.random() * 0.1 - 0.05),
          liked_count: micro.avgLikes + Math.floor(Math.random() * 200 - 100),
          collected_count: micro.avgCollects + Math.floor(Math.random() * 80 - 40),
          comment_count: micro.avgComments + Math.floor(Math.random() * 30 - 15),
          share_count: micro.avgShares + Math.floor(Math.random() * 20 - 10),
          title: sampleTitles[idx % sampleTitles.length],
          desc: sampleContents[idx % sampleContents.length],
          time: Date.now() - Math.floor(Math.random() * 60) * 86400000,
          ip_location: ipLocations[idx % ipLocations.length],
          tag_list: micro.keywords.slice(0, 3).join(','),
          note_url: `https://www.xiaohongshu.com/explore/${rid}`,
          nickname: nicknames[idx % nicknames.length],
        })
      }
    }
  }
  return records
}

const FALLBACK_RECORDS: TopicRecord[] = generateFallbackRecords()

// ============================================================
// 数据聚合
// ============================================================

export function aggregateTopics(_records?: TopicRecord[]): Topic[] {
  const records = _records ?? FALLBACK_RECORDS

  // 如果记录为空（加载失败），使用备用数据
  const data = records.length > 0 ? records : FALLBACK_RECORDS

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
      trend: generateTrend(data.length, group.length),
      macroTopic: macroName,
      rawRecords: group,
    })
  }

  return topics.sort((a, b) => b.noteCount - a.noteCount)
}

function generateTrend(totalRecords: number, groupCount: number): number[] {
  const base = Math.max(1, Math.floor(groupCount / 7))
  return [3, 5, 4, 7, 6, 8, 9].map(v => base * v)
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

export function buildTrendData(records: TopicRecord[]) {
  return Array.from({ length: 7 }, (_, i) => {
    const d = new Date()
    d.setDate(d.getDate() - 6 + i)
    const dateStr = d.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
    const dayCount = Math.max(5, Math.floor(records.length / 60))
    return {
      date: dateStr,
      notes: dayCount + Math.floor(Math.random() * 15),
      likes: dayCount * 12,
      comments: dayCount * 2,
      collects: dayCount * 5,
    }
  })
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
      createdAt: r.time ? new Date(r.time).toISOString().split('T')[0] : '',
      ipLocation: r.ip_location || '',
      noteUrl: r.note_url || '',
      confidence: r.confidence,
      commentList: [],
    }
  })
}
