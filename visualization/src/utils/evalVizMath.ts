/**
 * 与 comment/evaluate_model.py 中 matplotlib/sklearn 可视化一致的前端数值计算。
 */

export const EVAL_CLASS_LABELS = ['负向情感(-1)', '中性情感(0)', '正向情感(1)'] as const
/** 与 CommentAnalysis 侧栏情感条 SENT_COLORS 一致：负向 / 中性 / 正向 */
export const EVAL_COLORS = ['#f43f5e', '#94a3b8', '#10b981'] as const

export function maxSoftmaxPerRow(probs: number[][]): number[] {
  return probs.map((p) => Math.max(p[0] ?? 0, p[1] ?? 0, p[2] ?? 0))
}

/** 梯形法 AUC（与 sklearn.metrics.auc 在单调点上等价） */
export function aucTrapezoid(fpr: number[], tpr: number[]): number {
  let a = 0
  for (let i = 1; i < fpr.length; i++) {
    a += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) * 0.5
  }
  return a
}

/**
 * 二分类 ROC（y ∈ {0,1}，score 越高越正类），与 sklearn.roc_curve 在相同排序下一致。
 */
export function rocCurveBinary(yTrue: number[], scores: number[]): {
  fpr: number[]
  tpr: number[]
  auc: number
} {
  const n = yTrue.length
  const pos = yTrue.reduce((s, v) => s + v, 0)
  const neg = n - pos
  if (pos === 0 || neg === 0) {
    return { fpr: [0, 1], tpr: [0, 1], auc: NaN }
  }
  const idx = Array.from({ length: n }, (_, i) => i)
  idx.sort((a, b) => scores[b]! - scores[a]!)
  const fpr: number[] = [0]
  const tpr: number[] = [0]
  let tp = 0
  let fp = 0
  for (const i of idx) {
    if (yTrue[i]! >= 0.5) tp++
    else fp++
    fpr.push(fp / neg)
    tpr.push(tp / pos)
  }
  return { fpr, tpr, auc: aucTrapezoid(fpr, tpr) }
}

/** One-vs-Rest：第 classId 类，正类为 true_label===classId */
export function rocOvrClass(
  trueLabels: number[],
  probs: number[][],
  classId: number,
): { fpr: number[]; tpr: number[]; auc: number } {
  const y = trueLabels.map((t) => (t === classId ? 1 : 0))
  const scores = probs.map((p) => p[classId] ?? 0)
  return rocCurveBinary(y, scores)
}

/** sklearn 微平均：y_true_bin.ravel() vs probs.ravel() */
export function rocMicroOvR(
  trueLabels: number[],
  probs: number[][],
): { fpr: number[]; tpr: number[]; auc: number } {
  const y: number[] = []
  const s: number[] = []
  for (let i = 0; i < trueLabels.length; i++) {
    for (let c = 0; c < 3; c++) {
      y.push(trueLabels[i] === c ? 1 : 0)
      s.push(probs[i]?.[c] ?? 0)
    }
  }
  return rocCurveBinary(y, s)
}

export function buildConfidenceHistogram(
  trueLabels: number[],
  predLabels: number[],
  probs: number[][],
  numBins = 40,
): {
  binLabel: string
  correct: number
  wrong: number
}[] {
  const maxP = maxSoftmaxPerRow(probs)
  const rows: { binLabel: string; correct: number; wrong: number }[] = []
  for (let b = 0; b < numBins; b++) {
    const lo = b / numBins
    const hi = (b + 1) / numBins
    rows.push({
      binLabel: `${(lo * 100).toFixed(0)}–${(hi * 100).toFixed(0)}%`,
      correct: 0,
      wrong: 0,
    })
  }
  for (let i = 0; i < maxP.length; i++) {
    const v = maxP[i]!
    const ok = trueLabels[i] === predLabels[i]
    let bi = Math.floor(v * numBins)
    if (bi >= numBins) bi = numBins - 1
    if (bi < 0) bi = 0
    if (ok) rows[bi]!.correct++
    else rows[bi]!.wrong++
  }
  return rows
}

function quantileSorted(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0
  const pos = (sorted.length - 1) * q
  const base = Math.floor(pos)
  const rest = pos - base
  if (sorted[base + 1] !== undefined) {
    return sorted[base]! + rest * (sorted[base + 1]! - sorted[base]!)
  }
  return sorted[base]!
}

/** ECharts boxplot 单行：[min, Q1, median, Q3, max] */
export function boxplotRow(values: number[]): [number, number, number, number, number] | null {
  if (values.length === 0) return null
  const s = [...values].sort((a, b) => a - b)
  return [
    s[0]!,
    quantileSorted(s, 0.25),
    quantileSorted(s, 0.5),
    quantileSorted(s, 0.75),
    s[s.length - 1]!,
  ]
}

export function perClassMaxProbSamples(trueLabels: number[], probs: number[][]): number[][] {
  const maxP = maxSoftmaxPerRow(probs)
  const out: number[][] = [[], [], []]
  for (let i = 0; i < trueLabels.length; i++) {
    const c = trueLabels[i]!
    if (c >= 0 && c <= 2) out[c]!.push(maxP[i]!)
  }
  return out
}

/** 三分类（标签 0/1/2）每类 precision / recall / f1 / support，与 sklearn classification_report 逐类一致 */
export function perClassMetricsFromLabels(
  trueLabels: number[],
  predLabels: number[],
): { name: string; precision: number; recall: number; f1: number; support: number }[] {
  const n = trueLabels.length
  if (n === 0 || n !== predLabels.length) {
    return EVAL_CLASS_LABELS.map((name) => ({
      name,
      precision: 0,
      recall: 0,
      f1: 0,
      support: 0,
    }))
  }
  const C = 3
  const support = [0, 0, 0]
  for (const t of trueLabels) {
    if (t >= 0 && t < C && Number.isFinite(t)) support[Math.floor(t)]++
  }
  const rows: { name: string; precision: number; recall: number; f1: number; support: number }[] = []
  for (let c = 0; c < C; c++) {
    let tp = 0
    let fp = 0
    let fn = 0
    for (let i = 0; i < n; i++) {
      const tr = trueLabels[i]!
      const pr = predLabels[i]!
      if (tr === c && pr === c) tp++
      else if (tr !== c && pr === c) fp++
      else if (tr === c && pr !== c) fn++
    }
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0
    const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0
    rows.push({
      name: EVAL_CLASS_LABELS[c]!,
      precision: prec,
      recall: rec,
      f1,
      support: support[c]!,
    })
  }
  return rows
}

/** sklearn 的 weighted F1：各类 F1 按 support 加权平均 */
export function weightedF1FromClasses(rows: { f1: number; support: number }[]): number {
  let num = 0
  let den = 0
  for (const r of rows) {
    const s = Math.max(0, r.support)
    num += r.f1 * s
    den += s
  }
  return den > 0 ? num / den : 0
}
