/**
 * 笔记主题页「宏观主题」展示色（与 content/bertopic_visualize.py 中 MACRO_ANCHORS、
 * 数据概览环形图图例一致）。禁止按主题在列表中的序号取色。
 */

export const MACRO_ANCHOR_COLORS: Record<string, string> = {
  AI内容创作: '#E74C3C',
  'AI应用与测评': '#3498DB',
  'AI学习教程': '#2ECC71',
  'AI赋能工作生活': '#F39C12',
  'AI社会反思': '#9B59B6',
};

/** 五大锚点顺序（推断诊断等固定维度图用） */
export const MACRO_ANCHOR_ORDER = [
  'AI内容创作',
  'AI应用与测评',
  'AI学习教程',
  'AI赋能工作生活',
  'AI社会反思',
] as const;

export const SUB_OVERVIEW_FALLBACK_COLORS = [
  '#f43f5e',
  '#8b5cf6',
  '#06b6d4',
  '#22c55e',
  '#f59e0b',
  '#ec4899',
  '#14b8a6',
  '#6366f1',
  '#a855f7',
];

export function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h << 5) - h + s.charCodeAt(i);
  return h;
}

function noiseLikeMacroName(name: string): boolean {
  const n = name.trim().toLowerCase();
  return n.includes('噪声') || n.includes('outlier');
}

/**
 * 与环形图 / 桑基右侧宏观列 / 侧栏宏观主题条一致：先按名称命中五大锚点色，否则稳定哈希回退。
 */
export function macroTopicDisplayColor(rawName: string): string {
  const name = (rawName ?? '').trim();
  if (name === '噪声数据' || noiseLikeMacroName(name)) {
    return '#94a3b8';
  }
  const anchor = MACRO_ANCHOR_COLORS[name];
  if (anchor) return anchor;
  return SUB_OVERVIEW_FALLBACK_COLORS[Math.abs(hashString(name)) % SUB_OVERVIEW_FALLBACK_COLORS.length];
}

/** 将 #RRGGBB 转为 rgba，用于卡片浅底、描边等 */
export function macroTopicColorWithAlpha(rawName: string, alpha: number): string {
  const hex = macroTopicDisplayColor(rawName).replace('#', '');
  if (hex.length !== 6) return `rgba(148, 163, 184, ${alpha})`;
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
