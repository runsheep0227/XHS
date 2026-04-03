/**
 * 环境检测脚本
 * 运行方式: node scripts/check-setup.js
 *
 * 检测内容：
 * 1. Node.js 版本
 * 2. pnpm 是否安装
 * 3. content 目录是否存在
 * 4. 数据文件是否齐全
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { execSync } from 'node:child_process'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
// content 目录在 visualization 的上一级
const CONTENT_ROOT = path.resolve(ROOT, '..', 'content')

function exists(p) {
  try { fs.accessSync(p); return true } catch { return false }
}

function formatSize(bytes) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

console.log()
console.log('\u2550'.repeat(52))
console.log('   \u{1F4CA}  小红书 AI 可视化 — 环境检测')
console.log('\u2550'.repeat(52))
console.log()

// 1. Node.js 版本
console.log('[\u{1F4BB}] 1/4  Node.js 版本检查')
const nodeVersion = process.version
const major = parseInt(nodeVersion.slice(1).split('.')[0])
console.log(`    ${major >= 18 ? '\u2705' : '\u274C'}  Node.js ${nodeVersion}  ${major >= 18 ? '(满足要求)' : '(需要 >= 18)'}`)
console.log()

// 2. pnpm 检查
console.log('[\u{1F4BC}] 2/4  pnpm 检查')
try {
  const v = execSync('pnpm --version', { encoding: 'utf-8', stdio: 'pipe' }).trim()
  console.log(`    \u2705  pnpm ${v} 已安装`)
} catch {
  console.log('    \u26A0\uFE0F  pnpm 未安装，运行: npm install -g pnpm')
}
console.log()

// 3. content 目录
console.log('[\u{1F4C1}] 3/4  content 目录检查')
if (exists(CONTENT_ROOT)) {
  console.log(`    \u2705  content 目录: ${CONTENT_ROOT}`)
} else {
  console.log(`    \u274C  content 目录不存在`)
  console.log(`         期望路径: ${CONTENT_ROOT}`)
  console.log('         \u2192 请创建目录并放入数据文件（见 README.md）')
}
console.log()

// 4. 数据文件
console.log('[\u{1F4C4}] 4/4  数据文件检查')
const checks = [
  {
    dir: path.join(CONTENT_ROOT, 'rawdata'),
    file: 'search_contents_2026-01-29.json',
    alt: f => f.startsWith('search_contents') && f.endsWith('.json'),
    desc: '原始笔记数据（JSON）',
  },
  {
    dir: path.join(CONTENT_ROOT, 'bertopic_results_optimized'),
    file: 'final_pro_topics.csv',
    alt: f => f.endsWith('.csv'),
    desc: 'BERTopic 分类结果（CSV）',
  },
]

let allOk = true
for (const { dir, file, alt, desc } of checks) {
  if (!exists(dir)) {
    console.log(`    \u26A0\uFE0F  目录不存在: ${dir}`)
    allOk = false
    continue
  }
  const filePath = path.join(dir, file)
  if (exists(filePath)) {
    const size = formatSize(fs.statSync(filePath).size)
    console.log(`    \u2705  ${file}  (${size})`)
  } else {
    // 找替代文件
    let found = null
    if (exists(dir)) {
      const files = fs.readdirSync(dir)
      found = files.find(f => alt(f))
      if (found) {
        const size = formatSize(fs.statSync(path.join(dir, found)).size)
        console.log(`    \u2705  ${found}  (${size})  \u2192 将自动替代 ${file}`)
      }
    }
    if (!found) {
      console.log(`    \u274C  缺少: ${file}  (${desc})`)
      allOk = false
    }
  }
}
console.log()

// 总结
console.log('\u2550'.repeat(52))
const hasContent = exists(CONTENT_ROOT)
if (hasContent && allOk) {
  console.log('  \u{1F389} 环境就绪！运行: pnpm run dev')
} else if (hasContent) {
  console.log('  \u26A0\uFE0F  部分文件缺失，页面将使用内置模拟数据展示')
  console.log('  \u2192 放入数据文件后重启即可生效')
} else {
  console.log('  \u274C  content 目录未找到')
  console.log()
  console.log('  请按以下结构创建目录并放入数据文件:')
  console.log()
  console.log('  E:\\document\\PG\\studio\\')
  console.log('  ├── content\\              \u2190 创建这个目录')
  console.log('  │   ├── rawdata\\')
  console.log('  │   │   └── search_contents_2026-01-29.json')
  console.log('  │   └── bertopic_results_optimized\\')
  console.log('  │       └── final_pro_topics.csv')
  console.log('  └── visualization\\        \u2192 当前项目（pnpm run dev）')
}
console.log('\u2550'.repeat(52))
console.log()
