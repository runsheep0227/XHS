import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import sourceIdentifierPlugin from 'vite-plugin-source-identifier'
import http from "http"
import fs from "fs"

// 是否为生产构建（pnpm run build:prod）
const isProd = process.env.BUILD_MODE === 'prod'

// ─── 内容数据目录 ───────────────────────────────────────────────
//
//  本项目依赖与 visualization 同级的 content 目录：
//
//   E:\document\PG\studio\
//   ├── content\                    ← 数据文件放这里
//   │   ├── rawdata\
//   │   │   └── search_contents_*.json
//   │   └── bertopic_results_optimized\
//   │       └── final_pro_topics.csv
//   └── visualization\              ← pnpm run dev 在此目录运行
//
//  如目录路径不同，修改 CONTENT_ROOT 即可。
// ────────────────────────────────────────────────────────────────
const CONTENT_ROOT = path.resolve(__dirname, '..', 'content')

// ─── 轻量文件服务器：仅在开发模式启动，托管 content 目录供前端 fetch 访问 ───
function startContentServer(port = 4173) {
  if (!fs.existsSync(CONTENT_ROOT)) {
    console.warn(`[Content Server] 目录不存在: ${CONTENT_ROOT}`)
    console.warn(`[Content Server] 请确认 content 文件夹位于: ${path.resolve(__dirname, '..')}`)
  }

  const server = http.createServer((req, res) => {
    const url = req.url || ''
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

    if (req.method === 'OPTIONS') {
      res.writeHead(204)
      res.end()
      return
    }

    try {
      if (url === '/__manifest__') {
        const rawdataDir = path.join(CONTENT_ROOT, 'rawdata')
        const bertDir = path.join(CONTENT_ROOT, 'bertopic_results_optimized')
        const rawdata = fs.existsSync(rawdataDir)
          ? fs.readdirSync(rawdataDir).filter(f => f.endsWith('.json'))
          : []
        const bertopic = fs.existsSync(bertDir)
          ? fs.readdirSync(bertDir).filter(f => f.endsWith('.csv'))
          : []
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' })
        res.end(JSON.stringify({ rawdata, bertopic }))
        return
      }

      const rawMatch = url.match(/^\/rawdata\/(.+\.json)$/)
      if (rawMatch) {
        const filePath = path.join(CONTENT_ROOT, 'rawdata', rawMatch[1])
        if (fs.existsSync(filePath)) {
          res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' })
          res.end(fs.readFileSync(filePath, 'utf-8'))
          return
        }
      }

      const bertMatch = url.match(/^\/bertopic\/(.+\.csv)$/)
      if (bertMatch) {
        const filePath = path.join(CONTENT_ROOT, 'bertopic_results_optimized', bertMatch[1])
        if (fs.existsSync(filePath)) {
          res.writeHead(200, { 'Content-Type': 'text/csv; charset=utf-8' })
          res.end(fs.readFileSync(filePath, 'utf-8'))
          return
        }
      }

      res.writeHead(404)
      res.end(`Not Found: ${url}`)
    } catch (err: any) {
      console.error('[Content Server] 错误:', err.message)
      res.writeHead(500)
      res.end(`Server Error: ${err.message}`)
    }
  })

  server.listen(port, () => {
    console.log(`[Content Server] http://localhost:${port}`)
    console.log(`[Content Server] 托管目录: ${CONTENT_ROOT}`)
  })

  // 确保构建完成后能正常退出
  server.on('error', (err: any) => {
    if (err.code === 'EADDRINUSE') {
      console.warn(`[Content Server] 端口 ${port} 已被占用，跳过启动`)
    }
  })
}

// ─── 仅在开发模式（非生产构建）启动 Content Server ───
if (!isProd) {
  startContentServer()
}

export default defineConfig({
  base: '/',
  plugins: [
    react(),
    sourceIdentifierPlugin({
      enabled: !isProd,
      attributePrefix: 'data-matrix',
      includeProps: true,
    }),
  ],
  server: {
    port: 1306,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
