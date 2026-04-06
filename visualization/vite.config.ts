import path from "path"
import { fileURLToPath } from "node:url"
import { exec, spawnSync } from "node:child_process"
import react from "@vitejs/plugin-react"
import { defineConfig, type Plugin } from "vite"
import sourceIdentifierPlugin from 'vite-plugin-source-identifier'
import fs from "fs"
import type { ServerResponse } from "node:http"

const DEV_PORT = 1306

// package.json 为 "type": "module" 时，直接用 node 跑配置或按需解析路径需自行定义 __dirname
const __dirname = path.dirname(fileURLToPath(import.meta.url))

// 是否为生产构建（pnpm run build:prod）
const isProd = process.env.BUILD_MODE === 'prod'

const STUDIO_ROOT = path.resolve(__dirname, '..')
const CONTENT_ROOT = path.join(STUDIO_ROOT, 'content')
const COMMENT_ROOT = path.join(STUDIO_ROOT, 'comment')

/** comment/ 下允许直传的静态文件（相对 comment 根） */
const COMMENT_STATIC_FILES: Record<string, string> = {
  'comment_results/prediction_stats_polarity.csv': 'text/csv; charset=utf-8',
  'comment_results/prediction_stats_notes.csv': 'text/csv; charset=utf-8',
  'comment_results/prediction_stats.txt': 'text/plain; charset=utf-8',
  'results/evaluation_report.txt': 'text/plain; charset=utf-8',
  'results/01_confusion_matrix.png': 'image/png',
  'results/misclassified_test.csv': 'text/csv; charset=utf-8',
  'meaningless_word.txt': 'text/plain; charset=utf-8',
}

type WriteHeadRes = Pick<ServerResponse, 'setHeader' | 'writeHead' | 'end'>

/**
 * 处理 content/ 与 comment/ 的本地读盘路由。
 * 一律用 pathname 匹配，避免 `?query` 导致 `url === '...'` 失败。
 * @returns 已处理则 true（勿再 next）
 */
function handleContentRepoRoutes(
  rawUrl: string,
  method: string | undefined,
  res: WriteHeadRes
): boolean {
  if (method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
    res.writeHead(204)
    res.end()
    return true
  }

  if (method !== 'GET' && method !== undefined) {
    return false
  }

  let pathname = '/'
  try {
    pathname = new URL(rawUrl, 'http://127.0.0.1').pathname
  } catch {
    pathname = rawUrl.split('?')[0] || '/'
  }
  const search =
    rawUrl.includes('?') ? rawUrl.slice(rawUrl.indexOf('?')) : ''

  try {
    if (pathname === '/__manifest__') {
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
      return true
    }

    const rawMatch = pathname.match(/^\/rawdata\/(.+\.json)$/)
    if (rawMatch) {
      const filePath = path.join(CONTENT_ROOT, 'rawdata', rawMatch[1])
      if (fs.existsSync(filePath)) {
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' })
        res.end(fs.readFileSync(filePath, 'utf-8'))
        return true
      }
    }

    const bertMatch = pathname.match(/^\/bertopic\/(.+\.csv)$/)
    if (bertMatch) {
      const filePath = path.join(CONTENT_ROOT, 'bertopic_results_optimized', bertMatch[1])
      if (fs.existsSync(filePath)) {
        res.writeHead(200, { 'Content-Type': 'text/csv; charset=utf-8' })
        res.end(fs.readFileSync(filePath, 'utf-8'))
        return true
      }
    }

    if (pathname.startsWith('/comment/predictions')) {
      const u = new URL(pathname + search, 'http://127.0.0.1')
      /** 单次响应条数上限（全量 predicted_comments.json 约 7.5 万条；设大些以免截断） */
      const PREDICTIONS_API_MAX = 120_000
      const limit = Math.min(
        Math.max(1, parseInt(u.searchParams.get('limit') || '12000', 10)),
        PREDICTIONS_API_MAX,
      )
      const offset = Math.max(0, parseInt(u.searchParams.get('offset') || '0', 10))
      const fp = path.join(COMMENT_ROOT, 'comment_results', 'predicted_comments.json')
      if (!fs.existsSync(fp)) {
        res.writeHead(404, { 'Content-Type': 'application/json' })
        res.end(
          JSON.stringify({
            error: 'predicted_comments.json 不存在',
            path: 'comment/comment_results/',
          })
        )
        return true
      }
      try {
        const raw = fs.readFileSync(fp, 'utf-8')
        const arr = JSON.parse(raw) as unknown[]
        const sample = arr.slice(offset, offset + limit)
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' })
        res.end(JSON.stringify({ total: arr.length, offset, limit: sample.length, sample }))
      } catch (e: any) {
        res.writeHead(500, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: e?.message || String(e) }))
      }
      return true
    }

    if (pathname === '/comment/training_history') {
      try {
        const p = path.join(
          COMMENT_ROOT,
          'checkpoint_temp',
          'checkpoint-8000',
          'trainer_state.json'
        )
        if (!fs.existsSync(p)) {
          res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' })
          res.end(
            JSON.stringify({
              error: 'checkpoint_temp/checkpoint-8000/trainer_state.json 缺失',
            })
          )
          return true
        }
        const state = JSON.parse(fs.readFileSync(p, 'utf-8')) as {
          log_history?: Array<Record<string, unknown>>
        }
        const hist = state.log_history ?? []
        const losses: { epoch: number; loss: number }[] = []
        for (const h of hist) {
          if (typeof h.epoch === 'number' && typeof h.loss === 'number') {
            losses.push({ epoch: h.epoch, loss: h.loss })
          }
        }
        function trainLossNear(epoch: number): number {
          let best = -1
          let loss = 0
          for (const L of losses) {
            if (L.epoch <= epoch && L.epoch >= best) {
              best = L.epoch
              loss = L.loss
            }
          }
          return loss
        }
        const out: Array<{
          epoch: number
          trainLoss: number
          valLoss: number
          valAcc: number
          valF1: number
        }> = []
        for (const h of hist) {
          if (h.eval_accuracy != null && typeof h.epoch === 'number') {
            const e = h.epoch
            out.push({
              epoch: Math.round(e * 100) / 100,
              trainLoss: trainLossNear(e),
              valLoss: Number(h.eval_loss ?? 0),
              valAcc: Number(h.eval_accuracy),
              valF1: Number(h.eval_f1_macro ?? 0),
            })
          }
        }
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' })
        res.end(JSON.stringify({ history: out }))
      } catch (e: any) {
        res.writeHead(500, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: e?.message || String(e) }))
      }
      return true
    }

    if (pathname.startsWith('/comment/')) {
      const sub = pathname.slice('/comment/'.length).replace(/^\/+/, '')
      const ctype = COMMENT_STATIC_FILES[sub]
      if (ctype) {
        const filePath = path.join(COMMENT_ROOT, sub)
        const resolved = path.resolve(filePath)
        if (resolved.startsWith(path.resolve(COMMENT_ROOT)) && fs.existsSync(resolved)) {
          res.writeHead(200, { 'Content-Type': ctype })
          res.end(fs.readFileSync(resolved))
          return true
        }
      }
    }

    return false
  } catch (err: any) {
    console.error('[LocalRepo] 错误:', err.message)
    res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' })
    res.end(`Server Error: ${err.message}`)
    return true
  }
}

/** 开发时把 content/、comment/ 读盘路由挂在本机 Vite 端口（默认 1306） */
function localRepoVitePlugin(): Plugin {
  return {
    name: 'local-repo-data',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? '/'
        let pathname = '/'
        try {
          pathname = new URL(url, 'http://127.0.0.1').pathname
        } catch {
          pathname = url.split('?')[0] || '/'
        }

        if (pathname === '/api/judge' && req.method === 'POST') {
          const chunks: Buffer[] = []
          req.on('data', (c: Buffer) => {
            chunks.push(c)
          })
          req.on('end', () => {
            const body = Buffer.concat(chunks).toString('utf-8')
            const py = process.env.PYTHON_PATH || 'python'
            const script = path.join(STUDIO_ROOT, 'judge_infer.py')
            const r = spawnSync(py, [script], {
              cwd: STUDIO_ROOT,
              input: body || '{}',
              encoding: 'utf-8',
              maxBuffer: 32 * 1024 * 1024,
              timeout: 180_000,
              env: { ...process.env, PYTHONUTF8: '1' },
            })
            res.setHeader('Content-Type', 'application/json; charset=utf-8')
            if (r.error) {
              res.writeHead(500)
              res.end(
                JSON.stringify({
                  error: `无法启动 Python：${String((r.error as Error)?.message || r.error)}`,
                  hint: '可设置环境变量 PYTHON_PATH 指向 python.exe',
                }),
              )
              return
            }
            if (r.status !== 0) {
              res.writeHead(500)
              res.end(
                JSON.stringify({
                  error: r.stderr?.trim() || `Python 退出码 ${r.status}`,
                  stdout: r.stdout?.slice(0, 4000),
                }),
              )
              return
            }
            const out = (r.stdout || '').trim() || '{}'
            res.writeHead(200)
            res.end(out)
          })
          return
        }

        const handled = handleContentRepoRoutes(url, req.method, res)
        if (handled) return
        next()
      })
      server.httpServer?.once('listening', () => {
        console.log(
          `[vite] studio 数据已接入本端口：/__manifest__、/rawdata、/bertopic、/comment/*、POST /api/judge → ${CONTENT_ROOT} 与 ${COMMENT_ROOT}（推理脚本：${path.join(STUDIO_ROOT, 'judge_infer.py')}）`,
        )
        console.log(
          `[vite] 前端 http://localhost:${DEV_PORT} — 集成终端下将尝试打开编辑器内「简单浏览器」；未自动弹出时请 Ctrl+Shift+P →「Simple Browser: Show」输入上述地址`,
        )
      })
    },
  }
}

/**
 * 在 Cursor / VS Code 集成终端里启动 dev 时，用 vscode:// 打开「Simple Browser」内嵌页（非系统浏览器）。
 * - VITE_OPEN_SIMPLE_BROWSER=0：不自动打开
 * - VITE_OPEN_SIMPLE_BROWSER=1：任意终端都尝试打开
 * - 未设置：仅在检测到集成终端时自动打开
 */
function openEmbeddedSimpleBrowserPlugin(port: number): Plugin {
  let done = false
  return {
    name: 'open-embedded-simple-browser',
    configureServer(server) {
      server.httpServer?.once('listening', () => {
        if (done) return
        const off = process.env.VITE_OPEN_SIMPLE_BROWSER === '0'
        const force = process.env.VITE_OPEN_SIMPLE_BROWSER === '1'
        const inEditorTerminal =
          process.env.TERM_PROGRAM === 'vscode' ||
          Boolean(process.env.VSCODE_IPC_HOOK_CLI) ||
          Boolean(process.env.VSCODE_INJECTION)

        if (off) return

        if (!force && !inEditorTerminal) {
          console.log(
            `[vite] 未检测到集成终端，跳过自动内嵌预览。请 Ctrl+Shift+P →「Simple Browser: Show」→ http://localhost:${port}，或执行 VITE_OPEN_SIMPLE_BROWSER=1 npm run dev`,
          )
          return
        }

        done = true
        const pageUrl = `http://127.0.0.1:${port}`
        const uri = `vscode://vscode.simple-browser/show?url=${encodeURIComponent(pageUrl)}`
        const cmd =
          process.platform === 'win32'
            ? `cmd /c start "" "${uri}"`
            : process.platform === 'darwin'
              ? `open "${uri}"`
              : `xdg-open "${uri}"`

        setTimeout(() => {
          exec(cmd, err => {
            if (err) {
              console.warn(
                `[vite] 无法唤起内嵌简单浏览器，请手动：Ctrl+Shift+P → Simple Browser: Show → ${pageUrl}`,
              )
            }
          })
        }, 500)
      })
    },
  }
}

export default defineConfig({
  base: '/',
  plugins: [
    react(),
    ...(!isProd ? [localRepoVitePlugin(), openEmbeddedSimpleBrowserPlugin(DEV_PORT)] : []),
    sourceIdentifierPlugin({
      enabled: !isProd,
      attributePrefix: 'data-matrix',
      includeProps: true,
    }),
  ],
  server: {
    port: DEV_PORT,
    open: false,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
