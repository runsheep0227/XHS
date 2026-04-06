# 小红书 AI 内容可视化分析仪表盘

> 小红书 AI 内容主题分析 + 评论情感分类的可视化仪表盘

## 📁 目录结构

```
E:\document\PG\studio\
├── content\                                 ← 【数据文件放置位置】
│   ├── rawdata\
│   │   └── search_contents_2026-01-29.json ← 原始笔记数据（JSON）
│   └── bertopic_results_optimized\
│       └── final_pro_topics.csv            ← BERTopic 分类结果（CSV）
│
└── visualization\                          ← 可视化项目（clone 到本地后使用）
    ├── src/
    │   ├── data/
    │   │   ├── topicData.ts                ← 主题数据加载层（含关联合并逻辑）
    │   │   └── commentData.ts              ← 评论分析模拟数据
    │   ├── pages/
    │   │   ├── TopicAnalysis.tsx           ← 主题分析页（6 个视图）
    │   │   ├── CommentAnalysis.tsx         ← 评论分析页（RoBERTa 模型视图）
    │   │   └── ContentJudge.tsx             ← AI 内容判断页
    │   └── vite.config.ts                  ← Vite 配置 + Content Server
    ├── package.json
    └── ...
```

---

## 🚀 快速开始

### 1. 确认数据文件就位

把以下文件放入 `content` 目录（与 `visualization` 同级）：

```
E:\document\PG\studio\content\
├── rawdata\
│   └── search_contents_2026-01-29.json    ← 原始笔记（含 title/liked_count 等）
└── bertopic_results_optimized\
    └── final_pro_topics.csv               ← BERTopic 分类结果
```

> **提示**：如果 `content` 目录不存在，手动创建即可。

### 2. 安装依赖

```bash
# 进入项目目录
cd E:\document\PG\studio\visualization

# 安装依赖（推荐使用 pnpm）
pnpm install
```

> **如果没有 pnpm**：先安装 `npm install -g pnpm`

### 3. 启动开发服务器

```bash
pnpm run dev
```

启动成功后看到 Vite 监听 **http://localhost:1306**。同一端口还会提供 `content/`、`comment/` 的读盘 API（见控制台 `[vite] studio 数据已接入本端口…`）。

直接打开 **http://localhost:1306** 即可使用。

---

## 📊 数据说明

### 数据来源与关联

| 数据文件 | 内容 | 对应字段 |
|---|---|---|
| `search_contents_2026-01-29.json` | 原始笔记（含互动数据） | `note_id`、`title`、`liked_count`、`ip_location` 等 |
| `final_pro_topics.csv` | BERTopic 分类结果 | `macro_topic_name`、`mapping_confidence` 等 |

**关联方式**：按 `note_id` 字段自动关联合并，无需手动处理。

### CSV 字段映射（自动处理）

| CSV 列名 | 可视化字段名 | 说明 |
|---|---|---|
| `macro_topic_name` | `macro_topic` | 宏观主题名 |
| `mapping_confidence` | `confidence` | 分类置信度 |
| `micro_topic_id` | `micro_topic_id` | 微观主题ID（直接使用）|
| `content` | `content` | 笔记正文（直接使用）|

### 笔记主题数据

**笔记主题分析页**使用仓库 `content/` 下数据（`final_pro_topics.csv` 与 `rawdata/*.json`）。**评论分析页**使用仓库 `comment/` 下预测与统计结果，均由开发服务器同源提供，无需第二个端口。

---

## 🛠️ 常用命令

```bash
pnpm run dev      # 开发模式（热更新，端口 1306）
pnpm run build    # 生产构建
pnpm run build:prod  # 生产构建（跳过提示）
pnpm run preview  # 预览生产构建结果
pnpm run clean   # 清除构建缓存
```

---

## ⚙️ 端口说明

| 端口 | 用途 |
|---|---|
| **1306** | 前端 + `content/`、`comment/` 数据路由（开发默认，可在 `vite.config.ts` 修改 `DEV_PORT`） |

> 生产构建不再内置读盘服务；若需离线包，请自行部署能提供相同路径的静态服务或设置 `VITE_CONTENT_SERVER`。

---

## 🔧 常见问题

### Q: 页面显示"未检测到主题数据"
**原因**：`content` 目录不存在，或文件路径不对。  
**解决**：确认目录结构为 `E:\document\PG\studio\content\...`，重启 `pnpm run dev`。

### Q: 页面加载后数据为空（0 条笔记）
**原因**：`final_pro_topics.csv` 中 `note_id` 与 `search_contents_*.json` 中的 `note_id` 无法匹配（ID 格式不同）。  
**解决**：目前数据加载兼容两种格式，如仍有问题，检查 CSV 文件是否包含 `note_id` 列。

### Q: 端口被占用
**解决**：终止占用进程，或修改 `vite.config.ts` 中的 `port: 1306`。

### Q: 控制台提示 content 或 comment 目录不存在
**解决**：确认仓库结构与 `visualization` 平级的 `content/`、`comment/` 存在；路径相对于 `visualization` 上一级目录解析。

### Q: `pnpm install` 失败
**解决**：清理缓存后重试
```bash
pnpm store prune
rm -rf node_modules
pnpm install
```

---

## 📂 项目技术栈

| 模块 | 技术 |
|---|---|
| 前端框架 | React 18 + TypeScript |
| 构建工具 | Vite 6 |
| 样式 | TailwindCSS v3 |
| 路由 | React Router v6 |
| 图表 | Recharts + 原生 SVG |
| 数据加载 | 原生 Fetch API + 自定义 Content Server |

---

## 🌐 部署到线上

```bash
pnpm run build
```

构建产物输出到 `dist/` 目录，可直接部署到任意静态托管服务（Vercel、Netlify 等）。

> ⚠️ **注意**：部署到线上时，需要将 `content` 目录中的数据文件也同步上传到服务器，或修改 `topicData.ts` 使用其他数据源。
