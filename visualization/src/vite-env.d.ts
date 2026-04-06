/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 数据请求基地址；开发时不设则与页面同源（如 http://localhost:1306） */
  readonly VITE_CONTENT_SERVER?: string
  /** 评论预测 JSON 单次拉取条数上限（默认约 12 万；内存吃紧时可改小，如 20000） */
  readonly VITE_COMMENT_PRED_SAMPLE_LIMIT?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
