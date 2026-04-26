# -*- coding: utf-8 -*-
"""
开发时常驻推理服务：供 visualization Vite 转发 POST /api/judge → 本机 /judge。

约定（与 visualization/vite.config.ts 一致）：
- 环境变量 JUDGE_SERVER_HOST（默认 127.0.0.1）、JUDGE_SERVER_PORT（默认 18999）
- 启动后尽快在 stdout 打印一行 JSON：{"event": "ready"}
- GET /health → { "engine": { "loading", "loaded", "loadError" } }
- POST /judge JSON：{ "note": str, "comment": str }

依赖：bertopic、sentence-transformers、torch、transformers、pandas（与 content/comment 脚本一致）。

环境变量 JUDGE_EMBEDDING_MODEL：句向量模型名（默认 BAAI/bge-large-zh-v1.5，须与训练时一致）。
环境变量 JUDGE_NORMALIZE_EMBEDDINGS：设为 1/true 时 encode 使用归一化向量（默认关闭，与训练时全文 encode 一致）。
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# 与 content/predict.py 一致，减少 numba / 缓存问题
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

STUDIO_ROOT = Path(__file__).resolve().parent
CONTENT_ROOT = STUDIO_ROOT / "content"
COMMENT_ROOT = STUDIO_ROOT / "comment"

BERTOPIC_MODEL_DIR = Path(
    os.environ.get("JUDGE_BERTOPIC_DIR", str(CONTENT_ROOT / "bertopic_results_optimized" / "saved_model"))
)
TOPIC_STATS_CSV = CONTENT_ROOT / "bertopic_results_optimized" / "topic_distribution_stats.csv"
# 与 content/predict.py、bertopic_train 一致；保存的 BERTopic 包内往往不含可反序列化的 embedding，load 时必须显式传入
DEFAULT_EMBEDDING_MODEL = os.environ.get("JUDGE_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

LOG = logging.getLogger("judge_server")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )


def resolve_latest_comment_checkpoint(comment_root: Path) -> Path | None:
    base = comment_root / "checkpoint_temp"
    if not base.is_dir():
        return None
    best_n = -1
    best_dir: Path | None = None
    for ent in base.iterdir():
        if not ent.is_dir():
            continue
        m = re.match(r"^checkpoint-(\d+)$", ent.name)
        if not m:
            continue
        n = int(m.group(1))
        if not (ent / "config.json").is_file():
            continue
        if n > best_n:
            best_n = n
            best_dir = ent
    return best_dir


def load_micro_to_macro_csv(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mid = int(row.get("micro_topic_id", "").strip())
            except (TypeError, ValueError):
                continue
            macro = (row.get("macro_topic") or "").strip()
            if macro and mid not in out:
                out[mid] = macro
    return out


class EngineState:
    __slots__ = (
        "lock",
        "loading",
        "loaded",
        "load_error",
        "topic_model",
        "embedding_model",
        "micro_to_macro",
        "sentiment_tokenizer",
        "sentiment_model",
        "sentiment_device",
        "sentiment_dir",
        "sentiment_error",
    )

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.loading = True
        self.loaded = False
        self.load_error: str | None = None
        self.topic_model = None
        self.embedding_model = None
        self.micro_to_macro: dict[int, str] = {}
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.sentiment_device: str = "cpu"
        self.sentiment_dir: str | None = None
        self.sentiment_error: str | None = None


STATE = EngineState()


def load_models() -> None:
    """后台线程：先加载 BERTopic，再加载评论情感模型（可选）。"""
    try:
        import torch
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        with STATE.lock:
            STATE.loading = False
            STATE.load_error = f"缺少依赖：{e}（请安装 bertopic、transformers、torch 等）"
        LOG.error(STATE.load_error)
        return

    # ---- BERTopic ----
    try:
        if not BERTOPIC_MODEL_DIR.is_dir():
            raise FileNotFoundError(f"BERTopic 目录不存在：{BERTOPIC_MODEL_DIR}")
        LOG.info("加载 BERTopic：%s", BERTOPIC_MODEL_DIR)
        STATE.micro_to_macro = load_micro_to_macro_csv(TOPIC_STATS_CSV)
        if not STATE.micro_to_macro:
            LOG.warning("未读到 micro→macro 映射（%s），将只显示微观主题编号", TOPIC_STATS_CSV)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOG.info("加载句向量模型 %s（设备 %s）…", DEFAULT_EMBEDDING_MODEL, device)
        embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, device=device)
        embedding_model.max_seq_length = 512
        STATE.embedding_model = embedding_model
        # load 时传入 embedding_model；推理时仍用「外部向量 + transform(embeddings=…)」双保险（部分 bertopic 版本 load 后内部仍丢引用）
        topic_model = BERTopic.load(str(BERTOPIC_MODEL_DIR), embedding_model=embedding_model)
        STATE.topic_model = topic_model
    except Exception as e:
        with STATE.lock:
            STATE.loading = False
            STATE.load_error = f"BERTopic 加载失败：{e}"
        LOG.exception("BERTopic 加载失败")
        return

    # ---- 评论情感 MacBERT ----
    sentiment_dir = os.environ.get("JUDGE_COMMENT_MODEL_DIR")
    if sentiment_dir:
        ckpt = Path(sentiment_dir)
    else:
        ckpt = resolve_latest_comment_checkpoint(COMMENT_ROOT)

    if ckpt is None or not ckpt.is_dir():
        STATE.sentiment_error = f"未找到 comment checkpoint（请训练或设置 JUDGE_COMMENT_MODEL_DIR）"
        LOG.warning(STATE.sentiment_error)
    else:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            STATE.sentiment_device = str(device)
            LOG.info("加载情感模型：%s | %s", ckpt, device)
            tok = AutoTokenizer.from_pretrained(str(ckpt))
            mdl = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
            mdl.to(device)
            mdl.eval()
            STATE.sentiment_tokenizer = tok
            STATE.sentiment_model = mdl
            STATE.sentiment_dir = str(ckpt)
            STATE.sentiment_error = None
        except Exception as e:
            STATE.sentiment_error = str(e)
            LOG.exception("情感模型加载失败")

    with STATE.lock:
        STATE.loading = False
        STATE.loaded = True
        if STATE.load_error is None and STATE.topic_model is None:
            STATE.load_error = "主题模型未初始化"
    LOG.info("引擎就绪：主题=%s，情感=%s", "OK", "OK" if STATE.sentiment_model else STATE.sentiment_error or "跳过")


def run_topic(note_text: str) -> dict[str, Any]:
    assert STATE.topic_model is not None
    assert STATE.embedding_model is not None
    text = (note_text or "").strip()
    if not text:
        return {
            "topic": "（无笔记正文）",
            "topicConfidence": 0.0,
            "keywords": [],
            "topicError": "未提供笔记正文，无法判断主题",
        }
    # 显式编码再传入 transform，不依赖 topic_model 内部是否挂上了 embedding（避免 “No embedding model was found”）
    # 与 content/bertopic_train 中 fit 阶段对全文 encode 一致，默认不归一化（normalize 会改变与 UMAP/HDBSCAN 训练分布）
    norm = os.environ.get("JUDGE_NORMALIZE_EMBEDDINGS", "").lower() in ("1", "true", "yes")
    embeddings = STATE.embedding_model.encode(
        [text],
        show_progress_bar=False,
        normalize_embeddings=norm,
    )
    try:
        topics, probs = STATE.topic_model.transform([text], embeddings=embeddings)
    except TypeError:
        # 极少数旧版 bertopic 无 embeddings 参数：显式挂回再 transform
        tm = STATE.topic_model
        if hasattr(tm, "embedding_model"):
            tm.embedding_model = STATE.embedding_model
        topics, probs = tm.transform([text])
    tid = int(topics[0]) if topics is not None and len(topics) else -1
    conf = 0.78
    if probs is not None and len(probs) > 0:
        try:
            import numpy as np

            row = probs[0]
            if hasattr(row, "__iter__") and not isinstance(row, (str, bytes)):
                arr = np.asarray(row).ravel()
                if arr.size > 0:
                    conf = float(max(arr.max(), 0.01))
        except Exception:
            pass
    if tid == -1:
        macro = "噪声/未归类"
        label = f"{macro}（微观 Outlier）"
        conf = min(conf, 0.45)
    else:
        macro = STATE.micro_to_macro.get(tid, "")
        if macro:
            label = f"{macro} · 微观 T{tid}"
        else:
            label = f"微观主题 T{tid}"
    kws: list[str] = []
    try:
        tw = STATE.topic_model.get_topic(tid)
        if tw:
            kws = [str(w) for w, _ in tw[:12]]
    except Exception:
        pass
    return {"topic": label, "topicConfidence": conf, "keywords": kws}


def class_idx_to_sentiment(idx: int) -> str:
    # 与 comment/evaluate_model.py 训练约定：0→负向(-1)，1→中性(0)，2→正向(1)
    if idx == 0:
        return "negative"
    if idx == 1:
        return "neutral"
    return "positive"


def run_sentiment(comment_text: str) -> dict[str, Any]:
    if STATE.sentiment_model is None or STATE.sentiment_tokenizer is None:
        return {
            "sentiment": "neutral",
            "sentimentScore": 0.0,
            "sentimentError": STATE.sentiment_error or "情感模型未加载",
        }
    text = (comment_text or "").strip()
    if not text:
        return {
            "sentiment": "neutral",
            "sentimentScore": 0.0,
            "sentimentError": "未填写评论，跳过情感判断",
        }
    import torch
    import torch.nn.functional as F

    device = torch.device(STATE.sentiment_device)
    tok = STATE.sentiment_tokenizer
    mdl = STATE.sentiment_model
    max_len = 256
    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = F.softmax(logits, dim=-1)[0]
        pred = int(probs.argmax().item())
        score = float(probs[pred].item())
    return {
        "sentiment": class_idx_to_sentiment(pred),
        "sentimentScore": score,
    }


def handle_judge_payload(body: dict[str, Any]) -> dict[str, Any]:
    note = body.get("note") if isinstance(body.get("note"), str) else ""
    comment = body.get("comment") if isinstance(body.get("comment"), str) else ""
    debug: list[str] = []
    if STATE.sentiment_dir:
        debug.append(f"sentiment_model_dir={STATE.sentiment_dir}")
    debug.append(f"bertopic_dir={BERTOPIC_MODEL_DIR}")

    out: dict[str, Any] = {}
    # 主题：优先笔记正文；若仅有评论则用语料做主题推断
    topic_src = note.strip() or comment.strip()
    tpart = run_topic(topic_src if topic_src else "")
    out.update(tpart)
    spart = run_sentiment(comment)
    out.update(spart)
    if debug:
        out["debug"] = debug
    return out


class JudgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/health":
            self._send_json(404, {"error": "not found"})
            return
        with STATE.lock:
            body = {
                "ok": True,
                "engine": {
                    "loading": STATE.loading,
                    "loaded": STATE.loaded,
                    "loadError": STATE.load_error,
                    "sentimentError": STATE.sentiment_error,
                    "bertopicPath": str(BERTOPIC_MODEL_DIR),
                },
            }
        self._send_json(200, body)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/judge":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        with STATE.lock:
            loading = STATE.loading
            loaded = STATE.loaded
            err = STATE.load_error

        if loading:
            self._send_json(
                503,
                {"error": "模型加载中", "hint": "请稍后重试"},
            )
            return
        if not loaded or err:
            self._send_json(
                503,
                {"error": err or "引擎未就绪"},
            )
            return

        try:
            result = handle_judge_payload(body if isinstance(body, dict) else {})
            self._send_json(200, result)
        except Exception as e:
            LOG.exception("推理失败")
            self._send_json(500, {"error": str(e)})


def main() -> None:
    _setup_logging()
    host = os.environ.get("JUDGE_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("JUDGE_SERVER_PORT", "18999"))

    httpd = ThreadingHTTPServer((host, port), JudgeHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(json.dumps({"event": "ready"}, ensure_ascii=False), flush=True)
    LOG.info("HTTP 监听 http://%s:%s （/health, POST /judge）", host, port)

    threading.Thread(target=load_models, daemon=True).start()

    try:
        t.join()
    except KeyboardInterrupt:
        LOG.info("退出")
        httpd.shutdown()


if __name__ == "__main__":
    main()
