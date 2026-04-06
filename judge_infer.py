#!/usr/bin/env python3
"""
供可视化「AI 判断」页调用：stdin 读入 JSON { "note": str, "comment": str }，
stdout 输出 JSON，与前端 JudgeResult 字段对齐。

仅使用训练/导出的模型，不做关键词等启发式兜底：
- 主题：BERTopic.load(content/bertopic_results_optimized/saved_model) + transform
- 情感：transformers 加载 comment/saved_model（三分类 MacBERT，id2label 为 -1/0/1）
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
CONTENT = REPO / "content"
COMMENT = REPO / "comment"
BERTOPIC_DIR = CONTENT / "bertopic_results_optimized" / "saved_model"
FINAL_TOPICS_CSV = CONTENT / "bertopic_results_optimized" / "final_pro_topics.csv"
COMMENT_MODEL_DIR = COMMENT / "saved_model"


def _read_stdin_json() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def _topic_err_msg(code: str) -> str:
    if code == "bertopic_not_installed":
        return "未安装 bertopic，请在训练所用环境中安装后与之一致。"
    if code == "bertopic_dir_missing":
        return f"缺少 BERTopic 导出目录：{BERTOPIC_DIR}"
    if code == "empty_note":
        return "笔记内容为空，无法推断主题。"
    if code.startswith("bertopic_load_or_transform:"):
        return f"BERTopic 加载或推理失败：{code.split(':', 1)[1]}"
    return f"主题推断失败：{code}"


def predict_sentiment(comment: str) -> tuple[str, float, str | None]:
    """非空评论：必须用本地分类模型；成功返回 (sentiment, max_prob, None)，否则 (_, _, error)。"""
    text = (comment or "").strip()
    if not text:
        return "neutral", 0.0, None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        return "neutral", 0.0, f"情感模型依赖缺失：{e}（需安装 torch、transformers）"
    if not COMMENT_MODEL_DIR.is_dir():
        return "neutral", 0.0, f"找不到评论模型目录：{COMMENT_MODEL_DIR}"
    weight_files = list(COMMENT_MODEL_DIR.glob("model.safetensors")) + list(
        COMMENT_MODEL_DIR.glob("pytorch_model.bin")
    )
    if not weight_files:
        return (
            "neutral",
            0.0,
            "comment/saved_model 下缺少 model.safetensors 或 pytorch_model.bin，请先完成训练并导出。",
        )
    try:
        tok = AutoTokenizer.from_pretrained(str(COMMENT_MODEL_DIR), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(COMMENT_MODEL_DIR), local_files_only=True
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        max_len = min(getattr(model.config, "max_position_embeddings", 512), 512)
        enc = tok(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        max_p = float(probs[pred_id].item())
        id2l = model.config.id2label
        raw = id2l[str(pred_id)]
        pol = int(raw)
        if pol == 1:
            sent = "positive"
        elif pol == -1:
            sent = "negative"
        else:
            sent = "neutral"
        return sent, max_p, None
    except Exception as e:
        return "neutral", 0.0, f"情感模型推理失败：{e}"


def _load_micro_to_macro() -> dict[int, str]:
    import pandas as pd

    if not FINAL_TOPICS_CSV.is_file():
        return {}
    df = pd.read_csv(FINAL_TOPICS_CSV, encoding="utf-8-sig")
    if "micro_topic_id" not in df.columns or "macro_topic_name" not in df.columns:
        return {}
    m: dict[int, str] = {}
    for mid, g in df.groupby("micro_topic_id"):
        try:
            k = int(mid)
        except (TypeError, ValueError):
            continue
        mode = g["macro_topic_name"].mode()
        m[k] = str(mode.iloc[0]) if len(mode) else str(g["macro_topic_name"].iloc[0])
    return m


def predict_topic_bertopic(note: str) -> tuple[str | None, float, str | None, list[str]]:
    """成功返回 (宏观主题名, 置信度, None, 关键词)；失败 (_, _, 错误码/信息, [])。"""
    text = (note or "").strip()
    if not text:
        return None, 0.0, "empty_note", []
    try:
        from bertopic import BERTopic
    except ImportError:
        return None, 0.0, "bertopic_not_installed", []
    if not BERTOPIC_DIR.is_dir():
        return None, 0.0, "bertopic_dir_missing", []
    try:
        tm = BERTopic.load(str(BERTOPIC_DIR))
        topics, probs = tm.transform([text])
        tid = int(topics[0])
        kw: list[str] = []
        try:
            info = tm.get_topic(tid)
            if info:
                kw = [w for w, _ in info[:12]]
        except Exception:
            pass
        conf = 0.45
        if probs is not None:
            try:
                import numpy as np

                arr = np.asarray(probs)
                if arr.ndim >= 2:
                    conf = float(arr[0].max()) if arr.size else 0.45
                elif arr.ndim == 1:
                    conf = float(arr.max()) if arr.size else 0.45
            except Exception:
                conf = 0.45
        micro_to_macro = _load_micro_to_macro()
        macro = micro_to_macro.get(tid, f"微观主题 #{tid}" if tid != -1 else "噪声/未归类")
        if tid == -1:
            macro = "噪声数据(Outliers)"
        return macro, min(1.0, max(0.15, conf)), None, kw
    except Exception as e:
        return None, 0.0, f"bertopic_load_or_transform:{e}", []


def main() -> None:
    data = _read_stdin_json()
    note = (data.get("note") or "").strip()
    comment = (data.get("comment") or "").strip()

    topic = ""
    topic_conf = 0.0
    keywords: list[str] = []
    topic_error: str | None = None
    sentiment_error: str | None = None
    debug: list[str] = []

    if note:
        macro, conf, terr, kw = predict_topic_bertopic(note)
        if terr is None and macro is not None:
            topic, topic_conf, keywords = macro, conf, kw
            debug.append("topic_source:bertopic_saved_model")
        else:
            topic = "—"
            topic_conf = 0.0
            keywords = []
            topic_error = _topic_err_msg(terr or "unknown")
            debug.append(f"topic_failed:{terr or 'unknown'}")
    else:
        topic = "（未填写笔记，未做主题推断）"
        topic_conf = 0.0
        debug.append("topic_skipped:no_note")

    sentiment = "neutral"
    sentiment_conf = 0.0
    if comment:
        sent, sc, serr = predict_sentiment(comment)
        if serr:
            sentiment_error = serr
            sentiment, sentiment_conf = "neutral", 0.0
            debug.append("sentiment_failed")
        else:
            sentiment, sentiment_conf = sent, sc
            debug.append("sentiment_source:comment_saved_model")
    else:
        debug.append("sentiment_skipped:no_comment")

    out: dict = {
        "topic": topic,
        "topicConfidence": round(topic_conf, 4),
        "sentiment": sentiment,
        "sentimentScore": round(sentiment_conf, 4),
        "keywords": keywords[:12],
        "debug": debug,
    }
    if topic_error:
        out["topicError"] = topic_error
    if sentiment_error:
        out["sentimentError"] = sentiment_error
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = {"error": traceback.format_exc()}
        print(json.dumps(err, ensure_ascii=False))
        sys.exit(1)
