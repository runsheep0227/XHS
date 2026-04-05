import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from torch.utils.data import DataLoader, Dataset

CHECKPOINT_VERSION = 1

# 与 train_roberta / evaluate_model 三分类一致（展示用）
POLARITY_TEXT = {
    -1: "负向情感(-1)",
    0: "中性情感(0)",
    1: "正向情感(1)",
}


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if not text.strip():
            text = " "

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


def load_records(input_path: str) -> tuple[list[dict], list[str]]:
    """返回 (行记录列表, 文本列列表)，顺序与文件一致。"""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON 输入应为对象数组")
        texts = [str(item.get("content", "") or "") for item in data]
        return data, texts

    if ext == ".csv":
        df = pd.read_csv(input_path, encoding="utf-8-sig")
        if "txt" in df.columns:
            col = "txt"
        elif "content" in df.columns:
            col = "content"
        else:
            raise ValueError("CSV 需包含列 txt 或 content")
        texts = df[col].fillna("").astype(str).tolist()
        records = df.to_dict("records")
        return records, texts

    raise ValueError(f"不支持的输入类型: {ext}（请使用 .json 或 .csv）")


def pred_to_polarity(pred_id: int, id2label: dict) -> int:
    raw = id2label[str(int(pred_id))]
    return int(raw)


def _checkpoint_path(output_file: str) -> str:
    base, _ext = os.path.splitext(output_file)
    return base + ".predict_ckpt.json"


def _file_fingerprint(path: str) -> tuple[float, int] | None:
    try:
        st = os.stat(path)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def _load_checkpoint(
    ckpt_path: str,
    input_path: str,
    model_dir: str,
    max_length: int,
    total_len: int,
    model_config_path: str,
) -> list[int] | None:
    if not os.path.isfile(ckpt_path):
        return None
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if raw.get("version") != CHECKPOINT_VERSION:
        return None
    cur_in = _file_fingerprint(input_path)
    cur_cfg = _file_fingerprint(model_config_path)
    if cur_in is None or cur_cfg is None:
        return None
    if raw.get("input_path") != input_path:
        return None
    if tuple(raw.get("input_stat", ())) != tuple(cur_in):
        return None
    if raw.get("model_dir") != model_dir:
        return None
    if int(raw.get("max_length", -1)) != int(max_length):
        return None
    if tuple(raw.get("model_config_stat", ())) != tuple(cur_cfg):
        return None
    if int(raw.get("total_len", -1)) != int(total_len):
        return None
    preds = raw.get("preds")
    if not isinstance(preds, list):
        return None
    out: list[int] = []
    for x in preds:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        out.append(int(x))
    if len(out) > total_len:
        return None
    return out


def _save_checkpoint_atomic(
    ckpt_path: str,
    payload: dict[str, Any],
) -> None:
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, ckpt_path)


def _build_ckpt_payload(
    *,
    input_path: str,
    input_stat: tuple[float, int],
    model_dir: str,
    model_config_stat: tuple[float, int],
    max_length: int,
    total_len: int,
    preds: list[int],
) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "input_path": input_path,
        "input_stat": list(input_stat),
        "model_dir": model_dir,
        "model_config_stat": list(model_config_stat),
        "max_length": int(max_length),
        "total_len": int(total_len),
        "preds": preds,
    }


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_json = os.path.join(current_dir, "bert_data", "final_cleaned_comments.json")
    default_out_dir = os.path.join(current_dir, "comment_results")
    p = argparse.ArgumentParser(description="全量评论情感打标（加载 saved_model 三分类）")
    p.add_argument(
        "--input",
        type=str,
        default=default_json,
        help="输入：final_cleaned_comments.json 或 bert_train_ready.csv（含 txt 或 content 列）",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(current_dir, "saved_model"),
        help="微调模型目录（与 train_roberta 的 export_dir 一致）",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=default_out_dir,
        help="打标结果输出目录（默认 comment/comment_results/）",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="predicted_comments.json",
        help="输出 JSON 文件名（写在 output-dir 下）",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="与训练时 --max_length 对齐；训练默认 512",
    )
    p.add_argument(
        "--log-every-batches",
        type=int,
        default=0,
        metavar="N",
        help="每完成 N 个 batch 在终端打印一行进度与当前类计数（0=关闭，例如 20）",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用断点：从头推理，并删除同输出名对应的 .predict_ckpt.json",
    )
    p.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=50,
        metavar="N",
        help="断点保存：每 N 个 batch 写入一次；0=不在中途写盘（仍会在推理结束及 Ctrl+C 时保存）",
    )
    return p.parse_args()


def _enable_line_buffered_stdout() -> None:
    """尽量行缓冲 stdout，便于在终端/IDE 里实时看到 print 与 tqdm。"""
    if not hasattr(sys.stdout, "reconfigure"):
        return
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (OSError, ValueError):
        pass


def _log(msg: str) -> None:
    print(msg, flush=True)


def main():
    args = parse_args()
    _enable_line_buffered_stdout()
    input_path = os.path.abspath(args.input)
    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(args.output_dir)
    output_file = os.path.join(output_dir, args.output_name)
    ckpt_file = _checkpoint_path(output_file)
    model_config_path = os.path.join(model_dir, "config.json")

    if not os.path.isdir(model_dir):
        _log(f"❌ 找不到模型目录：{model_dir}")
        sys.exit(1)
    if not os.path.isfile(input_path):
        _log(f"❌ 找不到输入文件：{input_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    if args.no_resume and os.path.isfile(ckpt_file):
        try:
            os.remove(ckpt_file)
            _log(f"🗑 已按 --no-resume 删除断点：{ckpt_file}")
        except OSError as e:
            _log(f"⚠️ 无法删除断点文件（将仍尝试覆盖）：{e}")

    _log("📁 读取待预测数据…")
    all_data, texts = load_records(input_path)
    total_len = len(all_data)
    _log(f"✅ 共 {total_len} 条")

    input_stat = _file_fingerprint(input_path)
    cfg_stat = _file_fingerprint(model_config_path)
    if input_stat is None or cfg_stat is None:
        _log("❌ 无法读取输入或模型 config 的文件信息")
        sys.exit(1)

    all_preds: list[int] = []
    if not args.no_resume:
        had_ckpt_file = os.path.isfile(ckpt_file)
        loaded = _load_checkpoint(
            ckpt_file,
            input_path,
            model_dir,
            args.max_length,
            total_len,
            model_config_path,
        )
        if loaded is not None:
            all_preds = loaded
            if len(all_preds) > 0:
                _log(
                    f"⏯ 断点续跑：已恢复 {len(all_preds)}/{total_len} 条（{ckpt_file}）"
                )
        elif had_ckpt_file:
            _log(
                "⚠️ 发现断点文件但与当前输入/模型/max_length 不一致，已忽略，将从头推理。"
            )

    n_done = len(all_preds)
    if n_done > total_len:
        _log("❌ 断点条数异常，请删除检查点后重试")
        sys.exit(1)

    id2label: dict[str, str]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if n_done >= total_len:
        _log("✅ 检查点已覆盖全部样本，跳过 GPU 推理，直接合并导出…")
        cfg = AutoConfig.from_pretrained(model_dir)
        id2label = {str(k): str(v) for k, v in dict(cfg.id2label).items()}
    else:
        _log(f"🧠 加载模型：{model_dir}  |  设备：{device}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        num_labels = getattr(model.config, "num_labels", None)
        if num_labels is None and model.config.id2label:
            num_labels = len(model.config.id2label)
        if num_labels != 3:
            _log(
                f"⚠️ 当前模型 num_labels={num_labels}，本脚本按三分类（-1/0/1）写入 polarity；"
                "若标签空间不同请改脚本或换模型。"
            )

        id2label = {str(k): str(v) for k, v in dict(model.config.id2label).items()}

        texts_tail = texts[n_done:]
        remaining = len(texts_tail)
        dataset = InferenceDataset(texts_tail, tokenizer, max_length=args.max_length)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        class_ctr: Counter[int] = Counter(all_preds)
        log_every = max(0, int(args.log_every_batches))
        ck_every = max(0, int(args.checkpoint_every_batches))
        num_batches = (remaining + args.batch_size - 1) // args.batch_size

        def persist_ckpt() -> None:
            _save_checkpoint_atomic(
                ckpt_file,
                _build_ckpt_payload(
                    input_path=input_path,
                    input_stat=input_stat,
                    model_dir=model_dir,
                    model_config_stat=cfg_stat,
                    max_length=args.max_length,
                    total_len=total_len,
                    preds=all_preds,
                ),
            )

        try:
            with torch.no_grad():
                bar = tqdm(
                    total=total_len,
                    initial=n_done,
                    desc="打标",
                    file=sys.stdout,
                    dynamic_ncols=True,
                    mininterval=0.2,
                    smoothing=0.05,
                    unit="条",
                )
                for batch_idx, batch in enumerate(dataloader):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
                    batch_n = len(preds)
                    for p in preds:
                        ip = int(p)
                        all_preds.append(ip)
                        class_ctr[ip] += 1
                    bar.update(batch_n)

                    if ck_every and (batch_idx + 1) % ck_every == 0:
                        persist_ckpt()

                    if log_every and (batch_idx + 1) % log_every == 0:
                        done = len(all_preds)
                        pct = 100.0 * done / total_len if total_len else 0.0
                        dist = ", ".join(
                            f"class{k}:{class_ctr[k]}"
                            for k in sorted(class_ctr.keys())
                        )
                        _log(
                            f"   … 本段批次 {batch_idx + 1}/{num_batches}  "
                            f"累计已推理 {done}/{total_len} 条 ({pct:.1f}%)  |  {dist}"
                        )
                bar.close()
        except KeyboardInterrupt:
            _log("\n⚠️ 收到中断，正在保存断点…")
            persist_ckpt()
            _log(f"💾 已保存断点 {len(all_preds)}/{total_len} 条 → {ckpt_file}")
            _log("   再次运行同一命令即可续跑。")
            sys.exit(130)

        # 推理已全部在内存中，先落盘全量 preds，再写大 JSON；避免 dump 中途失败导致重来
        persist_ckpt()

        if len(all_preds) != total_len:
            _log("❌ 推理条数与数据不一致，已中止（未写最终 JSON）")
            sys.exit(1)

    class_ctr = Counter(all_preds)

    _log("📦 写入预测字段…")
    for i in range(total_len):
        cid = all_preds[i]
        pol = pred_to_polarity(cid, id2label)
        rec = all_data[i]
        rec["sentiment_class_id"] = cid
        rec["sentiment_polarity"] = pol
        rec["sentiment_text"] = POLARITY_TEXT.get(pol, f"未知({pol})")

    def _json_default(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"不可 JSON 序列化: {type(o)!r}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2, default=_json_default)

    try:
        os.remove(ckpt_file)
    except OSError:
        pass

    dist_final = ", ".join(
        f"class{k}:{class_ctr[k]}" for k in sorted(class_ctr.keys())
    )
    _log("==================================================================")
    _log(f"💾 已保存 {total_len} 条 → {output_file}")
    _log("   字段：sentiment_class_id（0/1/2）、sentiment_polarity（-1/0/1）、sentiment_text")
    if dist_final:
        _log(f"   全量类分布（内部 id）：{dist_final}")
    _log("   （已完成，断点文件已清除）")
    _log("==================================================================")


if __name__ == "__main__":
    main()
