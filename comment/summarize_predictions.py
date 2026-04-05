"""
统计 predict_all.py 输出的带标签评论 JSON（或结构相同的文件）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(here, "comment_results", "predicted_comments.json")
    p = argparse.ArgumentParser(description="统计情感打标结果分布")
    p.add_argument(
        "--input",
        type=str,
        default=default_in,
        help="predict_all 输出的 JSON 路径",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="prediction_stats",
        help=(
            "输出文件名前缀；报告固定写入脚本同级的 comment_results/ "
            "（.txt 与 _polarity.csv、_class.csv、_notes.csv）"
        ),
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="不写 CSV，仅终端与 .txt",
    )
    return p.parse_args()


def load_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入须为 JSON 数组")
    return data


def pct(n: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * n / total


def main() -> None:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "comment_results")
    path_in = os.path.abspath(args.input)
    if not os.path.isfile(path_in):
        print(f"❌ 找不到输入文件：{path_in}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    print(f"📁 读取：{path_in}")
    print(f"📂 报告将写入：{os.path.abspath(out_dir)}")
    rows = load_rows(path_in)
    total = len(rows)
    if total == 0:
        print("无数据。")
        sys.exit(0)

    pol_ctr: Counter[int] = Counter()
    cls_ctr: Counter[int] = Counter()
    text_ctr: Counter[str] = Counter()
    missing_pol = 0
    missing_cls = 0
    note_ctr: Counter[str] = Counter()

    for r in rows:
        nid = r.get("note_id")
        if nid is not None and str(nid).strip() != "":
            note_ctr[str(nid)] += 1

        if "sentiment_polarity" not in r or r["sentiment_polarity"] is None:
            missing_pol += 1
        else:
            try:
                pol_ctr[int(r["sentiment_polarity"])] += 1
            except (TypeError, ValueError):
                missing_pol += 1

        if "sentiment_class_id" not in r or r["sentiment_class_id"] is None:
            missing_cls += 1
        else:
            try:
                cls_ctr[int(r["sentiment_class_id"])] += 1
            except (TypeError, ValueError):
                missing_cls += 1

        st = r.get("sentiment_text")
        if isinstance(st, str) and st.strip():
            text_ctr[st.strip()] += 1

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("情感打标结果统计")
    lines.append("=" * 60)
    lines.append(f"报告目录（固定）：{os.path.abspath(out_dir)}")
    lines.append(f"输入文件：{path_in}")
    lines.append(f"评论总条数：{total:,}")
    lines.append(f"缺 sentiment_polarity：{missing_pol:,}")
    lines.append(f"缺 sentiment_class_id：{missing_cls:,}")
    lines.append("")

    pol_order = [-1, 0, 1]
    lines.append("— 按极性 sentiment_polarity —")
    for p in pol_order:
        c = pol_ctr.get(p, 0)
        lines.append(f"  {p:>3}  {c:>8,}  ({pct(c, total):.2f}%)")
    lines.append("")

    cls_order = [0, 1, 2]
    lines.append("— 按模型类 sentiment_class_id —")
    for k in cls_order:
        c = cls_ctr.get(k, 0)
        lines.append(f"  {k:>3}  {c:>8,}  ({pct(c, total):.2f}%)")
    lines.append("")

    if text_ctr:
        lines.append("— sentiment_text 频次（前 10）—")
        for label, c in text_ctr.most_common(10):
            lines.append(f"  {c:>8,}  {label}")
        lines.append("")

    n_notes = len(note_ctr)
    lines.append("— 按笔记 note_id —")
    lines.append(f"  不同 note_id 数量：{n_notes:,}")
    if n_notes > 0:
        per = list(note_ctr.values())
        lines.append(f"  每条笔记下评论数：最小 {min(per):,}  最大 {max(per):,}")
        lines.append(
            f"  平均 {statistics.mean(per):.2f}  中位数 {statistics.median(per):.2f}"
        )
        if len(per) >= 2:
            try:
                q = statistics.quantiles(per, n=4)
                lines.append(
                    f"  四分位 Q1={q[0]:.1f} Q2={q[1]:.1f} Q3={q[2]:.1f}"
                )
            except statistics.StatisticsError:
                pass
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    txt_path = os.path.join(out_dir, f"{args.prefix}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n💾 文本报告：{txt_path}")

    if args.no_csv:
        return

    pol_csv = os.path.join(out_dir, f"{args.prefix}_polarity.csv")
    with open(pol_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentiment_polarity", "count", "ratio_pct"])
        for p in pol_order:
            c = pol_ctr.get(p, 0)
            w.writerow([p, c, round(pct(c, total), 4)])
    print(f"💾 {pol_csv}")

    cls_csv = os.path.join(out_dir, f"{args.prefix}_class.csv")
    with open(cls_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentiment_class_id", "count", "ratio_pct"])
        for k in cls_order:
            c = cls_ctr.get(k, 0)
            w.writerow([k, c, round(pct(c, total), 4)])
    print(f"💾 {cls_csv}")

    notes_csv = os.path.join(out_dir, f"{args.prefix}_notes.csv")
    with open(notes_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["note_id", "comment_count"])
        for nid, c in sorted(note_ctr.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([nid, c])
    print(f"💾 {notes_csv}（按评论数降序）")


if __name__ == "__main__":
    main()
