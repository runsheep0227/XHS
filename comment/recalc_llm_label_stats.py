"""
根据已人工修正的 llm_labeled_result.csv 重新统计标签分布，
并写入 llm_labeled_result_stats.csv（与 llm_annotate 输出格式一致）。

同时导出「标注失败/异常」行的 note_id，便于在 llm_labeled_result.csv 中定位查看内容与 raw_response：
  - llm_labeled_failure_ids.txt       每行一个 note_id
  - llm_labeled_failures_report.csv   note_id、失败原因、label、content、raw_response

用法（在 comment 目录下）:
    python recalc_llm_label_stats.py

可选环境变量:
    LLM_LABELED_CSV  覆盖输入 CSV 路径（默认 comment/bert_data/llm_labeled_result.csv）
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

BASE_NAME = "llm_labeled_result"

# 与 llm_annotate 一致：2=解析失败，-2=调用/API 失败
FAIL_PARSE = 2
FAIL_API = -2
VALID_EMOTION = {1, 0, -1}
KNOWN_ALL = {-1, 0, 1, 2, -2}


def extract_note_id_for_search(raw: str) -> str:
    """
    小红书 note_id 多为 24 位十六进制。若 CSV 缺逗号导致 note_id 与正文粘在一起，
    仍可从字符串前缀抽出真实 id，便于在 llm_labeled_result.csv 里 Ctrl+F。
    """
    s = str(raw).strip()
    m = re.match(r"^([0-9a-f]{24})", s, re.IGNORECASE)
    return m.group(1) if m else s


def assign_failure_reasons(lab: pd.Series) -> pd.Series:
    """对每行给出失败原因说明；成功行（1/0/-1）为 NaN。"""
    reason = pd.Series(np.nan, index=lab.index, dtype=object)
    reason.loc[lab == FAIL_PARSE] = "parse_fail (label=2，模型返回无法解析为合法 JSON/标签)"
    reason.loc[lab == FAIL_API] = "api_fail (label=-2，请求超时/连接断开等)"
    reason.loc[lab.isna()] = "label_invalid (label 为空或无法转为数字)"
    other = lab.notna() & ~lab.isin(KNOWN_ALL)
    if other.any():
        reason.loc[other] = "unexpected_label (非标准编码): " + lab.loc[other].astype(int).astype(str)
    return reason


def get_label_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """与 llm_annotate 语义一致：有效情感仅为 1/0/-1；占比(%) 以有效条数为分母。"""
    lab = df["label"]
    positive_count = int((lab == 1).sum())
    neutral_count = int((lab == 0).sum())
    negative_count = int((lab == -1).sum())
    total_success = positive_count + neutral_count + negative_count
    label_counts = lab.value_counts().sort_index()

    n = len(df)
    stats_data = [
        {
            "标签编码": 1,
            "标签名称": "正向情感",
            "数量": positive_count,
            "占比(%)": round((positive_count / total_success) * 100, 2) if total_success > 0 else 0,
        },
        {
            "标签编码": 0,
            "标签名称": "中性情感",
            "数量": neutral_count,
            "占比(%)": round((neutral_count / total_success) * 100, 2) if total_success > 0 else 0,
        },
        {
            "标签编码": -1,
            "标签名称": "负向情感",
            "数量": negative_count,
            "占比(%)": round((negative_count / total_success) * 100, 2) if total_success > 0 else 0,
        },
        {
            "标签编码": 2,
            "标签名称": "解析失败",
            "数量": int(label_counts.get(2, 0)),
            "占比(%)": round((label_counts.get(2, 0) / n) * 100, 2) if n > 0 else 0,
        },
        {
            "标签编码": -2,
            "标签名称": "调用失败",
            "数量": int(label_counts.get(-2, 0)),
            "占比(%)": round((label_counts.get(-2, 0) / n) * 100, 2) if n > 0 else 0,
        },
        {
            "标签编码": "总计",
            "标签名称": "有效标注",
            "数量": total_success,
            "占比(%)": round((total_success / n) * 100, 2) if n > 0 else 0,
        },
        {
            "标签编码": "总计",
            "标签名称": "标注失败",
            "数量": int(label_counts.get(-2, 0) + label_counts.get(2, 0)),
            "占比(%)": round(
                ((label_counts.get(-2, 0) + label_counts.get(2, 0)) / n) * 100, 2
            )
            if n > 0
            else 0,
        },
        {
            "标签编码": "总计",
            "标签名称": "数据总数",
            "数量": n,
            "占比(%)": 100.0,
        },
    ]
    return pd.DataFrame(stats_data)


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "bert_data", f"{BASE_NAME}.csv")
    input_csv = os.environ.get("LLM_LABELED_CSV", default_csv)
    out_stats = os.path.join(os.path.dirname(input_csv), f"{BASE_NAME}_stats.csv")

    if not os.path.isfile(input_csv):
        print(f"[错误] 找不到文件：{input_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if "label" not in df.columns:
        print("[错误] CSV 中缺少列 label", file=sys.stderr)
        return 1

    raw = df["label"]
    coerced = pd.to_numeric(raw, errors="coerce")
    bad_parse = coerced.isna() & raw.notna()
    if bad_parse.any():
        print(f"[警告] 有 {int(bad_parse.sum())} 行 label 无法解析为数字，请检查原表。")
    n_empty = int(coerced.isna().sum())
    if n_empty:
        print(f"[警告] 共 {n_empty} 行 label 为空或非数字，不计入各标签数量。")

    df = df.copy()
    df["label"] = coerced

    known = {1, 0, -1, 2, -2}
    valid = df["label"].notna()
    other = df.loc[valid & ~df["label"].isin(known)]
    if len(other) > 0:
        print(
            f"[警告] 存在 {len(other)} 条非标准标签（不在 1,0,-1,2,-2），"
            "「正向/中性/负向」占比仍以标准三类之和为分母。"
        )

    stats_df = get_label_statistics(df)
    stats_df.to_csv(out_stats, index=False, encoding="utf-8-sig")

    out_dir = os.path.dirname(input_csv)
    reasons = assign_failure_reasons(df["label"])
    fail_mask = reasons.notna()
    n_fail = int(fail_mask.sum())

    ids_path = os.path.join(out_dir, "llm_labeled_failure_ids.txt")
    report_path = os.path.join(out_dir, "llm_labeled_failures_report.csv")

    if n_fail > 0:
        fail_df = df.loc[fail_mask].copy()
        fail_df.insert(0, "failure_reason", reasons.loc[fail_mask].values)
        # 便于对照原表：保留转数字后的 label（NaN 保持空）
        fail_df.rename(columns={"label": "label_numeric"}, inplace=True)

        if "note_id" in fail_df.columns:
            note_ids_raw = fail_df["note_id"].astype(str)
        else:
            note_ids_raw = fail_df.index.astype(str)
            fail_df.insert(0, "note_id", note_ids_raw.values)

        note_ids_search = note_ids_raw.map(extract_note_id_for_search)
        fail_df.insert(1, "note_id_search", note_ids_search.values)

        with open(ids_path, "w", encoding="utf-8") as f:
            for nid in note_ids_search:
                f.write(f"{nid.strip()}\n")

        # 报告列顺序：先 id（及可搜索 id）与原因，再全文 content / raw_response
        report_cols = ["note_id", "note_id_search", "failure_reason", "label_numeric"]
        for c in ("content", "raw_response"):
            if c in fail_df.columns:
                report_cols.append(c)
        for c in fail_df.columns:
            if c not in report_cols:
                report_cols.append(c)
        fail_df = fail_df[[c for c in report_cols if c in fail_df.columns]]
        fail_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    else:
        for p in (ids_path, report_path):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    print(f"[完成] 已读取：{input_csv}（{len(df)} 行）")
    print(f"[完成] 统计已写入：{out_stats}")
    print()
    for _, row in stats_df.iterrows():
        print(f"   {row['标签名称']}: {row['数量']} ({row['占比(%)']}%)")

    print()
    if n_fail > 0:
        print(f"[完成] 标注失败/异常共 {n_fail} 条。")
        print(f"        note_id 列表：{ids_path}")
        print(f"        详情（含正文与 raw_response）：{report_path}")
        print("        也可在 llm_labeled_result.csv 中用 Ctrl+F 搜索上述 note_id。")
    else:
        print("[完成] 无标注失败行（label 均为 1/0/-1，且无空/非法标签）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
