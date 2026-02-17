"""Compare labels between two CSVs for shared columns.

Usage:
    python scripts/compare_label_csvs.py \
        --left-csv data/MANUAL_LABELS.csv \
        --right-csv "outputs/Full Dataset Labeling/manual_set_labels_gpt-5-nano.csv" \
        --output-dir "outputs/Full Dataset Labeling/compare_label_csvs"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

META_COLUMNS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_seconds",
    "estimated_cost_usd",
    "model_version",
    "request_id",
    "retry_count",
    "started_at_utc",
    "ended_at_utc",
    "report_hash",
    "status",
    "error_message",
    "report_text",
}


def _strip_suffixes(value: str, suffixes: Iterable[str]) -> str:
    if not isinstance(value, str):
        return value
    for suffix in suffixes:
        if suffix and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _strip_reconstruction_suffix(value: str) -> str:
    if not isinstance(value, str):
        return value
    head, sep, tail = value.rpartition("_")
    if sep and tail.isdigit():
        return head
    return value


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {col.lower(): col for col in df.columns}


def _get_case_insensitive_column(df: pd.DataFrame, target: str) -> str:
    lookup = _normalize_columns(df)
    key = target.lower()
    if key not in lookup:
        raise KeyError(f"Column '{target}' not found (case-insensitive).")
    return lookup[key]


def _shared_label_columns(left_df: pd.DataFrame, right_df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    left_cols = {c.lower(): c for c in left_df.columns}
    right_cols = {c.lower(): c for c in right_df.columns}
    exclude_set = {name.lower() for name in exclude}
    shared = [
        left_cols[name]
        for name in left_cols.keys() & right_cols.keys()
        if name not in exclude_set
        and name not in {c.lower() for c in META_COLUMNS}
    ]
    return sorted(shared)


def calculate_binary_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> Dict[str, float]:
    y_true = pd.to_numeric(df[true_col], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    if (tp + fn) == 0:
        if fp == 0:
            precision = recall = f1 = 1.0
        else:
            precision = recall = f1 = 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _build_discrepancies(
    df: pd.DataFrame,
    volume_col: str,
    labels: Iterable[Tuple[str, str, str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, true_col, pred_col in labels:
        y_true = pd.to_numeric(df[true_col], errors="coerce").fillna(0).astype(int)
        y_pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int)
        mismatch = y_true != y_pred
        if not mismatch.any():
            continue
        subset = df.loc[mismatch, [volume_col]].copy()
        subset["Label"] = label
        subset["Predicted"] = y_pred[mismatch].values
        subset["Actual"] = y_true[mismatch].values
        rows.extend(subset.to_dict(orient="records"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare label columns between two CSVs.")
    parser.add_argument("--left-csv", required=True, help="Path to left CSV")
    parser.add_argument("--right-csv", required=True, help="Path to right CSV")
    parser.add_argument(
        "--left-name",
        default="left",
        help="Label to use for left CSV in summaries (default: left).",
    )
    parser.add_argument(
        "--right-name",
        default="right",
        help="Label to use for right CSV in summaries (default: right).",
    )
    parser.add_argument(
        "--left-id-col",
        default="VolumeName",
        help="ID column name in left CSV (default: VolumeName).",
    )
    parser.add_argument(
        "--right-id-col",
        default="VolumeName",
        help="ID column name in right CSV (default: VolumeName).",
    )
    parser.add_argument(
        "--strip-suffixes",
        default=".nii.gz,.nii",
        help="Comma-separated suffixes to strip from ID values (default: .nii.gz,.nii).",
    )
    parser.add_argument(
        "--keep-reconstruction-suffix",
        action="store_true",
        help="Keep trailing _<number> in ID values (default: drop).",
    )
    parser.add_argument(
        "--exclude-cols",
        default="VolumeName,report_text",
        help="Comma-separated column names to exclude from label comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/Full Dataset Labeling/compare_label_csvs",
        help="Directory to write metrics and discrepancies.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated list of label columns to compare.",
    )

    args = parser.parse_args()

    left_path = Path(args.left_csv)
    right_path = Path(args.right_csv)
    output_dir = Path(args.output_dir)

    if not left_path.exists():
        raise FileNotFoundError(f"Left CSV not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right CSV not found: {right_path}")

    left_df = pd.read_csv(left_path)
    right_df = pd.read_csv(right_path)

    left_id_col = _get_case_insensitive_column(left_df, args.left_id_col)
    right_id_col = _get_case_insensitive_column(right_df, args.right_id_col)

    suffixes = [s.strip() for s in args.strip_suffixes.split(",") if s.strip()]
    if suffixes:
        left_df[left_id_col] = left_df[left_id_col].astype(str).map(lambda v: _strip_suffixes(v, suffixes))
        right_df[right_id_col] = right_df[right_id_col].astype(str).map(lambda v: _strip_suffixes(v, suffixes))

    if not args.keep_reconstruction_suffix:
        left_df[left_id_col] = left_df[left_id_col].astype(str).map(_strip_reconstruction_suffix)
        right_df[right_id_col] = right_df[right_id_col].astype(str).map(_strip_reconstruction_suffix)

        left_df = left_df.drop_duplicates(subset=[left_id_col], keep="first").reset_index(drop=True)
        right_df = right_df.drop_duplicates(subset=[right_id_col], keep="first").reset_index(drop=True)

    if args.labels.strip():
        label_list = [label.strip() for label in args.labels.split(",") if label.strip()]
    else:
        exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
        label_list = _shared_label_columns(left_df, right_df, exclude_cols)

    if not label_list:
        raise SystemExit("No shared label columns found between the two CSVs.")

    label_pairs: List[Tuple[str, str, str]] = []
    for label in label_list:
        left_col = _get_case_insensitive_column(left_df, label)
        right_col = _get_case_insensitive_column(right_df, label)
        label_pairs.append((label, left_col, right_col))

    left_subset = left_df[[left_id_col] + [m for _, m, _ in label_pairs]].copy()
    left_subset = left_subset.rename(
        columns={left_id_col: "VolumeName", **{m: f"true_{label}" for label, m, _ in label_pairs}}
    )

    right_subset = right_df[[right_id_col] + [g for _, _, g in label_pairs]].copy()
    right_subset = right_subset.rename(
        columns={right_id_col: "VolumeName", **{g: f"pred_{label}" for label, _, g in label_pairs}}
    )

    merged = left_subset.merge(
        right_subset,
        on="VolumeName",
        how="inner",
    )

    left_only = set(left_subset["VolumeName"]) - set(merged["VolumeName"])
    right_only = set(right_subset["VolumeName"]) - set(merged["VolumeName"])

    if merged.empty:
        raise SystemExit("No matching VolumeName values after normalization.")

    metric_rows = []
    discrepancies = []

    total_correct = 0
    total_labels = 0

    for label, _, _ in label_pairs:
        true_col = f"true_{label}"
        pred_col = f"pred_{label}"

        stats = calculate_binary_metrics(merged, true_col, pred_col)
        stats["label"] = label
        metric_rows.append(stats)

        y_true = pd.to_numeric(merged[true_col], errors="coerce").fillna(0).astype(int)
        y_pred = pd.to_numeric(merged[pred_col], errors="coerce").fillna(0).astype(int)
        total_correct += int((y_true == y_pred).sum())
        total_labels += int(len(merged))

    discrepancies = _build_discrepancies(
        merged,
        "VolumeName",
        [(label, f"true_{label}", f"pred_{label}") for label, _, _ in label_pairs],
    )

    metrics_df = pd.DataFrame(metric_rows)

    if not metrics_df.empty:
        macro_avg = {
            "label": "MACRO_AVERAGE",
            "precision": round(metrics_df["precision"].mean(), 4),
            "recall": round(metrics_df["recall"].mean(), 4),
            "f1": round(metrics_df["f1"].mean(), 4),
            "accuracy": round(metrics_df["accuracy"].mean(), 4),
            "tp": int(metrics_df["tp"].sum()),
            "fp": int(metrics_df["fp"].sum()),
            "fn": int(metrics_df["fn"].sum()),
            "tn": int(metrics_df["tn"].sum()),
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([macro_avg])], ignore_index=True)

        micro_tp = int(metrics_df.loc[metrics_df["label"] != "MACRO_AVERAGE", "tp"].sum())
        micro_fp = int(metrics_df.loc[metrics_df["label"] != "MACRO_AVERAGE", "fp"].sum())
        micro_fn = int(metrics_df.loc[metrics_df["label"] != "MACRO_AVERAGE", "fn"].sum())
        micro_tn = int(metrics_df.loc[metrics_df["label"] != "MACRO_AVERAGE", "tn"].sum())
        micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = (
            2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )
        micro_accuracy = (
            (micro_tp + micro_tn) / (micro_tp + micro_tn + micro_fp + micro_fn)
            if (micro_tp + micro_tn + micro_fp + micro_fn) > 0
            else 0.0
        )

        micro_row = {
            "label": "MICRO_AVERAGE",
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
            "accuracy": round(micro_accuracy, 4),
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "tn": micro_tn,
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([micro_row])], ignore_index=True)

    exact_match = 0
    if label_pairs:
        label_cols = [f"true_{label}" for label, _, _ in label_pairs]
        pred_cols = [f"pred_{label}" for label, _, _ in label_pairs]
        exact_match = int((merged[label_cols].values == merged[pred_cols].values).all(axis=1).sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.csv"
    discrepancies_path = output_dir / "discrepancies.csv"
    summary_path = output_dir / "run_summary.json"

    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(discrepancies).to_csv(discrepancies_path, index=False)

    summary = {
        "total_reports": int(len(merged)),
        "matched_reports": int(len(merged)),
        f"{args.left_name}_only": int(len(left_only)),
        f"{args.right_name}_only": int(len(right_only)),
        "labels_compared": label_list,
        "total_labels": int(total_labels),
        "total_correct_labels": int(total_correct),
        "label_accuracy": round(total_correct / total_labels, 4) if total_labels else 0.0,
        "exact_match_count": int(exact_match),
        "exact_match_rate": round(exact_match / len(merged), 4) if len(merged) else 0.0,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("LABEL COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Matched reports: {len(merged)}")
    print(f"{args.left_name}-only reports: {len(left_only)}")
    print(f"{args.right_name}-only reports: {len(right_only)}")
    print(f"Total labels compared: {total_labels}")
    print(f"Total correct labels: {total_correct}")
    if total_labels:
        print(f"Label-wise accuracy: {summary['label_accuracy']:.4f}")
    if len(merged):
        print(f"Exact match rate: {summary['exact_match_rate']:.4f}")
    print("-" * 60)
    print(f"Metrics saved to:       {metrics_path}")
    print(f"Discrepancies saved to: {discrepancies_path}")
    print(f"Run summary saved to:   {summary_path}")
    print("-" * 60)

    if not metrics_df.empty:
        display_cols = ["label", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn"]
        print(metrics_df[display_cols].to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
