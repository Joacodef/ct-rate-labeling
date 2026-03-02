"""Evaluate agreement metrics across two or more label CSVs.

Usage:
    python scripts/evaluate_label_agreement.py \
        --csvs "outputs/Full Dataset Labeling/full_labels_1/full_labels_gpt-5-nano.csv" \
               "outputs/Full Dataset Labeling/full_labels_2/full_labels_gpt-5-nano.csv" \
               "outputs/Full Dataset Labeling/full_labels_3/full_labels_gpt-5-nano.csv" \
        --names round1 round2 round3 \
        --output-dir "outputs/Comparisons/agreement_gpt5nano_rounds"
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    """Remove the first matching suffix from a string value.

    Non-string inputs are returned unchanged.
    """
    if not isinstance(value, str):
        return value
    for suffix in suffixes:
        if suffix and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _strip_reconstruction_suffix(value: str) -> str:
    """Drop trailing reconstruction token in the form '_<number>' from an ID."""
    if not isinstance(value, str):
        return value
    head, sep, tail = value.rpartition("_")
    if sep and tail.isdigit():
        return head
    return value


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Build a case-insensitive lookup mapping lowercase name -> original name."""
    return {col.lower(): col for col in df.columns}


def _get_case_insensitive_column(df: pd.DataFrame, target: str) -> str:
    """Resolve a column name from a DataFrame using case-insensitive matching."""
    lookup = _normalize_columns(df)
    key = target.lower()
    if key not in lookup:
        raise KeyError(f"Column '{target}' not found (case-insensitive).")
    return lookup[key]


def _resolve_shared_labels(dfs: Sequence[pd.DataFrame], labels_arg: str, exclude_cols_arg: str) -> List[str]:
    """Resolve label columns common to all inputs, or use the explicit --labels list."""
    if labels_arg.strip():
        return [label.strip() for label in labels_arg.split(",") if label.strip()]

    exclude_set = {name.strip().lower() for name in exclude_cols_arg.split(",") if name.strip()}
    exclude_set.update({name.lower() for name in META_COLUMNS})

    shared_lower = None
    canonical_first: Dict[str, str] = {}
    for index, df in enumerate(dfs):
        cols = {col.lower(): col for col in df.columns}
        if index == 0:
            canonical_first = cols
            shared_lower = set(cols.keys())
        else:
            shared_lower &= set(cols.keys())

    if not shared_lower:
        return []

    labels = [canonical_first[low] for low in sorted(shared_lower) if low not in exclude_set]
    return labels


def _calculate_agreement_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Compute agreement, Cohen's kappa, and MCC from confusion-count totals."""
    n = tp + fp + fn + tn
    if n == 0:
        return {
            "agreement": 0.0,
            "cohen_kappa": 0.0,
            "mcc": 0.0,
            "n": 0,
        }

    agreement = (tp + tn) / n

    p_a_pos = (tp + fn) / n
    p_a_neg = (tn + fp) / n
    p_b_pos = (tp + fp) / n
    p_b_neg = (tn + fn) / n
    p_expected = (p_a_pos * p_b_pos) + (p_a_neg * p_b_neg)

    if (1 - p_expected) == 0:
        cohen_kappa = 1.0 if agreement == 1.0 else 0.0
    else:
        cohen_kappa = (agreement - p_expected) / (1 - p_expected)

    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den > 0 else 0.0

    return {
        "agreement": round(float(agreement), 4),
        "cohen_kappa": round(float(cohen_kappa), 4),
        "mcc": round(float(mcc), 4),
        "n": int(n),
    }


def _confusion_counts(a: pd.Series, b: pd.Series) -> Dict[str, int]:
    """Build binary confusion counts for two label series after numeric coercion."""
    a_bin = pd.to_numeric(a, errors="coerce").fillna(0).astype(int)
    b_bin = pd.to_numeric(b, errors="coerce").fillna(0).astype(int)

    tp = int(((a_bin == 1) & (b_bin == 1)).sum())
    fp = int(((a_bin == 0) & (b_bin == 1)).sum())
    fn = int(((a_bin == 1) & (b_bin == 0)).sum())
    tn = int(((a_bin == 0) & (b_bin == 0)).sum())

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _load_and_normalize(
    csv_path: Path,
    id_col_name: str,
    suffixes: List[str],
    keep_reconstruction_suffix: bool,
) -> pd.DataFrame:
    """Load one CSV, normalize IDs, drop duplicate IDs, and standardize ID column name."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    id_col = _get_case_insensitive_column(df, id_col_name)

    if suffixes:
        df[id_col] = df[id_col].astype(str).map(lambda v: _strip_suffixes(v, suffixes))

    if not keep_reconstruction_suffix:
        df[id_col] = df[id_col].astype(str).map(_strip_reconstruction_suffix)

    df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    df = df.rename(columns={id_col: "VolumeName"})
    return df


def main() -> None:
    """Parse arguments, align all CSVs, compute agreement outputs, and write reports."""
    parser = argparse.ArgumentParser(
        description="Compute pairwise agreement metrics across 2+ label CSVs (no ground-truth assumption)."
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="List of CSV paths to compare.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional display names for each CSV (same order as --csvs).",
    )
    parser.add_argument(
        "--id-col",
        default="VolumeName",
        help="ID column name present in each CSV (default: VolumeName).",
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
        help="Comma-separated columns to exclude from labels when --labels is omitted.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated list of label columns to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/Comparisons/label_agreement",
        help="Directory to write outputs.",
    )

    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csvs]
    if len(csv_paths) < 2:
        raise SystemExit("Provide at least two CSVs in --csvs.")

    if args.names is None or len(args.names) == 0:
        csv_names = [path.stem for path in csv_paths]
    else:
        if len(args.names) != len(csv_paths):
            raise SystemExit("If provided, --names must have exactly one value per CSV in --csvs.")
        csv_names = args.names

    suffixes = [s.strip() for s in args.strip_suffixes.split(",") if s.strip()]

    normalized_dfs = [
        _load_and_normalize(
            csv_path=csv_path,
            id_col_name=args.id_col,
            suffixes=suffixes,
            keep_reconstruction_suffix=args.keep_reconstruction_suffix,
        )
        for csv_path in csv_paths
    ]

    labels = _resolve_shared_labels(normalized_dfs, labels_arg=args.labels, exclude_cols_arg=args.exclude_cols)
    if not labels:
        raise SystemExit("No shared label columns found across all input CSVs.")

    aligned_tables = []
    for name, df in zip(csv_names, normalized_dfs):
        selected_cols = ["VolumeName"] + [_get_case_insensitive_column(df, label) for label in labels]
        subset = df[selected_cols].copy()
        subset = subset.rename(columns={_get_case_insensitive_column(df, label): f"{name}__{label}" for label in labels})
        aligned_tables.append(subset)

    merged = aligned_tables[0]
    for table in aligned_tables[1:]:
        merged = merged.merge(table, on="VolumeName", how="inner")

    if merged.empty:
        raise SystemExit("No matching VolumeName values after normalization.")

    pair_rows: List[Dict[str, object]] = []
    pair_macro_rows: List[Dict[str, object]] = []

    pairs = list(itertools.combinations(csv_names, 2))
    for name_a, name_b in pairs:
        per_label_rows = []
        for label in labels:
            col_a = f"{name_a}__{label}"
            col_b = f"{name_b}__{label}"
            counts = _confusion_counts(merged[col_a], merged[col_b])
            agree = _calculate_agreement_metrics(**counts)

            row = {
                "pair": f"{name_a} vs {name_b}",
                "label": label,
                **agree,
                **counts,
            }
            pair_rows.append(row)
            per_label_rows.append(row)

        pair_df = pd.DataFrame(per_label_rows)
        if not pair_df.empty:
            pair_macro_rows.append(
                {
                    "pair": f"{name_a} vs {name_b}",
                    "label": "MACRO_AVERAGE",
                    "agreement": round(float(pair_df["agreement"].mean()), 4),
                    "cohen_kappa": round(float(pair_df["cohen_kappa"].mean()), 4),
                    "mcc": round(float(pair_df["mcc"].mean()), 4),
                    "n": int(pair_df["n"].iloc[0]) if len(pair_df) else 0,
                    "tp": int(pair_df["tp"].sum()),
                    "fp": int(pair_df["fp"].sum()),
                    "fn": int(pair_df["fn"].sum()),
                    "tn": int(pair_df["tn"].sum()),
                }
            )

    pairwise_df = pd.DataFrame(pair_rows)
    pairwise_summary_df = pd.DataFrame(pair_macro_rows)

    all_sets_rows: List[Dict[str, object]] = []
    for label in labels:
        label_cols = [f"{name}__{label}" for name in csv_names]
        values = merged[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        all_equal = values.nunique(axis=1) == 1
        positive_consensus = (values.sum(axis=1) == len(csv_names)).sum()
        negative_consensus = (values.sum(axis=1) == 0).sum()

        all_sets_rows.append(
            {
                "label": label,
                "all_sets_agreement": round(float(all_equal.mean()), 4),
                "all_positive_consensus": int(positive_consensus),
                "all_negative_consensus": int(negative_consensus),
                "n": int(len(values)),
            }
        )

    all_sets_df = pd.DataFrame(all_sets_rows)
    overall_all_sets_agreement = round(float(all_sets_df["all_sets_agreement"].mean()), 4) if not all_sets_df.empty else 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_path = output_dir / "pairwise_agreement.csv"
    pairwise_summary_path = output_dir / "pairwise_agreement_summary.csv"
    all_sets_path = output_dir / "all_sets_agreement.csv"
    summary_path = output_dir / "run_summary.json"

    pairwise_df.to_csv(pairwise_path, index=False)
    pairwise_summary_df.to_csv(pairwise_summary_path, index=False)
    all_sets_df.to_csv(all_sets_path, index=False)

    summary = {
        "num_sets": len(csv_names),
        "set_names": csv_names,
        "total_reports_matched": int(len(merged)),
        "labels_compared": labels,
        "num_pairs": len(pairs),
        "overall_all_sets_agreement": overall_all_sets_agreement,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("MULTI-SET LABEL AGREEMENT COMPLETE")
    print("=" * 60)
    print(f"Sets compared: {len(csv_names)}")
    print(f"Matched reports: {len(merged)}")
    print(f"Labels compared: {len(labels)}")
    print(f"Pairwise comparisons: {len(pairs)}")
    print("-" * 60)
    print(f"Pairwise agreement:        {pairwise_path}")
    print(f"Pairwise summary:          {pairwise_summary_path}")
    print(f"All-sets agreement:        {all_sets_path}")
    print(f"Run summary:               {summary_path}")
    print("-" * 60)

    if not pairwise_summary_df.empty:
        print("Pairwise macro summary:")
        print(pairwise_summary_df[["pair", "agreement", "cohen_kappa", "mcc"]].to_string(index=False))
        print("-" * 60)

    if not all_sets_df.empty:
        print("All-sets agreement (per label):")
        print(all_sets_df[["label", "all_sets_agreement", "n"]].to_string(index=False))

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
