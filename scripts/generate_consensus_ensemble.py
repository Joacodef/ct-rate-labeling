"""Generate consensus label files from multiple prediction CSVs.

This script aligns two or more label CSVs by normalized volume ID and produces
ensemble predictions using three consensus strategies:

- any_positive: label is 1 if at least one run predicts 1
- majority: label is 1 if at least half + 1 runs predict 1
- unanimity: label is 1 only if all runs predict 1

Usage:
    python scripts/generate_consensus_ensemble.py \
        --csvs "outputs/Full Dataset Labeling/full_labels_1/full_labels_gpt-5-nano.csv" \
               "outputs/Full Dataset Labeling/full_labels_2/full_labels_gpt-5-nano.csv" \
               "outputs/Full Dataset Labeling/full_labels_3/full_labels_gpt-5-nano.csv" \
        --names run1 run2 run3 \
        --output-dir "outputs/Evaluations/gpt_consensus_ensemble"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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

    Args:
        value: Raw identifier value from the CSV.
        suffixes: Suffixes that should be removed when matched.

    Returns:
        Normalized string with at most one removed suffix.
    """
    if not isinstance(value, str):
        return value
    for suffix in suffixes:
        if suffix and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _strip_reconstruction_suffix(value: str) -> str:
    """Drop a trailing reconstruction token in the form ``_<number>``.

    Args:
        value: Raw or partially normalized identifier.

    Returns:
        Identifier without trailing reconstruction suffix when present.
    """
    if not isinstance(value, str):
        return value
    head, sep, tail = value.rpartition("_")
    if sep and tail.isdigit():
        return head
    return value


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Build a case-insensitive column map.

    Args:
        df: Input DataFrame.

    Returns:
        Mapping from lowercase column name to original column name.
    """
    return {col.lower(): col for col in df.columns}


def _get_case_insensitive_column(df: pd.DataFrame, target: str) -> str:
    """Resolve one DataFrame column name via case-insensitive lookup.

    Args:
        df: Input DataFrame.
        target: Requested column name.

    Returns:
        Exact column name as stored in the DataFrame.

    Raises:
        KeyError: If no case-insensitive match exists.
    """
    lookup = _normalize_columns(df)
    key = target.lower()
    if key not in lookup:
        raise KeyError(f"Column '{target}' not found (case-insensitive).")
    return lookup[key]


def _resolve_shared_labels(dfs: Sequence[pd.DataFrame], labels_arg: str, exclude_cols_arg: str) -> List[str]:
    """Resolve labels shared across all files.

    Args:
        dfs: Sequence of normalized DataFrames.
        labels_arg: Optional comma-separated labels provided by CLI.
        exclude_cols_arg: Optional comma-separated columns to exclude.

    Returns:
        Ordered list of labels to process.
    """
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

    return [canonical_first[low] for low in sorted(shared_lower) if low not in exclude_set]


def _load_and_normalize(
    csv_path: Path,
    id_col_name: str,
    suffixes: List[str],
    keep_reconstruction_suffix: bool,
) -> pd.DataFrame:
    """Load one CSV and normalize IDs for cross-file alignment.

    Args:
        csv_path: Path to one label CSV.
        id_col_name: Identifier column name to locate.
        suffixes: Suffixes to strip from IDs.
        keep_reconstruction_suffix: Whether to preserve trailing ``_<number>``.

    Returns:
        DataFrame with normalized IDs in column ``VolumeName``.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
    """
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


def _threshold_for_strategy(strategy: str, num_sets: int) -> int:
    """Resolve vote threshold for a consensus strategy.

    Args:
        strategy: One of ``any_positive``, ``majority``, ``unanimity``.
        num_sets: Number of prediction sets being ensembled.

    Returns:
        Minimum number of positive votes required.

    Raises:
        ValueError: If strategy is unknown.
    """
    if strategy == "any_positive":
        return 1
    if strategy == "majority":
        return int(math.floor(num_sets / 2) + 1)
    if strategy == "unanimity":
        return num_sets
    raise ValueError(f"Unknown strategy: {strategy}")


def _build_consensus(
    merged: pd.DataFrame,
    labels: List[str],
    set_names: List[str],
    threshold: int,
) -> pd.DataFrame:
    """Build a consensus prediction table for one threshold.

    Args:
        merged: Aligned table containing all run columns.
        labels: Label names to aggregate.
        set_names: Run names used as column prefixes.
        threshold: Minimum number of positive votes to emit 1.

    Returns:
        DataFrame with ``VolumeName`` and consensus binary labels.
    """
    output = pd.DataFrame({"VolumeName": merged["VolumeName"]})
    for label in labels:
        vote_cols = [f"{name}__{label}" for name in set_names]
        votes = merged[vote_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        output[label] = (votes.sum(axis=1) >= threshold).astype(int)
    return output


def main() -> None:
    """Parse arguments and write ensemble consensus files.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Generate consensus ensemble label files from 2+ prediction CSVs."
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="List of label CSV paths to ensemble.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional display names (same order as --csvs).",
    )
    parser.add_argument(
        "--id-col",
        default="VolumeName",
        help="ID column name in each input CSV (default: VolumeName).",
    )
    parser.add_argument(
        "--strip-suffixes",
        default=".nii.gz,.nii",
        help="Comma-separated suffixes to strip from IDs (default: .nii.gz,.nii).",
    )
    parser.add_argument(
        "--keep-reconstruction-suffix",
        action="store_true",
        help="Keep trailing _<number> in IDs (default: drop).",
    )
    parser.add_argument(
        "--exclude-cols",
        default="VolumeName,report_text",
        help="Comma-separated columns to exclude if --labels is omitted.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated list of label columns to ensemble.",
    )
    parser.add_argument(
        "--strategies",
        default="any_positive,majority,unanimity",
        help=(
            "Comma-separated consensus strategies to export. "
            "Allowed: any_positive,majority,unanimity"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/Evaluations/gpt_consensus_ensemble",
        help="Directory where ensemble files will be written.",
    )

    args = parser.parse_args()

    csv_paths = [Path(path) for path in args.csvs]
    if len(csv_paths) < 2:
        raise SystemExit("Provide at least two CSVs in --csvs.")

    if args.names is None or len(args.names) == 0:
        set_names = [path.stem for path in csv_paths]
    else:
        if len(args.names) != len(csv_paths):
            raise SystemExit("If provided, --names must have exactly one value per CSV in --csvs.")
        set_names = args.names

    suffixes = [suffix.strip() for suffix in args.strip_suffixes.split(",") if suffix.strip()]

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
    for name, df in zip(set_names, normalized_dfs):
        selected_cols = ["VolumeName"] + [_get_case_insensitive_column(df, label) for label in labels]
        subset = df[selected_cols].copy()
        subset = subset.rename(columns={_get_case_insensitive_column(df, label): f"{name}__{label}" for label in labels})
        aligned_tables.append(subset)

    merged = aligned_tables[0]
    for table in aligned_tables[1:]:
        merged = merged.merge(table, on="VolumeName", how="inner")

    if merged.empty:
        raise SystemExit("No matching VolumeName values after normalization.")

    requested_strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if not requested_strategies:
        raise SystemExit("No strategies provided in --strategies.")

    allowed_strategies = {"any_positive", "majority", "unanimity"}
    invalid = [strategy for strategy in requested_strategies if strategy not in allowed_strategies]
    if invalid:
        raise SystemExit(f"Invalid strategies: {invalid}. Allowed: {sorted(allowed_strategies)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: Dict[str, str] = {}
    strategy_meta: Dict[str, Dict[str, int]] = {}
    for strategy in requested_strategies:
        threshold = _threshold_for_strategy(strategy, len(set_names))
        consensus_df = _build_consensus(
            merged=merged,
            labels=labels,
            set_names=set_names,
            threshold=threshold,
        )
        out_path = output_dir / f"consensus_{strategy}.csv"
        consensus_df.to_csv(out_path, index=False)
        output_files[strategy] = str(out_path)
        strategy_meta[strategy] = {
            "threshold": threshold,
            "num_rows": int(len(consensus_df)),
            "num_positive_labels": int(consensus_df[labels].sum().sum()),
        }

    summary = {
        "num_sets": len(set_names),
        "set_names": set_names,
        "input_csvs": [str(path) for path in csv_paths],
        "matched_reports": int(len(merged)),
        "labels_compared": labels,
        "strategies": strategy_meta,
        "output_files": output_files,
    }

    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    print("\n" + "=" * 60)
    print("CONSENSUS ENSEMBLE COMPLETE")
    print("=" * 60)
    print(f"Sets ensembled: {len(set_names)}")
    print(f"Matched reports: {len(merged)}")
    print(f"Labels ensembled: {len(labels)}")
    print("-" * 60)
    for strategy in requested_strategies:
        print(
            f"{strategy:>12} (threshold={strategy_meta[strategy]['threshold']}): "
            f"{output_files[strategy]}"
        )
    print(f"Run summary: {summary_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
