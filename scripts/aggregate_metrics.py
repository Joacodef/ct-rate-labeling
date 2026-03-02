import argparse
import glob
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def load_metrics(files: List[str]) -> pd.DataFrame:
    if not files:
        raise SystemExit("No files matched the provided pattern.")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["_source"] = path
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _collect_input_files(glob_pattern: str | None, files: Iterable[str]) -> List[str]:
    collected = list(files)
    if glob_pattern:
        collected.extend(glob.glob(glob_pattern))
    unique_files = sorted(set(collected))
    if not unique_files:
        raise SystemExit("Provide at least one input file via --files and/or --glob.")
    return unique_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate metric CSV files and compute per-label mean/std.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help="Optional glob pattern for metric CSV files.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Optional explicit list of metric CSV files.",
    )
    parser.add_argument(
        "--output",
        default="aggregated_metrics_mean_std.csv",
        help="Output CSV path (default: aggregated_metrics_mean_std.csv).",
    )

    args = parser.parse_args()

    files = _collect_input_files(args.glob_pattern, args.files)
    metrics = load_metrics(files)

    numeric_cols = ["precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"]
    missing = [c for c in numeric_cols + ["label"] if c not in metrics.columns]
    if missing:
        raise SystemExit(f"Missing columns in input metrics: {missing}")

    grouped = metrics.groupby("label", as_index=False)[numeric_cols]
    means = grouped.mean().rename(columns={c: f"{c}_mean" for c in numeric_cols})
    stds = grouped.std(ddof=0).rename(columns={c: f"{c}_std" for c in numeric_cols})

    result = pd.merge(means, stds, on="label", how="left")
    result = result[[
        "label",
        *[f"{c}_mean" for c in numeric_cols],
        *[f"{c}_std" for c in numeric_cols],
    ]]

    float_cols = [c for c in result.columns if c != "label"]
    result[float_cols] = result[float_cols].round(4)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Wrote {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
