"""Summarize discrepancy errors from labeling runs.

Usage:
    python scripts/summarize_discrepancies.py path/to/discrepancies.csv [--top 20]

Prints error counts grouped by label and error type so you can see the most
common mistakes at a glance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

ErrorType = Literal["false_positive", "false_negative", "other"]


def _classify_error(predicted: int, actual: int) -> ErrorType:
    if predicted == 1 and actual == 0:
        return "false_positive"
    if predicted == 0 and actual == 1:
        return "false_negative"
    return "other"


def summarize_discrepancies(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Label", "Predicted", "Actual"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure numeric comparison regardless of CSV typing quirks.
    df["Predicted"] = pd.to_numeric(df["Predicted"], errors="coerce").fillna(0).astype(int)
    df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce").fillna(0).astype(int)
    df["error_type"] = df.apply(lambda row: _classify_error(row["Predicted"], row["Actual"]), axis=1)

    summary = (
        df.groupby(["Label", "error_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "Label"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return summary


def summarize_overall(summary: pd.DataFrame) -> pd.DataFrame:
    overall = (
        summary.groupby("Label", dropna=False)["count"].sum().reset_index(name="total_errors")
    )
    overall = overall.sort_values(["total_errors", "Label"], ascending=[False, True]).reset_index(drop=True)
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize discrepancies for quick error triage.")
    parser.add_argument("csv_path", type=Path, help="Path to discrepancies.csv")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit the number of rows shown in each table (default: show all)",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"File not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    summary = summarize_discrepancies(df)

    if summary.empty:
        print("No discrepancies to summarize.")
        return

    if args.top is not None and args.top > 0:
        display_summary = summary.head(args.top)
        display_overall = summarize_overall(summary).head(args.top)
    else:
        display_summary = summary
        display_overall = summarize_overall(summary)

    print("\nErrors by label + type (most common first):")
    print(display_summary.to_string(index=False))

    print("\nTotal errors by label:")
    print(display_overall.to_string(index=False))


if __name__ == "__main__":
    main()
