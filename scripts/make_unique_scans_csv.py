import argparse
import logging
import os
import re
from typing import List

import pandas as pd

log = logging.getLogger(__name__)

RECON_SUFFIX_PATTERN = re.compile(r"^(.*)_\d+\.nii\.gz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a de-duplicated CT-RATE CSV with one row per unique scan."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original CT-RATE CSV (with Findings/Impressions columns)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the de-duplicated CSV (VolumeName + report_text)."
    )
    parser.add_argument(
        "--report-columns",
        default="Findings_EN,Impressions_EN",
        help=(
            "Comma-separated columns to combine into report_text, "
            "in order. Default: Findings_EN,Impressions_EN."
        )
    )
    parser.add_argument(
        "--keep-columns",
        default="",
        help=(
            "Optional comma-separated columns to carry through to the output. "
            "By default only VolumeName and report_text are written."
        )
    )
    return parser.parse_args()


def extract_scan_id(volume_name: str) -> str:
    """Strip reconstruction suffix like _2.nii.gz -> base scan id (train_1_a)."""
    if not isinstance(volume_name, str):
        return ""
    match = RECON_SUFFIX_PATTERN.match(volume_name)
    if match:
        return match.group(1)
    return volume_name


def build_report_text(row: pd.Series, report_cols: List[str]) -> str:
    parts = []
    for col in report_cols:
        value = row.get(col, "")
        if pd.isna(value):
            value = ""
        value = str(value).strip()
        if value:
            parts.append(value)
    return "\n\n".join(parts).strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    report_cols = [c.strip() for c in args.report_columns.split(",") if c.strip()]
    if not report_cols:
        raise ValueError("--report-columns must include at least one column name.")

    keep_cols = [c.strip() for c in args.keep_columns.split(",") if c.strip()]

    df = pd.read_csv(args.input)
    if "VolumeName" not in df.columns:
        raise ValueError("Input CSV must include VolumeName column.")

    missing = [c for c in report_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing report columns: {missing}")

    df = df.copy()
    df["_scan_id"] = df["VolumeName"].map(extract_scan_id)
    df["report_text"] = df.apply(lambda r: build_report_text(r, report_cols), axis=1)
    df["_report_len"] = df["report_text"].str.len().fillna(0).astype(int)

    # Prefer the longest report_text per scan id.
    df_sorted = df.sort_values(by=["_scan_id", "_report_len"], ascending=[True, False])
    deduped = df_sorted.drop_duplicates(subset=["_scan_id"], keep="first")

    output_cols = ["VolumeName", "report_text"]
    for col in keep_cols:
        if col in deduped.columns and col not in output_cols:
            output_cols.append(col)

    deduped_out = deduped.copy()
    deduped_out["VolumeName"] = deduped_out["_scan_id"]

    deduped_out.to_csv(args.output, index=False, columns=output_cols)

    log.info("Input rows: %d", len(df))
    log.info("Unique scans: %d", len(deduped_out))
    log.info("Wrote: %s", args.output)


if __name__ == "__main__":
    main()