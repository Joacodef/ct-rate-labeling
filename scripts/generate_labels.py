import argparse
import hashlib
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig

# Load environment variables from .env file before Hydra config is parsed
load_dotenv()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="Path to a previous Hydra run directory whose config/output should be reused"
)
parsed_args, remaining = parser.parse_known_args()
RESUME_RUN_DIR = parsed_args.resume or ""
if RESUME_RUN_DIR:
    resolved_resume_dir = os.path.abspath(RESUME_RUN_DIR)
    hydra_dir_override = f"hydra.run.dir={resolved_resume_dir}"
    sys.argv = [sys.argv[0], hydra_dir_override] + remaining
    os.environ["HYDRA_RUN_DIR"] = resolved_resume_dir
    RESUME_RUN_DIR = resolved_resume_dir
else:
    sys.argv = [sys.argv[0]] + remaining
# If running locally without installation, fallback to adding src to path.
try:
    from ctr_labeling import LLMClient
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    from ctr_labeling import LLMClient

log = logging.getLogger(__name__)
UNPRICED_MODELS_WARNED = set()


def _safe_number(value, default=0.0):
    """Convert a value to float, guarding against pandas NA objects."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_resume_data(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Load an existing labels CSV to support resuming a previous run."""
    if not path:
        return {}, {}
    if not os.path.exists(path):
        log.warning("resume_from_csv path '%s' not found; starting fresh run.", path)
        return {}, {}

    try:
        resume_df = pd.read_csv(path)
    except Exception as exc:
        log.error("Failed to load resume CSV '%s': %s", path, exc)
        return {}, {}

    if "VolumeName" not in resume_df.columns:
        log.warning("Resume CSV '%s' lacks VolumeName column; ignoring resume request.", path)
        return {}, {}

    resume_df = resume_df.convert_dtypes()
    by_volume = {}
    by_hash = defaultdict(list)
    for _, row in resume_df.iterrows():
        record = row.to_dict()
        vol = record.get("VolumeName")
        if isinstance(vol, str) and vol:
            by_volume[vol] = record
        rhash = record.get("report_hash")
        if isinstance(rhash, str) and rhash:
            by_hash[rhash].append(record)

    log.info(
        "Loaded %d previously labeled rows from %s for resume support.",
        len(resume_df),
        path
    )
    return by_volume, by_hash


def remove_hash_record(
    hash_map: Dict[str, List[Dict[str, Any]]],
    rhash: str,
    record: Dict[str, Any]
) -> None:
    """Remove a specific record from the hash bucket if present."""
    if not rhash:
        return
    bucket = hash_map.get(rhash)
    if not bucket:
        return
    for idx, candidate in enumerate(bucket):
        if candidate is record:
            bucket.pop(idx)
            if not bucket:
                hash_map.pop(rhash, None)
            break

def estimate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_table: dict,
    precision: int = 8
) -> float:
    """Estimate USD cost using pricing data supplied via config."""
    if not pricing_table:
        if model_name not in UNPRICED_MODELS_WARNED:
            log.warning(
                "No pricing table configured. Estimated cost for model '%s' will be 0.",
                model_name
            )
            UNPRICED_MODELS_WARNED.add(model_name)
        return 0.0

    model_base = model_name.lower()
    sorted_models = sorted(
        ((key.lower(), value) for key, value in pricing_table.items()),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for match_key, price_info in sorted_models:
        if match_key in model_base:
            input_price = price_info.get("input_per_million")
            output_price = price_info.get("output_per_million")
            if input_price is None or output_price is None:
                log.warning(
                    "Pricing entry '%s' lacks input/output rates; reporting 0 cost.",
                    match_key
                )
                return 0.0

            cost = (prompt_tokens / 1_000_000 * input_price) + \
                   (completion_tokens / 1_000_000 * output_price)
            return round(cost, precision)

    if model_base not in UNPRICED_MODELS_WARNED:
        log.warning(
            "No pricing data matched model '%s'. Estimated cost will be reported as 0.",
            model_name
        )
        UNPRICED_MODELS_WARNED.add(model_base)
    return 0.0

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for generating binary classification labels from radiology reports.
    """
    resume_csv_path = ""
    if RESUME_RUN_DIR:
        overrides = [ov for ov in HydraConfig.get().overrides.task if not ov.startswith("hydra.run.dir=")]
        if overrides:
            log.error(
                "--resume was provided but additional Hydra overrides were detected (%s). "
                "Please rerun without extra overrides so the original config can be reused.",
                overrides
            )
            sys.exit(1)

        resume_dir = Path(RESUME_RUN_DIR)
        resume_cfg_path = resume_dir / ".hydra" / "config.yaml"
        if not resume_cfg_path.exists():
            log.error("Resume directory %s does not contain .hydra/config.yaml", resume_dir)
            sys.exit(1)

        cfg = OmegaConf.load(str(resume_cfg_path))
        resume_csv_path = str(resume_dir / cfg.io.output_csv)
        log.info(
            "Resuming with config from %s; resume CSV set to %s",
            resume_cfg_path,
            resume_csv_path
        )
        log.info(f"Loaded resume configuration:\n{OmegaConf.to_yaml(cfg)}")
    else:
        log.info(f"Starting label generation with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Validation of Input Config
    if not cfg.io.reports_csv:
        log.error("Input CSV path (io.reports_csv) is not defined.")
        sys.exit(1)

    input_path = cfg.io.reports_csv
    if not os.path.exists(input_path):
        log.error(f"Input file not found at: {input_path}")
        sys.exit(1)

    # 2. Load Data
    log.info(f"Loading reports from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        log.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    required_columns = ["VolumeName", "report_text"]
    if not all(col in df.columns for col in required_columns):
        log.error(f"Input CSV must contain columns: {required_columns}. Found: {df.columns.tolist()}")
        sys.exit(1)

    target_labels = list(cfg.prompt.labels)

    prompt_mode = str(cfg.prompt.get("mode", "multi")).lower()
    if prompt_mode not in {"multi", "single"}:
        log.error("prompt.mode must be either 'multi' or 'single'. Got '%s'", prompt_mode)
        sys.exit(1)

    cols = [
        "VolumeName",
        *target_labels,
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
        "error_message"
    ]

    int_cols = target_labels + [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "retry_count"
    ]
    float_cols = ["latency_seconds", "estimated_cost_usd"]

    if RESUME_RUN_DIR:
        output_dir = RESUME_RUN_DIR
    else:
        output_dir = HydraConfig.get().runtime.output_dir

    output_rel = cfg.io.output_csv
    if os.path.isabs(output_rel):
        output_path = output_rel
    else:
        output_path = os.path.join(output_dir, output_rel)

    output_parent = os.path.dirname(output_path)
    if output_parent and not os.path.exists(output_parent):
        os.makedirs(output_parent, exist_ok=True)

    if RESUME_RUN_DIR:
        resume_csv_path = output_path

    cost_precision = cfg.api.cost_precision if "cost_precision" in cfg.api else 8
    pricing_cfg = cfg.api.get("pricing", {})
    pricing_table = OmegaConf.to_container(pricing_cfg, resolve=True)
    pricing_table = pricing_table or {}

    output_initialized = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    def normalize_row(raw: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for col in cols:
            value = raw.get(col)
            if col in int_cols:
                normalized[col] = int(_safe_number(value, 0))
            elif col in float_cols:
                normalized[col] = float(_safe_number(value, 0.0))
            else:
                if value is None:
                    normalized[col] = ""
                    continue
                try:
                    if pd.isna(value):
                        normalized[col] = ""
                        continue
                except Exception:
                    pass
                normalized[col] = value if isinstance(value, str) else str(value)
        return normalized

    def append_row(row: Dict[str, Any]) -> None:
        nonlocal output_initialized
        df_row = pd.DataFrame([row], columns=cols)
        df_row.to_csv(
            output_path,
            mode="a",
            header=not output_initialized,
            index=False
        )
        output_initialized = True

    def combine_meta(meta_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        combined = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "status": "success",
            "error_message": "",
            "request_id": "",
            "model_version": "",
            "retry_count": 0,
            "started_at_utc": "",
            "ended_at_utc": ""
        }

        if not meta_batch:
            combined["status"] = "skipped"
            return combined

        request_ids: List[str] = []
        model_versions: List[str] = []
        statuses: List[str] = []
        error_messages: List[str] = []
        start_times: List[str] = []
        end_times: List[str] = []

        for meta in meta_batch:
            combined["prompt_tokens"] += int(_safe_number(meta.get("prompt_tokens"), 0))
            combined["completion_tokens"] += int(_safe_number(meta.get("completion_tokens"), 0))
            combined["total_tokens"] += int(_safe_number(meta.get("total_tokens"), 0))
            combined["latency_seconds"] += float(_safe_number(meta.get("latency_seconds"), 0.0))
            combined["retry_count"] += int(_safe_number(meta.get("retry_count"), 0))

            status_value = meta.get("status")
            if isinstance(status_value, str) and status_value:
                statuses.append(status_value)

            error_value = meta.get("error_message")
            if isinstance(error_value, str) and error_value:
                error_messages.append(error_value)

            request_value = meta.get("request_id")
            if isinstance(request_value, str) and request_value:
                request_ids.append(request_value)

            model_value = meta.get("model_version")
            if isinstance(model_value, str) and model_value and model_value not in model_versions:
                model_versions.append(model_value)

            start_value = meta.get("started_at_utc")
            if isinstance(start_value, str) and start_value:
                start_times.append(start_value)

            end_value = meta.get("ended_at_utc")
            if isinstance(end_value, str) and end_value:
                end_times.append(end_value)

        if any(status == "error" for status in statuses):
            combined["status"] = "error"
        elif statuses and all(status == "skipped" for status in statuses):
            combined["status"] = "skipped"
        else:
            combined["status"] = "success"

        if error_messages:
            unique_errors = list(dict.fromkeys(error_messages))
            combined["error_message"] = " | ".join(unique_errors)

        if request_ids:
            combined["request_id"] = " | ".join(request_ids)

        if model_versions:
            combined["model_version"] = " | ".join(model_versions)

        if start_times:
            combined["started_at_utc"] = min(start_times)

        if end_times:
            combined["ended_at_utc"] = max(end_times)

        return combined

    # 3. Initialize LLM Client
    try:
        client = LLMClient(cfg)
    except Exception as e:
        log.error(f"Failed to initialize LLM Client: {e}")
        sys.exit(1)

    def run_single_label_mode(report_text: str) -> Tuple[Dict[str, int], Dict[str, Any]]:
        per_label_meta: List[Dict[str, Any]] = []
        aggregated_labels: Dict[str, int] = {}

        for label in target_labels:
            label_response, label_meta = client.get_labels(report_text, labels_override=[label])
            aggregated_labels[label] = label_response.get(label, 0)
            per_label_meta.append(label_meta)

        combined_meta = combine_meta(per_label_meta)
        return aggregated_labels, combined_meta

    # 4. Optional resume support
    resume_volume_map, resume_hash_map = load_resume_data(resume_csv_path)
    resumed_count = 0
    hash_collision_skips = 0

    # 5. Process Reports
    log.info(f"Processing {len(df)} reports...")
    results = []
    
    # Accumulators for stats
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    latency_samples = []
    retry_samples = []
    status_counts: Counter[str] = Counter()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        vol_name = row["VolumeName"]
        report = row["report_text"]

        report_hash = ""
        if not pd.isna(report):
            report_hash = hashlib.sha256(str(report).encode("utf-8")).hexdigest()

        resume_row = None
        if resume_volume_map:
            resume_row = resume_volume_map.pop(vol_name, None)
            if resume_row:
                remove_hash_record(
                    resume_hash_map,
                    resume_row.get("report_hash"),
                    resume_row
                )
        if not resume_row and report_hash and resume_hash_map:
            bucket = resume_hash_map.get(report_hash)
            if bucket:
                if len(bucket) == 1:
                    resume_row = bucket.pop()
                    if not bucket:
                        resume_hash_map.pop(report_hash, None)
                    resume_volume_map.pop(resume_row.get("VolumeName"), None)
                else:
                    hash_collision_skips += 1

        row_was_resumed = resume_row is not None

        if row_was_resumed:
            resumed_count += 1
            row_result = dict(resume_row)
            row_result.setdefault("VolumeName", vol_name)
            row_result.setdefault("report_hash", report_hash)
            rounded_cost = round(_safe_number(row_result.get("estimated_cost_usd"), 0.0), cost_precision)
            row_result["estimated_cost_usd"] = rounded_cost
        else:
            # Delegate logic to the client package
            report_is_empty = pd.isna(report) or str(report).strip() == ""
            if prompt_mode == "single" and not report_is_empty:
                labels, meta = run_single_label_mode(report)
            else:
                labels, meta = client.get_labels(report)

            p_tok = meta.get("prompt_tokens", 0)
            c_tok = meta.get("completion_tokens", 0)
            total_tokens = meta.get("total_tokens", p_tok + c_tok)
            latency = meta.get("latency_seconds", 0.0)
            status = meta.get("status", "unknown")
            error_message = meta.get("error_message", "")
            request_id = meta.get("request_id", "")
            model_version = meta.get("model_version", cfg.api.model)
            retry_count = meta.get("retry_count", 0)
            started_at = meta.get("started_at_utc", "")
            ended_at = meta.get("ended_at_utc", "")
            row_cost = estimate_cost(
                cfg.api.model,
                p_tok,
                c_tok,
                pricing_table=pricing_table,
                precision=cost_precision
            )

            row_result = {
                "VolumeName": vol_name,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": total_tokens,
                "latency_seconds": latency,
                "estimated_cost_usd": row_cost,
                "model_version": model_version,
                "request_id": request_id,
                "retry_count": retry_count,
                "started_at_utc": started_at,
                "ended_at_utc": ended_at,
                "report_hash": report_hash,
                "status": status,
                "error_message": error_message
            }
            row_result.update(labels)

        normalized_row = normalize_row(row_result)
        results.append(normalized_row)

        p_tok = normalized_row.get("prompt_tokens", 0)
        c_tok = normalized_row.get("completion_tokens", 0)
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        total_cost += normalized_row.get("estimated_cost_usd", 0.0)
        latency = normalized_row.get("latency_seconds", 0.0)
        retry_count = normalized_row.get("retry_count", 0)
        status = normalized_row.get("status", "unknown") or "unknown"
        if status != "skipped":
            latency_samples.append(latency)
            retry_samples.append(retry_count)
        status_counts[status] += 1

        if not row_was_resumed:
            append_row(normalized_row)

    # 5. Save Output
    output_df = pd.DataFrame(results, columns=cols)
    
    # Robustness: Ensure we only save the requested columns and fill missing values sensibly
    numeric_cols = target_labels + [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_seconds",
        "estimated_cost_usd",
        "retry_count"
    ]
    output_df[numeric_cols] = output_df[numeric_cols].fillna(0)
    output_df[["model_version", "request_id", "started_at_utc", "ended_at_utc", "report_hash", "status", "error_message"]] = \
        output_df[["model_version", "request_id", "started_at_utc", "ended_at_utc", "report_hash", "status", "error_message"]].fillna("")
    output_df["estimated_cost_usd"] = output_df["estimated_cost_usd"].round(cost_precision)
    
    remaining_hash_rows = sum(len(items) for items in resume_hash_map.values())
    if resume_volume_map or remaining_hash_rows:
        log.warning(
            "%d resume rows were not matched to current input (by VolumeName) and %d unmatched hash entries remained.",
            len(resume_volume_map),
            remaining_hash_rows
        )
    if hash_collision_skips:
        log.warning(
            "Skipped %d resume rows due to duplicate report hashes; they were relabeled in this run.",
            hash_collision_skips
        )

    log.info(f"Saving labels to {output_path}")
    output_df.to_csv(output_path, index=False)
    
    # 6. Log Summary Statistics
    avg_latency = sum(latency_samples) / len(latency_samples) if latency_samples else 0.0
    avg_retries = sum(retry_samples) / len(retry_samples) if retry_samples else 0.0
    total_cost = round(total_cost, cost_precision)
    
    log.info("=== Run Summary ===")
    log.info(f"Total Reports Processed: {len(df)}")
    log.info(f"Previously Completed Rows: {resumed_count}")
    log.info(f"Total Prompt Tokens:     {total_prompt_tokens:,}")
    log.info(f"Total Completion Tokens: {total_completion_tokens:,}")
    log.info(f"Total Estimated Cost:    ${total_cost:.4f}")
    log.info(f"Average Latency (s):     {avg_latency:.2f}")
    log.info(f"Average Attempts/Call:   {avg_retries:.2f}")
    log.info(f"Successful Requests:     {status_counts.get('success', 0)}")
    log.info(f"Errored Requests:        {status_counts.get('error', 0)}")
    log.info(f"Skipped Reports:         {status_counts.get('skipped', 0)}")
    if any(k not in {"success", "error", "skipped"} for k in status_counts):
        log.info(f"Other Status Counts:     { {k: v for k, v in status_counts.items() if k not in {'success', 'error', 'skipped'}} }")
    log.info("===================")
    log.info("Label generation complete.")

if __name__ == "__main__":
    main()