import argparse
import hashlib
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig

# Load environment variables from .env file
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

# Boilerplate to ensure we can import from src/ even if not installed
try:
    from ctr_labeling import LLMClient, estimate_cost, safe_cfg_to_yaml
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    from ctr_labeling import LLMClient, estimate_cost, safe_cfg_to_yaml

log = logging.getLogger(__name__)

def calculate_binary_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> Dict[str, float]:
    """Compute precision/recall/F1 for binary classification with edge cases handled.

    Args:
        df: DataFrame containing the ground-truth and prediction columns.
        true_col: Column name holding manual labels (0/1).
        pred_col: Column name holding model predictions (0/1).

    Returns:
        Dictionary with precision, recall, F1, and confusion-matrix counts.
    """
    # Ensure inputs are numeric 0/1
    y_true = pd.to_numeric(df[true_col], errors='coerce').fillna(0).astype(int)
    y_pred = pd.to_numeric(df[pred_col], errors='coerce').fillna(0).astype(int)

    # True Positives, False Positives, False Negatives, True Negatives
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    # Check for "Zero Support" case (No positives in Ground Truth)
    if (tp + fn) == 0:
        if fp == 0:
            # Perfect Rejection: Model correctly predicted 0 for all cases.
            # We assign 1.0 to reflect that the model made no errors.
            return {
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
            }
        else:
            # Hallucination: Model predicted positive when none existed.
            # Precision is 0 (0 correct / N predicted). Recall is undefined (assigned 0).
            return {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
            }

    # Standard Calculation
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


def get_case_insensitive_column(df: pd.DataFrame, target_name: str) -> str:
    """Return the concrete column name matching ``target_name`` regardless of case.

    Args:
        df: DataFrame whose columns should be searched.
        target_name: Desired column name spelled in any case.

    Returns:
        The exact column name from ``df`` matching the requested name.

    Raises:
        KeyError: If the column cannot be found in any casing.
    """
    matches = [col for col in df.columns if col.lower() == target_name.lower()]
    if not matches:
        raise KeyError(f"Column '{target_name}' not found (case-insensitive search).")
    return matches[0]


def combine_meta(meta_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metadata from multiple per-label calls into a single record.

    Args:
        meta_batch: Sequence of dictionaries emitted by the LLM client per request.

    Returns:
        Dictionary summarizing token counts, latency, status, and identifiers.
    """
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
        combined["prompt_tokens"] += int(meta.get("prompt_tokens", 0) or 0)
        combined["completion_tokens"] += int(meta.get("completion_tokens", 0) or 0)
        combined["total_tokens"] += int(meta.get("total_tokens", meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0)) or 0)
        combined["latency_seconds"] += float(meta.get("latency_seconds", 0.0) or 0.0)
        combined["retry_count"] += int(meta.get("retry_count", 0) or 0)

        status_val = meta.get("status")
        if isinstance(status_val, str) and status_val:
            statuses.append(status_val)

        error_val = meta.get("error_message")
        if isinstance(error_val, str) and error_val:
            error_messages.append(error_val)

        req_val = meta.get("request_id")
        if isinstance(req_val, str) and req_val:
            request_ids.append(req_val)

        model_val = meta.get("model_version")
        if isinstance(model_val, str) and model_val and model_val not in model_versions:
            model_versions.append(model_val)

        start_val = meta.get("started_at_utc")
        if isinstance(start_val, str) and start_val:
            start_times.append(start_val)

        end_val = meta.get("ended_at_utc")
        if isinstance(end_val, str) and end_val:
            end_times.append(end_val)

    if any(status == "error" for status in statuses):
        combined["status"] = "error"
    elif statuses and all(status == "skipped" for status in statuses):
        combined["status"] = "skipped"

    if error_messages:
        combined["error_message"] = " | ".join(dict.fromkeys(error_messages))

    if request_ids:
        combined["request_id"] = " | ".join(request_ids)

    if model_versions:
        combined["model_version"] = " | ".join(model_versions)

    if start_times:
        combined["started_at_utc"] = min(start_times)

    if end_times:
        combined["ended_at_utc"] = max(end_times)

    return combined


def _safe_number(value: Any, default: float = 0.0) -> float:
    """Convert arbitrary input to float, guarding against pandas NA objects.

    Args:
        value: Raw input that may need coercion to float.
        default: Fallback value when conversion is unsafe.

    Returns:
        A floating point representation of ``value`` or ``default`` on failure.
    """
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
    """Load an existing predictions CSV so evaluation can resume without rework.

    Args:
        path: Location of the predictions CSV produced by a previous run.

    Returns:
        Tuple of (volume_map, hash_map) ready for resume lookup.
    """
    if not path or not os.path.exists(path):
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
    by_volume: Dict[str, Dict[str, Any]] = {}
    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for _, row in resume_df.iterrows():
        record = row.to_dict()
        vol = record.get("VolumeName")
        if isinstance(vol, str) and vol:
            by_volume[vol] = record
        rhash = record.get("report_hash")
        if isinstance(rhash, str) and rhash:
            by_hash[rhash].append(record)

    log.info(
        "Loaded %d previously evaluated rows from %s for resume support.",
        len(resume_df),
        path
    )
    return by_volume, by_hash


def remove_hash_record(
    hash_map: Dict[str, List[Dict[str, Any]]],
    rhash: str,
    record: Dict[str, Any]
) -> None:
    """Remove a specific resume row from the hash bucket when consumed.

    Args:
        hash_map: Mapping of report hashes to cached resume records.
        rhash: Hash key that should contain the target record.
        record: Resume row to remove from the hash bucket.

    Returns:
        None. Mutates ``hash_map`` in place.
    """
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

@hydra.main(version_base=None, config_path="../configs", config_name="prompt_engineering")
def main(cfg: DictConfig) -> None:
    """Evaluate prompt performance by comparing LLM predictions to ground truth.

    Args:
        cfg: Hydra configuration referencing the evaluation CSV, prompt, and model.

    Raises:
        SystemExit: If inputs/configuration are invalid or the LLM client fails.
    """
    resume_predictions_path = ""
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
        resume_predictions_path = str(resume_dir / "evaluation_predictions.csv")
        log.info(
            "Resuming evaluation with config from %s; predictions CSV set to %s",
            resume_cfg_path,
            resume_predictions_path
        )
        log.info("Loaded resume configuration:\n%s", safe_cfg_to_yaml(cfg))
    else:
        log.info("Starting prompt evaluation with configuration:\n%s", safe_cfg_to_yaml(cfg))

    # 1. Validate Input
    input_path = cfg.io.reports_csv
    if not os.path.exists(input_path):
        log.error(f"Input file not found at: {input_path}")
        sys.exit(1)

    log.info(f"Loading ground truth data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        log.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    try:
        volume_col = get_case_insensitive_column(df, "VolumeName")
    except KeyError as exc:
        log.error(str(exc))
        sys.exit(1)

    try:
        report_col = get_case_insensitive_column(df, "report_text")
    except KeyError as exc:
        log.error(str(exc))
        sys.exit(1)

    # 2. Validate Ground Truth Columns
    target_labels = list(cfg.prompt.labels)
    prompt_mode = str(cfg.prompt.get("mode", "multi")).lower()
    if prompt_mode not in {"multi", "single"}:
        log.error("prompt.mode must be either 'multi' or 'single'. Got '%s'", prompt_mode)
        sys.exit(1)
    missing_cols = [label for label in target_labels if label not in df.columns]
    
    if missing_cols:
        log.error(
            f"The input CSV is missing the following ground truth columns required for evaluation: {missing_cols}\n"
            "Ensure you are pointing to a file that contains manual labels (e.g., tuning_set.csv)."
        )
        sys.exit(1)

    if RESUME_RUN_DIR:
        output_dir = RESUME_RUN_DIR
    else:
        output_dir = HydraConfig.get().runtime.output_dir

    predictions_path = resume_predictions_path or os.path.join(output_dir, "evaluation_predictions.csv")

    prediction_cols = [
        "VolumeName",
        "report_hash",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_seconds",
        "model_version",
        "request_id",
        "retry_count",
        "started_at_utc",
        "ended_at_utc",
        "status",
        "error_message"
    ] + [f"pred_{label}" for label in target_labels]

    predictions_initialized = os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0

    def append_prediction_row(row: Dict[str, Any]) -> None:
        """Append a single prediction/metadata row to the CSV on disk.

        Args:
            row: Dictionary with prediction columns matching ``prediction_cols``.

        Returns:
            None. Writes streamed output to ``predictions_path``.
        """
        nonlocal predictions_initialized
        df_row = pd.DataFrame([row], columns=prediction_cols)
        df_row.to_csv(
            predictions_path,
            mode="a",
            header=not predictions_initialized,
            index=False
        )
        predictions_initialized = True

    resume_volume_map, resume_hash_map = load_resume_data(predictions_path)
    resumed_count = 0
    hash_collision_skips = 0

    # 3. Initialize Client
    try:
        client = LLMClient(cfg)
    except Exception as e:
        log.error(f"Failed to initialize LLM Client: {e}")
        sys.exit(1)

    # Setup Cost & Stats tracking
    pricing_cfg = cfg.api.get("pricing", {})
    # Convert OmegaConf object to standard dict for the estimator
    pricing_table = OmegaConf.to_container(pricing_cfg, resolve=True) if pricing_cfg else {}
    
    total_cost = 0.0
    latencies = []

    # 4. Processing Loop
    predictions = []
    discrepancies = []

    log.info(f"Evaluating on {len(df)} reports...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        vol_name = row.get(volume_col, "Unknown")
        report_text = row.get(report_col, "")

        report_hash = ""
        if not pd.isna(report_text):
            report_hash = hashlib.sha256(str(report_text).encode("utf-8")).hexdigest()

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
            predicted_labels = {
                label: int(_safe_number(resume_row.get(f"pred_{label}"), 0))
                for label in target_labels
            }
            meta = {
                "prompt_tokens": int(_safe_number(resume_row.get("prompt_tokens"), 0)),
                "completion_tokens": int(_safe_number(resume_row.get("completion_tokens"), 0)),
                "total_tokens": int(_safe_number(resume_row.get("total_tokens"), 0)),
                "latency_seconds": float(_safe_number(resume_row.get("latency_seconds"), 0.0)),
                "model_version": resume_row.get("model_version", cfg.api.model),
                "request_id": resume_row.get("request_id", ""),
                "retry_count": int(_safe_number(resume_row.get("retry_count"), 0)),
                "started_at_utc": resume_row.get("started_at_utc", ""),
                "ended_at_utc": resume_row.get("ended_at_utc", ""),
                "status": resume_row.get("status", "success"),
                "error_message": resume_row.get("error_message", "")
            }
            if not report_hash:
                report_hash = resume_row.get("report_hash", "")
        else:
            # Get LLM Predictions (single mode mirrors scripts/generate_labels.py behavior)
            if prompt_mode == "single" and not (pd.isna(report_text) or str(report_text).strip() == ""):
                per_label_meta: List[Dict[str, Any]] = []
                aggregated_predictions: Dict[str, int] = {}
                for label in target_labels:
                    label_resp, label_meta = client.get_labels(report_text, labels_override=[label])
                    aggregated_predictions[label] = label_resp.get(label, 0)
                    per_label_meta.append(label_meta)
                meta = combine_meta(per_label_meta)
                predicted_labels = aggregated_predictions
            else:
                # Note: Error handling logic inside client returns zeros on failure
                predicted_labels, meta = client.get_labels(report_text)
        
        # Track Stats
        latency = meta.get("latency_seconds", 0.0)
        latencies.append(latency)
        
        p_tok = meta.get("prompt_tokens", 0)
        c_tok = meta.get("completion_tokens", 0)
        total_tokens = meta.get("total_tokens", p_tok + c_tok)
        if (not total_tokens) and (p_tok or c_tok):
            total_tokens = p_tok + c_tok
        
        cost = estimate_cost(cfg.api.model, p_tok, c_tok, pricing_table)
        total_cost += cost

        storage_row = {
            "VolumeName": vol_name,
            "report_hash": report_hash,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": total_tokens,
            "latency_seconds": latency,
            "model_version": meta.get("model_version", cfg.api.model),
            "request_id": meta.get("request_id", ""),
            "retry_count": meta.get("retry_count", 0),
            "started_at_utc": meta.get("started_at_utc", ""),
            "ended_at_utc": meta.get("ended_at_utc", ""),
            "status": meta.get("status", ""),
            "error_message": meta.get("error_message", "")
        }
        storage_row.update({f"pred_{k}": predicted_labels.get(k, 0) for k in target_labels})

        # Record result row
        result_row = {"VolumeName": vol_name}
        result_row.update({f"pred_{k}": v for k, v in predicted_labels.items()})
        
        # Keep original truths for easier dataframe construction
        for label in target_labels:
            actual_val = row[label]
            result_row[f"true_{label}"] = actual_val
            
            # Check for Discrepancy
            # Normalize actual_val to int (0/1) for comparison
            try:
                actual_int = int(float(actual_val)) if pd.notna(actual_val) else 0
            except ValueError:
                actual_int = 0
                
            pred_int = predicted_labels.get(label, 0)
            
            if pred_int != actual_int:
                discrepancies.append({
                    "VolumeName": vol_name,
                    "Label": label,
                    "Predicted": pred_int,
                    "Actual": actual_int,
                    "Report": report_text
                })

        predictions.append(result_row)

        if not row_was_resumed:
            append_prediction_row(storage_row)

    remaining_hash_rows = sum(len(items) for items in resume_hash_map.values())
    if resume_volume_map or remaining_hash_rows:
        log.warning(
            "%d resume rows were not matched to current input (by VolumeName) and %d unmatched hash entries remained.",
            len(resume_volume_map),
            remaining_hash_rows
        )
    if hash_collision_skips:
        log.warning(
            "Skipped %d resume rows due to duplicate report hashes; they were reevaluated in this run.",
            hash_collision_skips
        )

    if resumed_count:
        log.info("Reused %d previously evaluated reports via resume support.", resumed_count)

    # 5. Calculate Metrics
    results_df = pd.DataFrame(predictions)
    metric_rows = []
    
    log.info("Calculating metrics...")
    for label in target_labels:
        true_col = f"true_{label}"
        pred_col = f"pred_{label}"
        
        stats = calculate_binary_metrics(results_df, true_col, pred_col)
        stats["label"] = label
        metric_rows.append(stats)

    metrics_df = pd.DataFrame(metric_rows)
    
    # Calculate Macro Averages
    macro_avg = {
        "label": "MACRO_AVERAGE",
        "precision": round(metrics_df["precision"].mean(), 4),
        "recall": round(metrics_df["recall"].mean(), 4),
        "f1": round(metrics_df["f1"].mean(), 4),
        "tp": metrics_df["tp"].sum(),
        "fp": metrics_df["fp"].sum(),
        "fn": metrics_df["fn"].sum(),
        "tn": metrics_df["tn"].sum(),
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([macro_avg])], ignore_index=True)

    # 6. Save Outputs
    metrics_path = os.path.join(output_dir, "evaluation_metrics.csv")
    discrepancies_path = os.path.join(output_dir, "discrepancies.csv")
    summary_path = os.path.join(output_dir, "run_summary.json")
    
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(discrepancies).to_csv(discrepancies_path, index=False)

    # Calculate run-level stats
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    # Save run summary to JSON
    import json
    with open(summary_path, 'w') as f:
        json.dump({
            "total_estimated_cost_usd": round(total_cost, 6),
            "average_latency_seconds": round(avg_latency, 4),
            "total_reports": len(df),
            "model_version": cfg.api.model,
            "macro_f1": macro_avg["f1"],
            "macro_precision": macro_avg["precision"],
            "macro_recall": macro_avg["recall"]
        }, f, indent=4)

    # 7. Print Summary to Console
    # avg_latency was calculated above for the JSON save
    
    print("\n" + "="*60)
    print(f"EVALUATION COMPLETE")
    print("="*60)
    print(f"Total Estimated Cost:   ${total_cost:.4f}")
    print(f"Avg Latency per Report: {avg_latency:.2f}s")
    print("-" * 60)
    print(f"Metrics saved to:       {metrics_path}")
    print(f"Discrepancies saved to: {discrepancies_path}")
    print(f"Run summary saved to:   {summary_path}")
    print("-" * 60)
    
    # Format for readability
    summary_view = metrics_df[["label", "precision", "recall", "f1", "tp", "fp", "fn"]]
    print(summary_view.to_string(index=False))
    print("="*60 + "\n")

if __name__ == "__main__":
    main()