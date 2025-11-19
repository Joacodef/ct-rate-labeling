import logging
import os
import sys
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Attempt to import from the installed package. 
# If running locally without installation, fallback to adding src to path.
try:
    from ctr_labeling import LLMClient
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    from ctr_labeling import LLMClient

log = logging.getLogger(__name__)

def estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimates cost in USD based on standard OpenAI pricing (as of late 2024/2025).
    Prices are per 1M tokens.
    """
    # Pricing dictionary: (Input Price per 1M, Output Price per 1M)
    pricing = {
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4o": (5.00, 15.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-3.5-turbo": (0.50, 1.50)
    }
    
    # specific model version mapping (e.g., gpt-4-0125-preview -> gpt-4-turbo)
    model_base = model_name.lower()
    for key in pricing:
        if key in model_base:
            input_price, output_price = pricing[key]
            cost = (prompt_tokens / 1_000_000 * input_price) + \
                   (completion_tokens / 1_000_000 * output_price)
            return cost
            
    # Fallback if model not found
    return 0.0

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for generating binary classification labels from radiology reports.
    """
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

    # 3. Initialize LLM Client
    try:
        client = LLMClient(cfg)
    except Exception as e:
        log.error(f"Failed to initialize LLM Client: {e}")
        sys.exit(1)

    # 4. Process Reports
    log.info(f"Processing {len(df)} reports...")
    results = []
    
    # Accumulators for stats
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        vol_name = row["VolumeName"]
        report = row["report_text"]
        
        # Delegate logic to the client package
        # Now expects a tuple return: (labels, metadata)
        labels, meta = client.get_labels(report)
        
        # Update accumulators
        p_tok = meta.get("prompt_tokens", 0)
        c_tok = meta.get("completion_tokens", 0)
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        
        row_result = {
            "VolumeName": vol_name,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok
        }
        row_result.update(labels)
        results.append(row_result)

    # 5. Save Output
    output_df = pd.DataFrame(results)
    
    target_labels = list(cfg.prompt.labels)
    # Include token columns in the final CSV for auditing
    cols = ["VolumeName"] + target_labels + ["prompt_tokens", "completion_tokens"]
    
    # Robustness: Ensure we only save the requested columns
    # (Handle case where some rows might miss keys if logic failed badly, though client handles this)
    output_df = output_df.reindex(columns=cols, fill_value=0)
    
    output_path = cfg.io.output_csv
    log.info(f"Saving labels to {os.path.abspath(output_path)}")
    output_df.to_csv(output_path, index=False)
    
    # 6. Log Summary Statistics
    total_cost = estimate_cost(cfg.api.model, total_prompt_tokens, total_completion_tokens)
    
    log.info("=== Run Summary ===")
    log.info(f"Total Reports Processed: {len(df)}")
    log.info(f"Total Prompt Tokens:     {total_prompt_tokens:,}")
    log.info(f"Total Completion Tokens: {total_completion_tokens:,}")
    log.info(f"Estimated Total Cost:    ${total_cost:.4f}")
    log.info("===================")
    log.info("Label generation complete.")

if __name__ == "__main__":
    main()