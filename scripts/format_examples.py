import logging
import os
import sys
import json
import random
from typing import List

import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Utility to sample reports from a CSV and format them as YAML examples
    for insertion into configs/prompt/default.yaml.
    
    Usage:
        python scripts/format_examples.py io.reports_csv=data/prompt_examples.csv +n=5
    """
    # 1. Load Data
    input_path = cfg.io.reports_csv
    if not os.path.exists(input_path):
        log.error(f"Input file not found at: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    
    # 2. Determine Examples to Select
    # If specific indices are provided via command line (e.g., +indices=[0,5,10]) use them
    # Otherwise sample N random rows (default 3)
    indices = cfg.get("indices", None)
    n_samples = cfg.get("n", 3)
    
    if indices:
        # Hydra might pass this as a ListConfig, convert to standard list
        target_indices = list(indices)
        # Validate indices
        valid_indices = [i for i in target_indices if i in df.index]
        if len(valid_indices) < len(target_indices):
            log.warning(f"Some requested indices were out of bounds. Using: {valid_indices}")
        selected_df = df.loc[valid_indices]
    else:
        # Sample random rows
        n = min(n_samples, len(df))
        selected_df = df.sample(n=n, random_state=42)

    # 3. Format and Print YAML
    target_labels = list(cfg.prompt.labels)
    
    print("\n" + "="*60)
    print("COPY THE FOLLOWING BLOCK INTO configs/prompt/default.yaml")
    print("Under the 'examples:' key")
    print("="*60 + "\n")
    
    for _, row in selected_df.iterrows():
        report_text = row.get("report_text", "").strip()
        
        # Build the truth dictionary (0/1 ints)
        output_dict = {}
        for label in target_labels:
            val = row.get(label, 0)
            # Safe conversion to int
            try:
                val = int(float(val))
            except (ValueError, TypeError):
                val = 0
            output_dict[label] = val
            
        # Manually construct YAML-like string to ensure readability
        # Using json.dumps for the output dict keeps it on one line which is cleaner for config files
        print("  - report: |")
        # Indent report text
        for line in report_text.splitlines():
            print(f"      {line}")
        print(f"    output: {json.dumps(output_dict)}")
        print("") # Empty line between examples

    print("="*60 + "\n")

if __name__ == "__main__":
    main()