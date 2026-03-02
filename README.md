# CT-RATE Labeling

Generate binary abnormality labels from unstructured CT radiology reports using LLM prompts, then evaluate, compare, and ensemble those predictions.

The primary output is a labels CSV intended for downstream use (for example as `--labels_csv` input in `ct-rate-feature-benchmarks/prepare_manifests.py`).

## Project Scope

This repository covers four main capabilities:

1. **Label generation** from report text (`scripts/generate_labels.py`)
2. **Prompt/model evaluation** against manual labels (`scripts/evaluate_prompt.py`)
3. **Agreement + performance analysis** across runs/CSV pairs (`scripts/evaluate_*.py`)
4. **Consensus ensembling + utility tooling** (`scripts/generate_consensus_ensemble.py` + helper scripts)

## Installation

This project uses [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Joacodef/ct-rate-labeling.git
cd ct-rate-labeling

uv venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Runtime only
uv pip install -e .

# Runtime + tests
uv pip install -e ".[tests]"
```

Run tests:

```bash
uv run pytest tests
```

## Required Input Format

Most workflows assume an input CSV with:

- `VolumeName` (case-insensitive lookup supported)
- `report_text`

For evaluation against ground truth (`evaluate_prompt.py`), the same file must also include all label columns defined in the selected prompt config.

## Configuration (Hydra)

- Root config: `configs/config.yaml` (defaults to `prompt_engineering`)
- Runtime config: `configs/prompt_engineering.yaml`
- Prompt presets: `configs/prompt/*.yaml`

Key fields in `configs/prompt_engineering.yaml`:

- `io.reports_csv`: input file
- `io.output_csv`: output filename (relative to Hydra run dir)
- `api.model`, `api.reasoning_effort`, `api.max_completion_tokens`, `api.max_retries`
- `api.pricing`: per-model cost estimation table
- `api.log_requests`, `api.log_path`, `api.log_requests_include_text`
- `prompt`: selected prompt file containing labels/system prompt/examples

Authentication:

```bash
# PowerShell
$env:OPENAI_API_KEY="sk-..."
```

## Core Workflows

### 1) Generate labels for a dataset

```bash
python scripts/generate_labels.py
```

Typical override example:

```bash
python scripts/generate_labels.py io.reports_csv=data/new_reports.csv api.model=gpt-5-nano prompt=3-shot_multi_v3
```

Resume a previous Hydra run without relabeling completed rows:

```bash
python scripts/generate_labels.py --resume outputs/2026-02-02/gpt-5-nano_3-shot_multi_v3_12-00-00-000000
```

Outputs:

- labels CSV (path from `io.output_csv`)
- run config snapshot (`run_config.yaml`)
- optional request log (`api.log_path`)

### 2) Evaluate prompt/model against manual labels

```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering prompt=zero-shot_multi api.model=gpt-5-nano
```

Single-label mode example:

```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering prompt=zero-shot_single api.model=gpt-5-nano
```

Hydra multirun model sweep:

```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering -m api.model=gpt-5-nano,gpt-5-mini,gpt-4.1 prompt=3-shot_multi_v3
```

Resume evaluation run:

```bash
python scripts/evaluate_prompt.py --resume outputs/2026-02-02/gpt-5-nano_3-shot_multi_v3_12-00-00-000000
```

Outputs:

- `evaluation_predictions.csv`
- `evaluation_metrics.csv`
- `discrepancies.csv`
- `run_summary.json`

## Script Reference

### Generation / Evaluation

- `scripts/generate_labels.py`  
    Generate binary labels per report with token/latency/cost metadata; supports `prompt.mode=multi|single` and `--resume`.

- `scripts/evaluate_prompt.py`  
    Run labeling on a ground-truth dataset and compute per-label metrics (precision/recall/F1 + macro summary), discrepancies, and run summary.

### Comparison / Agreement / Ensemble

- `scripts/evaluate_prediction_performance.py`  
    Compare two label CSVs as reference vs prediction (alignment by normalized IDs), output metrics/discrepancies/summary.

- `scripts/evaluate_label_agreement.py`  
    Compute pairwise agreement, Cohen’s kappa, MCC, and all-sets agreement across 2+ CSVs.

- `scripts/generate_consensus_ensemble.py`  
    Build consensus CSVs from 2+ runs using `any_positive`, `majority`, and/or `unanimity` voting.

### Utility Scripts

- `scripts/make_unique_scans_csv.py`  
    Deduplicate reconstruction-level CT-RATE rows into one row per scan and build `report_text` from report columns.

- `scripts/make_tiny_tuning_set.py`  
    Stratified multi-label sampler to create a small tuning subset while preserving prevalence.

- `scripts/format_examples.py`  
    Sample rows and print YAML-ready few-shot examples for prompt config files.

- `scripts/aggregate_metrics.py`  
    Aggregate multiple `evaluation_metrics.csv` files and append per-label std-dev columns.

- `scripts/summarize_discrepancies.py`  
    Summarize false positives/false negatives by label from a `discrepancies.csv` file.

- `run_ensemble_evaluations.ps1`  
    PowerShell batch runner that evaluates all three consensus files (`any_positive`, `majority`, `unanimity`) against both default ground truths (`data/all_predicted_labels.csv` and `data/MANUAL_LABELS.csv`).

## Example Utility Commands

Create a tiny stratified tuning set:

```bash
python scripts/make_tiny_tuning_set.py --input data/tuning_set.csv --output data/tiny_tuning_set.csv --prompt configs/prompt/3-shot_multi_v3.yaml --n 100
```

Create one-row-per-scan CSV from CT-RATE source export:

```bash
python scripts/make_unique_scans_csv.py --input data/raw_ct_rate.csv --output data/all_reports.csv --report-columns Findings_EN,Impressions_EN
```

Compare predictions against manual labels:

```bash
python scripts/evaluate_prediction_performance.py --left-csv data/MANUAL_LABELS.csv --right-csv "outputs/Full Dataset Labeling/full_labels_1/full_labels_gpt-5-nano.csv" --output-dir outputs/Evaluations/gpt_runs_vs_ground_truth
```

Compute agreement across 3 independent runs:

```bash
python scripts/evaluate_label_agreement.py --csvs "outputs/Full Dataset Labeling/full_labels_1/full_labels_gpt-5-nano.csv" "outputs/Full Dataset Labeling/full_labels_2/full_labels_gpt-5-nano.csv" "outputs/Full Dataset Labeling/full_labels_3/full_labels_gpt-5-nano.csv" --names run1 run2 run3 --output-dir outputs/Evaluations/gpt_runs_1_2_3_agreement
```

Generate consensus ensemble files:

```bash
python scripts/generate_consensus_ensemble.py --csvs "outputs/Full Dataset Labeling/full_labels_1/full_labels_gpt-5-nano.csv" "outputs/Full Dataset Labeling/full_labels_2/full_labels_gpt-5-nano.csv" "outputs/Full Dataset Labeling/full_labels_3/full_labels_gpt-5-nano.csv" --names run1 run2 run3 --output-dir outputs/gpt_runs_ensemble_consensus
```

Run all ensemble-vs-ground-truth evaluations (PowerShell):

```powershell
.\run_ensemble_evaluations.ps1
```

Summarize most frequent discrepancy types:

```bash
python scripts/summarize_discrepancies.py outputs/Evaluations/gpt_runs_vs_ground_truth/discrepancies.csv --top 20
```

Aggregate metrics from multiple prompt-engineering runs:

```bash
python scripts/aggregate_metrics.py --glob "outputs/**/evaluation_metrics.csv" --output outputs/Evaluations/aggregated_metrics.csv
```

## Prompt Engineering Notes

- Prompt files define:
    - label list (`labels`)
    - `mode` (`multi` or `single`)
    - `system_prompt`
    - optional few-shot `examples`
- `single` mode triggers one API call per label (more expensive, sometimes better on subtle findings).
- `multi` mode predicts all labels in one call per report.

## Output Organization

By default, Hydra writes run artifacts under `outputs/<date>/...` and stores resolved config in `.hydra/`.

Common artifacts generated by workflows:

- predictions/labels CSVs
- metrics CSVs
- discrepancy CSVs
- `run_summary.json`
- optional LLM request-response logs (`llm_calls.jsonl`)

## Core Package (`src/ctr_labeling`)

- `llm_client.py`: OpenAI client wrapper with retries, metadata capture, response parsing, and optional debug logging.
- `parsing.py`: JSON cleanup + robust 0/1 normalization/validation.
- `utils.py`: pricing-based cost estimation + config redaction helper.

## Notes

- Column matching for key fields is case-insensitive in major scripts.
- Most comparison scripts normalize IDs by stripping `.nii.gz/.nii` and reconstruction suffixes (unless overridden by flags).
- Resume mode (`--resume`) is designed to prevent duplicate API calls on interrupted runs.

