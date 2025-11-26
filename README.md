# CT-RATE Labeling

A repository for generating binary classification labels from unstructured radiology reports using a Large Language Model (LLM).

## 🎯 Project Goal

This project's sole purpose is to create a binary classification CSV file from raw radiology text reports. The output of this repository serves as a direct upstream input for the `ct-rate-feature-benchmarks` repository.

The generated CSV is designed to be consumed by the `prepare_manifests.py` script via its `--labels_csv` argument.

## 🚀 Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-org/ct-rate-labeling.git](https://github.com/your-org/ct-rate-labeling.git)
    cd ct-rate-labeling
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    This command installs the dependencies defined in `pyproject.toml` and the package in editable mode.
    ```bash
    uv pip install -e .
    ```

## ⚙️ Configuration

All runtime parameters are managed via [Hydra](https://hydra.cc/).

### 1. Input Data & API Settings (`configs/prompt_engineering.yaml`)
* **Input CSV**: Update `io.reports_csv` to point to your input file. This file must contain:
    * `VolumeName`: The unique identifier for the scan (e.g., `train_1_a_1.nii.gz`).
    * `report_text`: The raw radiology report.
* **API Key**: Set your OpenAI API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```
    The config automatically reads this via `${oc.env:OPENAI_API_KEY}`.

### 2. Labels and Prompts (`configs/prompt/default.yaml`)
* **Target Labels**: The list of abnormalities to extract is defined in `prompt.labels`.
    * **Crucial**: These names must exactly match the `training.target_labels` in the downstream `ct-rate-feature-benchmarks` project.
* **System Prompt**: You can modify the `system_prompt` in this file to refine the LLM's instructions (e.g., how to handle uncertainty or negation).

## 🧪 Methodology & Prompt Engineering

This repository employs an iterative approach to prompt tuning. The goal is to maximize classification performance (F1 Score) while minimizing inference costs before processing the full dataset.

### 1. Data Strategy
We split the validation process into two stages to balance speed and statistical significance:

* **Fast Loop (`data/tiny_tuning_set.csv`)**: A small, curated set (30–50 reports) used for rapid prototyping. Contains borderline cases and difficult negatives.
* **Validation Set (`data/tuning_set.csv`)**: A larger set (~600 reports) used for final confirmation of metrics.
* **Production Set**: The full dataset (e.g., `data/all_reports.csv`) is processed only after the configuration is frozen.

### 2. Experimental Workflow
We use `scripts/evaluate_prompt.py` to compare different models, prompting strategies, and modes.

#### Step A: Establish Baseline (Zero-Shot)
Start with the cheapest model and simplest prompt to set a baseline cost and F1 score.
```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
    prompt=zero-shot_multi \
    api.model=gpt-5-nano
````

#### Step B: Strategy Iteration

Test whether few-shot examples or single-label prompting improves performance on difficult pathologies.

  * **Few-Shot:** Inject curated examples (defined in `configs/prompt/3-shot_multi_v1.yaml`).
    ```bash
    python scripts/evaluate_prompt.py --config-name prompt_engineering \
        prompt=3-shot_multi_v1 \
        prompt.examples_enabled=true
    ```
  * **Single-Label Mode:** Force the LLM to focus on one label at a time (higher cost, potentially higher accuracy for subtle findings).
    ```bash
    python scripts/evaluate_prompt.py --config-name prompt_engineering \
        prompt=zero-shot_single \
        api.model=gpt-5-nano
    ```

#### Step C: Model Sweeps

Once the prompt strategy is fixed, use Hydra's multi-run (`-m`) capability to find the "intelligence vs. cost" sweet spot.

```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
    -m api.model=gpt-5-nano,gpt-5-mini,gpt-4.1 \
    prompt=3-shot_multi_v1
```

### 3\. Evaluation Metrics

The script outputs:

1.  **`evaluation_metrics.csv`**: Precision, Recall, and F1 for each label.
2.  **`run_summary.json`**: Total estimated cost (USD) and average latency.
3.  **`discrepancies.csv`**: A row-by-row log of where the LLM disagreed with the ground truth, useful for debugging vague reports.

<!-- end list -->


## ⚡ Usage

To run the label generation script:

```bash
python scripts/generate_labels.py
````

You can also override configuration values directly from the command line:

```bash
python scripts/generate_labels.py io.reports_csv=data/new_reports.csv api.model=gpt-4o
```

