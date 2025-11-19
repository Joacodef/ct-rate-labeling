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

### 1. Input Data & API Settings (`configs/config.yaml`)
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

## ⚡ Usage

To run the label generation script:

```bash
python scripts/generate_labels.py
````

You can also override configuration values directly from the command line:

```bash
python scripts/generate_labels.py io.reports_csv=data/new_reports.csv api.model=gpt-4o
```

