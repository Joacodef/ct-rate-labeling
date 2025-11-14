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
    This command installs the dependencies defined in `pyproject.toml`.
    ```bash
    uv pip install -e .
    ```

## ⚙️ Configuration

All runtime parameters (paths, API settings, and prompts) are managed via Hydra in `configs/config.yaml`.

1.  **Input Data**:
    * Prepare an input CSV file containing, at minimum, a `VolumeName` column and a `report_text` column.
    * Update `io.reports_csv` in `configs/config.yaml` to point to this file.

2.  **API Key**:
    * The script requires an LLM API key (e.g., for OpenAI). Set this key as an environment variable:
        ```bash
        export OPENAI_API_KEY="sk-..."
        ```
    * The `configs/config.yaml` is pre-configured to read this variable using `${oc.env:OPENAI_API_KEY}`.

3.  **Labels and Prompts**:
    * Before running, review the `prompt.labels` list in `configs/config.yaml`. **These label names must exactly match the `training.target_labels` in the downstream `ct-rate-feature-benchmarks` project.**
    * Adjust the `prompt.system_prompt` as needed to optimize LLM performance.

## ⚡ Usage

To run the label generation script:

```bash
python scripts/generate_labels.py