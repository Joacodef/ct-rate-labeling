## Main Experiment: Full Dataset & Manual Set Labeling (gpt-5-nano)

**Goal:** Evaluate `gpt-5-nano` performance across 3 independent runs using both expert manual labels (gold standard) and RadBERT-generated labels (large-scale secondary verification).

**Commands Executed (3 Runs):**

```bash
# 1. Generation
python scripts/generate_labels.py io.reports_csv=data/reports_no_manual.csv api.model=gpt-5-nano
python scripts/generate_labels.py io.reports_csv=data/MANUAL_LABELS.csv api.model=gpt-5-nano

# 2. Evaluation (Gold Standard - Expert Labels)
python scripts/evaluate_prediction_performance.py --left-csv "data/MANUAL_LABELS.csv" --right-csv "outputs/Full Dataset Labeling/full_labels_1/manual_set_labels_gpt-5-nano.csv" --left-name "expert_truth" --right-name "gpt_5_nano" --output-dir "outputs/Evaluations/manual_labels_1_vs_expert"

# 3. Evaluation (Secondary Verification - RadBERT Labels)
python scripts/evaluate_prediction_performance.py --left-csv "data/all_predicted_labels.csv" --right-csv "outputs/Full Dataset Labeling/full_labels_1/full_labels_gpt-5-nano.csv" --left-name "radbert_truth" --right-name "gpt_5_nano" --output-dir "outputs/Evaluations/full_labels_1_vs_all_predicted"

```

### Part A: Gold Standard Evaluation (vs. Expert Manual Labels)

* **Mean Macro F1 (3 Runs):** 0.8912 ± 0.0044
* **Stability Check:** 0.0044 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)

| Label | Precision | Recall | F1 Score | Status |
| --- | --- | --- | --- | --- |
| **Lung Opacity** | 0.7541 ± 0.0075 | 0.8670 ± 0.0000 | 0.8066 ± 0.0043 | Stable, but precision remains the primary limiter (highest false positive rate). |
| **Lymphadenopathy** | 0.9542 ± 0.0060 | 0.9668 ± 0.0047 | 0.9605 ± 0.0053 | Strong and stable agreement. |
| **Pulm. Fibrotic Seq.** | 0.9010 ± 0.0058 | **0.8127 ± 0.0286** ⚠️ | 0.8543 ± 0.0165 | Lower recall with the highest variance across runs. |
| Arterial wall calc. | 0.8408 ± 0.0047 | 0.8983 ± 0.0066 | 0.8685 ± 0.0032 | Stable baseline. |
| Lung nodule | 0.9545 ± 0.0041 | 0.9778 ± 0.0092 | 0.9660 ± 0.0051 | Strong baseline. |
| **MACRO AVERAGE** | **0.8809 ± 0.0008** | **0.9045 ± 0.0086** | **0.8912 ± 0.0044** |  |


### Part B: Secondary Verification (vs. RadBERT all_predicted_labels)

* **Mean Macro F1 (3 Runs):** 0.9101 ± 0.0011
* **Stability Check:** 0.0011 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)

| Label | Precision | Recall | F1 Score | Status |
| --- | --- | --- | --- | --- |
| **Lung Opacity** | 0.8479 ± 0.0013 | 0.9081 ± 0.0005 | 0.8770 ± 0.0008 | Improved precision compared to manual set, stable variance. |
| **Lymphadenopathy** | 0.9678 ± 0.0005 | 0.9711 ± 0.0012 | 0.9694 ± 0.0008 | Near-perfect alignment with RadBERT labels. |
| **Pulm. Fibrotic Seq.** | 0.9241 ± 0.0004 | **0.7859 ± 0.0054** ⚠️ | 0.8494 ± 0.0033 | Recall remains the weakest metric, mirroring the manual evaluation trend. |
| Arterial wall calc. | 0.8741 ± 0.0005 | 0.8991 ± 0.0012 | 0.8865 ± 0.0008 | Stable baseline. |
| Lung nodule | 0.9787 ± 0.0001 | 0.9584 ± 0.0006 | 0.9684 ± 0.0004 | Strong baseline. |
| **MACRO AVERAGE** | **0.9185 ± 0.0005** | **0.9045 ± 0.0016** | **0.9101 ± 0.0011** |  |

### Conclusion

* **Action:** The 3 individual `gpt-5-nano` runs demonstrate excellent F1 performance and remarkable inter-run stability against both ground truths. Proceed to ensemble merging.
* **Goal:** Use `scripts/generate_consensus_ensemble.py` to aggregate these 3 independent runs via majority voting into a single, high-fidelity final label set.


## Agreement Between Runs (gpt-5-nano)

**Goal:** Quantify the stability and inter-run reliability of the `gpt-5-nano` prompt across the 3 independent labeling passes to justify building a consensus ensemble.

**Command Executed:**

```bash
python scripts/evaluate_label_agreement.py --csvs "outputs/Full Dataset Labeling/full_labels_1/full_labels_gpt-5-nano.csv" "outputs/Full Dataset Labeling/full_labels_2/full_labels_gpt-5-nano.csv" "outputs/Full Dataset Labeling/full_labels_3/full_labels_gpt-5-nano.csv" --names run1 run2 run3 --output-dir "outputs/Evaluations/gpt_runs_1_2_3_agreement"

```

### Aggregate Pairwise Agreement

* **run1 vs run2 (Macro Cohen's Kappa):** 0.9477 (Raw Agreement: 0.9792)
* **run1 vs run3 (Macro Cohen's Kappa):** 0.9459 (Raw Agreement: 0.9785)
* **run2 vs run3 (Macro Cohen's Kappa):** 0.9471 (Raw Agreement: 0.9789)
* **Status:** Near-perfect pairwise reliability (>0.90 Kappa across all pairs).

### All-Sets Consensus (3-Way Agreement)

| Label | 3-Way Agreement Rate | All-Positive Consensus (N) | All-Negative Consensus (N) | Status |
| --- | --- | --- | --- | --- |
| **Lung Opacity** | 0.9789 | 9,844 | 15,305 | Extremely stable across runs. |
| **Lymphadenopathy** | 0.9854 | 6,329 | 18,988 | Highest agreement metric among all targets. |
| **Pulm. Fibrotic Seq.** | 0.9379 | 4,989 | 19,107 | Slightly more variance, consistently mirroring the F1 evaluations. |
| Arterial wall calc. | 0.9613 | 6,857 | 17,841 | Strong baseline stability. |
| Lung nodule | 0.9783 | 11,147 | 13,987 | Strong baseline stability. |
| **MACRO AVERAGE** | **0.9684** |  |  |  |

### Conclusion

* **Action:** The exceptional inter-run agreement (>96% average complete consensus) confirms the deterministic reliability of the `gpt-5-nano` labeling mechanism.
* **Goal:** Generate a single, highly confident dataset using majority voting (`scripts/generate_consensus_ensemble.py`) to smooth out the remaining <4% of labeling discrepancies.


## Final Ensemble Performance (gpt-5-nano)

**Goal:** Evaluate the consensus ensemble outputs (Any Positive, Majority Vote, Unanimity) across 3 runs against both expert manual labels (gold standard) and RadBERT labels (secondary verification) to determine the optimal aggregation strategy.

**Commands Executed:**

```bash
# 1. Generate Consensus Ensembles
python scripts/generate_consensus_ensemble.py --csvs "outputs/Full Dataset Labeling/full_labels_1/full_labels_gpt-5-nano.csv" "outputs/Full Dataset Labeling/full_labels_2/full_labels_gpt-5-nano.csv" "outputs/Full Dataset Labeling/full_labels_3/full_labels_gpt-5-nano.csv" --names run1 run2 run3 --output-dir "outputs/gpt_runs_ensemble_consensus"

# 2. Batch Ensemble Evaluation (PowerShell)
.\run_ensemble_evaluations.ps1

```

### Part A: Gold Standard Evaluation (vs. Expert Manual Labels)

**Aggregate Macro Metrics by Consensus Type:**
| Consensus Rule | Precision | Recall | F1 Score | Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Any Positive** | 0.8766 | **0.9338** | **0.9034** | **0.9382** | **Best F1 & Recall:** Effectively recovers the lower recall seen in isolated runs, maximizing overall F1. |
| **Majority** | 0.8871 | 0.9120 | 0.8986 | 0.9370 | **Balanced:** Strong baseline with moderate precision/recall trade-offs. |
| **Unanimity** | **0.8969** | 0.8555 | 0.8730 | 0.9269 | **High Precision:** Maximizes precision but heavily penalizes recall, leading to the lowest F1. |

### Part B: Secondary Verification (vs. RadBERT all_predicted_labels)

**Aggregate Macro Metrics by Consensus Type:**
| Consensus Rule | Precision | Recall | F1 Score | Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Any Positive** | 0.9108 | **0.9431** | **0.9260** | **0.9519** | **Best F1 & Recall:** Consistent with manual evaluations; yields the highest overall agreement with RadBERT. |
| **Majority** | 0.9204 | 0.9106 | 0.9143 | 0.9470 | **Balanced:** Solid performance mirroring the average of the single runs. |
| **Unanimity** | **0.9251** | 0.8599 | 0.8881 | 0.9354 | **High Precision:** Peaks in precision but drops significantly in recall. |

### Conclusion

* **Action:** The **"Any Positive"** ensemble criteria yields the highest Macro F1 score and Accuracy against both the expert gold standard (0.9034 F1) and the RadBERT secondary verification (0.9260 F1). It successfully mitigates the false negatives (weak recall) observed in earlier single-run prompt engineering.
* **Goal:** Select the "Any Positive" (or "Majority" if tighter precision is clinically required) consensus labels as the final, high-fidelity dataset for downstream modeling.