# Prompt Engineering Log

## Step 1: Baseline (Zero-Shot)
**Date:** 2025-12-15
**Config:** `zero-shot_multi` | **Model:** `gpt-5-nano`

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.6922 ± 0.0352
* **Stability Check:** 0.0352 (Target < 0.02) -> ❌ **FAIL**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.3561 ± 0.0106** 🚨 | 1.0000 ± 0.0000 | 0.5252 ± 0.0115 | **Primary Failure:** Consistent Hallucinations (High Recall, Low Precision). |
| **Lymphadenopathy** | 1.0000 ± 0.0000 | **0.3667 ± 0.0577** ⚠️ | 0.5348 ± 0.0635 | **Unstable:** Recall varies significantly between runs (0.30 vs 0.40). |
| **Pulm. Fibrotic Seq.** | 0.5238 ± 0.0412 | 0.8333 ± 0.1443 | 0.6424 ± 0.0735 | **Unstable:** Both Precision and Recall fluctuate noticeably. |
| Arterial wall calc. | 0.8214 ± 0.0618 | 1.0000 ± 0.0000 | 0.9011 ± 0.0381 | Stable / Good baseline. |
| Lung nodule | 0.7505 ± 0.0245 | 1.0000 ± 0.0000 | 0.8574 ± 0.0162 | Acceptable baseline. |
| **MACRO AVERAGE** | **0.6904 ± 0.0242** | **0.8400 ± 0.0361** | **0.6922 ± 0.0352** | |

### Conclusion
* **Action:** Proceed to Step 2 (Few-Shot).
* **Goal:** Use examples specifically to restrain "Lung Opacity" hallucinations and stabilize "Lymphadenopathy" definition.