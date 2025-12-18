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


## Step 2: Few-Shot Check
**Date:** 2025-12-17
**Config:** `3-shot_multi_v1` | **Model:** `gpt-5-nano`

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.6850 ± 0.0456
* **Stability Check:** 0.0456 (Target < 0.02) -> ❌ **FAIL** (Worse than baseline)

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.3752 ± 0.0118** 🚨 | 1.0000 ± 0.0000 | 0.5457 ± 0.0124 | **Primary Failure:** Hallucinations persist. Examples were insufficient to override uncertainty bias. |
| **Lymphadenopathy** | 1.0000 ± 0.0000 | **0.4000 ± 0.0000** ✅ | 0.5714 ± 0.0000 | **Success:** Recall stabilized perfectly. The specific example (size threshold) worked. |
| **Pulm. Fibrotic Seq.** | 0.4905 ± 0.0863 | 0.7500 ± 0.2500 | 0.5906 ± 0.1417 | **Unstable:** High variance introduced by the new prompt length/noise. |
| Arterial wall calc. | 0.7579 ± 0.0954 | 1.0000 ± 0.0000 | 0.8601 ± 0.0617 | Stable. |
| Lung nodule | 0.7505 ± 0.0245 | 1.0000 ± 0.0000 | 0.8574 ± 0.0162 | Stable. |
| **MACRO AVERAGE** | **0.6749 ± 0.0425** | **0.8300 ± 0.0500** | **0.6850 ± 0.0456** | |

### Conclusion
* **Action:** Proceed to Step 3 (Example Tuning / Constraints).
* **Goal:** The subtle few-shot approach failed for "Lung Opacity". We need **Negative Constraints** in the system prompt to explicitly forbid classifying Atelectasis/Nodules as Opacity unless consolidation is present.



## Step 3: Example Tuning (Negative Constraints)
**Date:** 2025-12-17
**Config:** `3-shot_multi_v2` | **Model:** `gpt-5-nano`

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.7406 ± 0.0089
* **Stability Check:** 0.0089 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.4570 ± 0.0169** 🔼 | 1.0000 ± 0.0000 | 0.6272 ± 0.0150 | **Improved:** Negative constraints reduced False Positives by ~30% (Precision 0.37 -> 0.46). |
| **Lymphadenopathy** | 1.0000 ± 0.0000 | **0.6333 ± 0.0577** 🔼 | 0.7745 ± 0.0424 | **Bonus:** Recall improved significantly (0.40 -> 0.63) likely due to better attention focus. |
| **Pulm. Fibrotic Seq.** | 0.5333 ± 0.0577 | 0.5833 ± 0.1443 | 0.5556 ± 0.0962 | **Unstable:** Still the most volatile label. |
| Arterial wall calc. | 0.7222 ± 0.0481 | 1.0000 ± 0.0000 | 0.8381 ± 0.0330 | Stable. |
| Lung nodule | 0.8327 ± 0.0589 | 1.0000 ± 0.0000 | 0.9080 ± 0.0358 | Excellent performance. |
| **MACRO AVERAGE** | **0.7091 ± 0.0108** | **0.8440 ± 0.0252** | **0.7406 ± 0.0089** | **SUCCESS: Crossed 0.70 threshold.** |

### Conclusion
* **Action:** Proceed to Step 4 (Mode Check).
* **Goal:** Test if "Single Mode" (focusing on one label at a time) can fix the remaining False Positives in Lung Opacity and stabilize Fibrotic Sequela.



## Step 4: Mode Check (Single Mode)
**Date:** 2025-12-17
**Config:** `3-shot_single_v2` | **Model:** `gpt-5-nano`

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.7448 ± 0.0245
* **ROI Check:** Delta F1 (+0.0042) is far below threshold (0.05) for 5x cost -> ❌ **FAIL**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | 0.4375 ± 0.0000 | 1.0000 ± 0.0000 | 0.6087 ± 0.0000 | **No Gain:** Identical/slightly worse precision than Multi-mode. |
| **Lymphadenopathy** | 1.0000 ± 0.0000 | 0.5667 ± 0.0577 | 0.7222 ± 0.0417 | Stable. |
| **Pulm. Fibrotic Seq.** | 0.5556 ± 0.1386 | 0.5833 ± 0.1443 | 0.5608 ± 0.1116 | **Unstable:** High variance persists. |
| Arterial wall calc. | 0.7857 ± 0.0619 | 1.0000 ± 0.0000 | 0.8791 ± 0.0381 | - |
| Lung nodule | 0.9137 ± 0.0946 | 1.0000 ± 0.0000 | 0.9532 ± 0.0519 | - |
| **MACRO AVERAGE** | **0.7385 ± 0.0050** | **0.8293 ± 0.0382** | **0.7448 ± 0.0245** | **Discard:** Not worth the 5x cost. |

### Conclusion
* **Action:** Revert to Multi-Mode (`3-shot_multi_v2`). Proceed to Step 5 (Model Upgrade).
* **Goal:** Test if a smarter model (`gpt-5-mini`) can solve the remaining "intelligence" errors (specifically Lung Opacity hallucinations) that prompt engineering alone couldn't fix.


## Step 5: Model Upgrade (GPT-5 Mini)
**Date:** 2025-12-17
**Config:** `3-shot_multi_v2` | **Model:** `gpt-5-mini`

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.7505 ± 0.0245
* **ROI Check:** Delta F1 (+0.01) is far below threshold (0.05) for 5x cost -> ❌ **FAIL**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | 0.4472 ± 0.0169 | 1.0000 ± 0.0000 | 0.6179 ± 0.0160 | **Failed:** Smarter model did not fix hallucinations (Precision ~0.45). |
| **Lymphadenopathy** | 1.0000 ± 0.0000 | **0.8000 ± 0.1000** 🔼 | 0.8866 ± 0.0620 | **Improved:** Higher sensitivity, but not enough to carry the average. |
| **Pulm. Fibrotic Seq.** | 0.5238 ± 0.0412 | 0.8333 ± 0.1443 | 0.6424 ± 0.0735 | Unstable. |
| Arterial wall calc. | 0.6222 ± 0.0385 | 1.0000 ± 0.0000 | 0.7667 ± 0.0289 | **Degraded:** Surprisingly worse precision than Nano. |
| Lung nodule | 0.7222 ± 0.0000 | 1.0000 ± 0.0000 | 0.8387 ± 0.0000 | Lower precision than Nano. |
| **MACRO AVERAGE** | **0.6631 ± 0.0146** | **0.9267 ± 0.0351** | **0.7505 ± 0.0245** | **Discard:** High cost, marginal gain. |

### Conclusion
* **Action:** Revert to **Step 3 Winner** (`gpt-5-nano` + `3-shot_multi_v2`).
* **Next:** Proceed to Step 6 (Final Validation).