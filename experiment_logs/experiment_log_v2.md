# Prompt Engineering Log (tiny tuning set with N=100)

## Dataset Definitions & Lineage
* **Tiny Tuning Set (`data/tiny_tuning_set.csv`)**: 
    * **Size:** N=100 reports.
    * **Origin:** Subset of the larger Tuning Set (stratified by the 5 most common labels).
* **Tuning / Validation Set (`data/tuning_set.csv`)**: 
    * **Size:** N=615 reports.
    * **Origin:** Derived from the `test_manual_train` partition of the original CT-RATE dataset, with duplicate reports removed to ensure unique validation.
    * **Characteristics:** Representative distribution. Used for final validation (Step 6).
---

## Step 1: Baseline (Zero-Shot)
**Date:** 2026-02-02

**Goal:** Establish baseline on the new N=100 stratified tiny tuning set.

**Command:**
```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
        prompt=zero-shot_multi \
        api.model=gpt-5-nano \
        io.reports_csv=data/tiny_tuning_set.csv
```

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.7895 ± 0.0109
* **Stability Check:** 0.0109 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.5540 ± 0.0142** 🚨 | 0.9688 ± 0.0000 | 0.7047 ± 0.0115 | **Primary Failure:** Persistent hallucinations (high recall, low precision). |
| **Lymphadenopathy** | 0.9697 ± 0.0429 | **0.3793 ± 0.0488** ⚠️ | 0.5439 ± 0.0533 | **Low Recall:** Sensitivity remains weak. |
| **Pulm. Fibrotic Seq.** | 0.8473 ± 0.0139 | 0.8571 ± 0.0000 | 0.8521 ± 0.0071 | Stable / good baseline. |
| Arterial wall calc. | 0.9245 ± 0.0297 | 0.9012 ± 0.0174 | 0.9127 ± 0.0228 | Stable / strong baseline. |
| Lung nodule | 0.8767 ± 0.0080 | 1.0000 ± 0.0000 | 0.9343 ± 0.0046 | Stable / strong baseline. |
| **MACRO AVERAGE** | **0.8344 ± 0.0049** | **0.8213 ± 0.0086** | **0.7895 ± 0.0109** | |

### Conclusion
* **Action:** Proceed to Step 2 (Few-Shot).
* **Goal:** Reduce Lung Opacity false positives and improve Lymphadenopathy recall.


## Step 2: Few-Shot Check
**Date:** 2026-02-02

**Goal:** Test if few-shot examples improve the two weakest labels without hurting overall stability.

**Command:**
```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
    prompt=3-shot_multi_v1 \
    api.model=gpt-5-nano \
    io.reports_csv=data/tiny_tuning_set.csv
```

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.7677 ± 0.0069
* **Stability Check:** 0.0069 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.5113 ± 0.0142** 🚨 | 0.9584 ± 0.0148 | 0.6668 ± 0.0156 | **Worse:** Precision dropped vs zero-shot; hallucinations persist. |
| **Lymphadenopathy** | 1.0000 ± 0.0000 | **0.2988 ± 0.0162** ⚠️ | 0.4599 ± 0.0195 | **Worse:** Recall decreased further. |
| **Pulm. Fibrotic Seq.** | 0.8410 ± 0.0151 | 0.8810 ± 0.0169 | 0.8605 ± 0.0144 | Slight improvement. |
| Arterial wall calc. | 0.9477 ± 0.0174 | 0.8889 ± 0.0000 | 0.9173 ± 0.0082 | Stable / strong baseline. |
| Lung nodule | 0.8767 ± 0.0080 | 1.0000 ± 0.0000 | 0.9343 ± 0.0046 | Stable / strong baseline. |
| **MACRO AVERAGE** | **0.8354 ± 0.0038** | **0.8054 ± 0.0054** | **0.7677 ± 0.0069** | |

### Conclusion
* **Action:** Discard `3-shot_multi_v1`; proceed to Step 3 (Negative Constraints).
* **Goal:** Add explicit constraints to reduce Lung Opacity false positives and recover Lymphadenopathy recall.


## Step 3: Example Tuning (v2)
**Date:** 2026-02-02

**Goal:** Add explicit negative constraints to reduce Lung Opacity false positives and improve Lymphadenopathy recall.

**Command:**
```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
    prompt=3-shot_multi_v2 \
    api.model=gpt-5-nano \
    io.reports_csv=data/tiny_tuning_set.csv
```

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.8346 ± 0.0043
* **Stability Check:** 0.0043 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.6838 ± 0.0121** 🔼 | 0.8334 ± 0.0148 | 0.7512 ± 0.0133 | **Improved:** Precision up significantly vs v1. |
| **Lymphadenopathy** | 0.9833 ± 0.0236 | **0.6667 ± 0.0430** 🔼 | 0.7938 ± 0.0329 | **Improved:** Recall recovered strongly. |
| **Pulm. Fibrotic Seq.** | 0.8354 ± 0.0066 | 0.7262 ± 0.0337 | 0.7767 ± 0.0223 | **Tradeoff:** F1 dropped vs v1. |
| Arterial wall calc. | 0.9328 ± 0.0181 | 0.8519 ± 0.0000 | 0.8904 ± 0.0082 | Slight drop vs v1. |
| Lung nodule | 0.9307 ± 0.0087 | 0.9926 ± 0.0105 | 0.9605 ± 0.0051 | Improved. |
| **MACRO AVERAGE** | **0.8732 ± 0.0109** | **0.8141 ± 0.0008** | **0.8346 ± 0.0043** | |

### Conclusion
* **Action:** Keep `3-shot_multi_v2` as current best. Consider a v3 prompt before finalizing.
* **Goal:** If pursuing v3, target Pulm. Fibrotic Seq. recall without hurting Lung Opacity precision.


## Step 3.2: Constraint Refinement (v3)
**Date:** 2026-02-02

**Goal:** Tighten label rules (opacity/nodule/fibrotic sequela/lymphadenopathy) to reduce false positives and recover recall.

**Command:**
```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
    prompt=3-shot_multi_v3 \
    api.model=gpt-5-nano \
    io.reports_csv=data/tiny_tuning_set.csv
```

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.8934 ± 0.0078
* **Stability Check:** 0.0078 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.7048 ± 0.0176** 🔼 | 0.8438 ± 0.0000 | 0.7679 ± 0.0104 | **Improved:** Precision up vs v2. |
| **Lymphadenopathy** | 0.9563 ± 0.0147 | **1.0000 ± 0.0000** 🔼 | 0.9776 ± 0.0077 | **Major improvement:** Recall perfect on tiny set. |
| **Pulm. Fibrotic Seq.** | 0.8609 ± 0.0169 | 0.8095 ± 0.0168 | 0.8344 ± 0.0152 | **Recovered:** F1 up vs v2. |
| Arterial wall calc. | 0.9230 ± 0.0024 | 0.8889 ± 0.0302 | 0.9054 ± 0.0169 | Slight improvement. |
| Lung nodule | 0.9712 ± 0.0097 | 0.9926 ± 0.0105 | 0.9817 ± 0.0052 | Improved. |
| **MACRO AVERAGE** | **0.8832 ± 0.0073** | **0.9070 ± 0.0082** | **0.8934 ± 0.0078** | |

### Conclusion
* **Action:** Promote `3-shot_multi_v3` as current best.
* **Goal:** Proceed to model upgrade checks (Step 5) before final validation.


## Step 4: Mode Check (Single vs Multi)
**Date:** 2026-02-02

**Goal:** Verify whether single-label mode improves performance vs multi-label mode.

**Command:**
```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
    prompt=3-shot_multi_v3 \
    api.model=gpt-5-nano \
    io.reports_csv=data/tiny_tuning_set.csv \
    prompt.mode=single
```

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.8729 ± 0.0089
* **Stability Check:** 0.0089 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.6838 ± 0.0071** | 0.8334 ± 0.0148 | 0.7512 ± 0.0073 | Slightly worse than multi. |
| **Lymphadenopathy** | 0.9459 ± 0.0147 | **1.0000 ± 0.0000** | 0.9722 ± 0.0077 | Slightly worse than multi. |
| **Pulm. Fibrotic Seq.** | 0.8233 ± 0.0060 | 0.8333 ± 0.0337 | 0.8281 ± 0.0198 | Slightly worse than multi. |
| Arterial wall calc. | 0.8760 ± 0.0298 | 0.7778 ± 0.0000 | 0.8237 ± 0.0132 | Worse than multi. |
| Lung nodule | 0.9783 ± 0.0000 | 1.0000 ± 0.0000 | 0.9890 ± 0.0000 | Slightly better than multi. |
| **MACRO AVERAGE** | **0.8615 ± 0.0105** | **0.8889 ± 0.0097** | **0.8729 ± 0.0089** | |

### Conclusion
* **Action:** Stick with multi-label mode (`3-shot_multi_v3`).
* **Goal:** Proceed to model upgrade checks (Step 5) before final validation.


## Step 5.1: Model Upgrade Check (GPT-5 Mini)
**Date:** 2026-02-02

**Goal:** Test if a higher-capacity model improves performance enough to justify cost.

**Command:**
```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
    prompt=3-shot_multi_v3 \
    api.model=gpt-5-mini \
    io.reports_csv=data/tiny_tuning_set.csv
```

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.9123 ± 0.0029
* **Stability Check:** 0.0029 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.6869 ± 0.0038** | 0.8229 ± 0.0148 | 0.7488 ± 0.0083 | Slightly worse than nano v3. |
| **Lymphadenopathy** | 0.9667 ± 0.0000 | **1.0000 ± 0.0000** | 0.9831 ± 0.0000 | Slightly better than nano v3. |
| **Pulm. Fibrotic Seq.** | 0.8404 ± 0.0024 | 0.9405 ± 0.0168 | 0.8876 ± 0.0088 | Improved vs nano v3. |
| Arterial wall calc. | 0.9103 ± 0.0146 | 1.0000 ± 0.0000 | 0.9530 ± 0.0080 | Improved vs nano v3. |
| Lung nodule | 0.9783 ± 0.0000 | 1.0000 ± 0.0000 | 0.9890 ± 0.0000 | Slightly better than nano v3. |
| **MACRO AVERAGE** | **0.8765 ± 0.0031** | **0.9527 ± 0.0032** | **0.9123 ± 0.0029** | |

### Conclusion
* **Action:** Not worth the ~5x cost for a modest +0.0189 macro F1 gain vs nano v3.
* **Goal:** Stick with `gpt-5-nano` for final validation.


## Step 5.2: Model Upgrade Check (GPT-5.1)
**Date:** 2026-02-02

**Goal:** Test if a much higher-capacity model improves performance enough to justify cost.

**Command:**
```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
    prompt=3-shot_multi_v3 \
    api.model=gpt-5.1 \
    io.reports_csv=data/tiny_tuning_set.csv
```

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.9180 ± 0.0017
* **Stability Check:** 0.0017 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.6896 ± 0.0038** | 0.8334 ± 0.0148 | 0.7547 ± 0.0083 | Slightly worse than nano v3. |
| **Lymphadenopathy** | 0.9667 ± 0.0000 | **1.0000 ± 0.0000** | 0.9831 ± 0.0000 | Slightly better than nano v3. |
| **Pulm. Fibrotic Seq.** | 0.8387 ± 0.0000 | 0.9286 ± 0.0000 | 0.8814 ± 0.0000 | Improved vs nano v3. |
| Arterial wall calc. | 0.9643 ± 0.0000 | 1.0000 ± 0.0000 | 0.9818 ± 0.0000 | Improved vs nano v3. |
| Lung nodule | 0.9783 ± 0.0000 | 1.0000 ± 0.0000 | 0.9890 ± 0.0000 | Slightly better than nano v3. |
| **MACRO AVERAGE** | **0.8875 ± 0.0008** | **0.9524 ± 0.0030** | **0.9180 ± 0.0017** | |

### Conclusion
* **Action:** Not worth the ~25x cost for a modest +0.0246 macro F1 gain vs nano v3.
* **Goal:** Stick with `gpt-5-nano` for final validation.


## Step 6: Final Validation (Full Tuning Set)
**Date:** 2026-02-03

**Goal:** Validate the chosen config on the full tuning set (N=615).

**Command:**
```bash
python scripts/evaluate_prompt.py --config-name prompt_engineering \
    prompt=3-shot_multi_v3 \
    api.model=gpt-5-nano \
    io.reports_csv=data/tuning_set.csv
```

### Aggregate Results (3 Runs)
* **Mean Macro F1:** 0.8889 ± 0.0017
* **Stability Check:** 0.0017 (Target < 0.02) -> ✅ **PASS**

### Per-Label Detailed Metrics (Mean ± Std Dev)
| Label | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lung Opacity** | **0.7471 ± 0.0089** | 0.8618 ± 0.0025 | 0.8004 ± 0.0061 | Improved vs tiny set. |
| **Lymphadenopathy** | 0.9642 ± 0.0025 | **0.9694 ± 0.0051** | 0.9668 ± 0.0034 | Stable / strong. |
| **Pulm. Fibrotic Seq.** | 0.8986 ± 0.0026 | 0.7889 ± 0.0227 | 0.8400 ± 0.0140 | Good, with recall as main limiter. |
| Arterial wall calc. | 0.8462 ± 0.0049 | 0.8953 ± 0.0082 | 0.8701 ± 0.0063 | Slightly lower than tiny set. |
| Lung nodule | 0.9528 ± 0.0001 | 0.9825 ± 0.0017 | 0.9674 ± 0.0009 | Strong / stable. |
| **MACRO AVERAGE** | **0.8817 ± 0.0013** | **0.8996 ± 0.0042** | **0.8889 ± 0.0017** | |

### Conclusion
* **Action:** Finalize `gpt-5-nano` + `3-shot_multi_v3`.
* **Goal:** Ship / productionize with this configuration.