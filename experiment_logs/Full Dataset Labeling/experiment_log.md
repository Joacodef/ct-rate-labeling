# Full Dataset Labeling Log

## Context
**Timing:** After Prompt Engineering V2 (see prompt engineering log for configuration selection).
**Model:** gpt-5-nano
**Task:** Full dataset labeling for CT-RATE (multi-label, 5 shared pathologies).
**Total reports labeled:** 25,692 (real amount of available scans).

## Procedure Summary
1. Labeled the full dataset with gpt-5-nano.
2. Compared GPT labels against manual gold standard labels.
3. Compared GPT labels against RadBERT labels provided by CT-CLIP authors.

## Comparison A: GPT vs Manual Gold Standard
**Metrics file:** outputs/Comparisons/compare_all_manual_vs_gpt-5-nano/evaluation_metrics.csv

### Per-label metrics
| Label | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arterial wall calcification | 0.9000 | 0.8465 | 0.8724 | 0.9274 | 171 | 19 | 31 | 468 |
| Lung nodule | 0.9650 | 0.9550 | 0.9600 | 0.9666 | 276 | 10 | 13 | 390 |
| Lung opacity | 0.8670 | 0.7621 | 0.8112 | 0.8723 | 189 | 29 | 59 | 412 |
| Lymphadenopathy | 0.9602 | 0.9461 | 0.9531 | 0.9724 | 193 | 8 | 11 | 477 |
| Pulmonary fibrotic sequela | 0.7732 | 0.8982 | 0.8310 | 0.9115 | 150 | 44 | 17 | 478 |

### Macro / Micro
| Average | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MACRO_AVERAGE | 0.8931 | 0.8816 | 0.8855 | 0.9300 | 979 | 110 | 131 | 2225 |
| MICRO_AVERAGE | 0.8990 | 0.8820 | 0.8904 | 0.9300 | 979 | 110 | 131 | 2225 |

## Comparison B: GPT vs RadBERT (CT-CLIP labels)
**Metrics file:** outputs/Comparisons/compare_radbert_vs_gpt-5-nano/evaluation_metrics.csv

### Per-label metrics
| Label | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arterial wall calcification | 0.8974 | 0.8734 | 0.8853 | 0.9354 | 6404 | 732 | 928 | 17628 |
| Lung nodule | 0.9586 | 0.9787 | 0.9686 | 0.9717 | 11195 | 483 | 244 | 13770 |
| Lung opacity | 0.9076 | 0.8463 | 0.8759 | 0.9054 | 8579 | 873 | 1558 | 14682 |
| Lymphadenopathy | 0.9708 | 0.9670 | 0.9689 | 0.9842 | 6323 | 190 | 216 | 18963 |
| Pulmonary fibrotic sequela | 0.7789 | 0.9237 | 0.8451 | 0.9236 | 5354 | 1520 | 442 | 18376 |

### Macro / Micro
| Average | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MACRO_AVERAGE | 0.9027 | 0.9178 | 0.9088 | 0.9441 | 37855 | 3798 | 3388 | 83419 |
| MICRO_AVERAGE | 0.9088 | 0.9179 | 0.9133 | 0.9441 | 37855 | 3798 | 3388 | 83419 |

## Conclusion
- Full-dataset baseline established for gpt-5-nano after Prompt Engineering V2.
- Against manual gold standard, overall performance is strong with macro $F_1 = 0.8855$ and micro $F_1 = 0.8904$.
- Against RadBERT labels, agreement is high with macro $F_1 = 0.9088$ and micro $F_1 = 0.9133$.
- These results define the reference baseline for future prompt/model iterations.