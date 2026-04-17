# Phase 0 Results

_All metrics in %. Columns grouped as Train / Val / Test. Best context per head highlighted with ← (by test AUROC)._

---
## sleep_staging  `seq2seq` · 5 classes

### Head: `lstm`

| Context | N-train | N-val | N-test | V-AUROC | V-Bal-Acc | V-Macro-F1 | V-Kappa | V-Wake | V-N1 | V-N2 | V-N3 | V-REM | T-AUROC | T-Bal-Acc | T-Macro-F1 | T-Kappa | T-Wake | T-N1 | T-N2 | T-N3 | T-REM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30s | 755,288 | 159,700 | 161,809 | 92.2 | 72.7 | 62.4 | 57.1 | 82.2 | 57.1 | 55.3 | 78.8 | 90.1 | 92.8 | 73.9 | 61.2 | 58.0 | 82.5 | 59.2 | 57.3 | 82.0 | 88.6 |
| 10m ← | 753,746 | 159,370 | 161,477 | 93.4 | 73.4 | 65.9 | 61.4 | 75.5 | 67.0 | 65.0 | 69.6 | 89.9 | 93.9 | 75.1 | 64.6 | 62.0 | 76.6 | 67.0 | 66.1 | 77.9 | 87.9 |
| 40m | 748,349 | 158,215 | 160,315 | 92.9 | 70.8 | 65.2 | 61.2 | 84.2 | 61.9 | 62.5 | 54.3 | 91.0 | 93.6 | 73.6 | 65.0 | 62.8 | 84.5 | 62.7 | 65.2 | 66.6 | 89.2 |

---
## apnea_binary  `seq2label` · 2 classes

### Head: `lstm`

| Context | N-train | N-val | N-test | V-AUROC | V-Bal-Acc | V-Macro-F1 | V-Rec-0 | V-Rec-1 | T-AUROC | T-Bal-Acc | T-Macro-F1 | T-Rec-0 | T-Rec-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30s | 47,960 | 10,275 | 10,285 | 69.4 | 63.2 | 63.1 | 59.4 | 66.9 | 67.5 | 61.3 | 61.3 | 58.2 | 64.4 |
| 30s | 10 | 10 | 10 | 56.0 | 50.0 | 33.3 | 100.0 | 0.0 | 60.0 | 50.0 | 33.3 | 100.0 | 0.0 |
| 30s | 10 | 10 | 10 | 52.0 | 50.0 | 33.3 | 100.0 | 0.0 | 48.0 | 50.0 | 33.3 | 100.0 | 0.0 |
| 10m | 47,941 | 10,275 | 10,278 | 71.2 | 64.7 | 64.5 | 73.1 | 56.3 | 69.5 | 63.2 | 62.9 | 72.3 | 54.2 |
| 40m ← | 47,918 | 10,275 | 10,275 | 74.4 | 67.1 | 67.1 | 71.7 | 62.6 | 73.4 | 66.3 | 66.2 | 70.2 | 62.3 |

---
## sleepiness_binary  `seq2label` · 2 classes

### Head: `lstm`

| Context | N-train | N-val | N-test | V-AUROC | V-Bal-Acc | V-Macro-F1 | V-Rec-0 | V-Rec-1 | T-AUROC | T-Bal-Acc | T-Macro-F1 | T-Rec-0 | T-Rec-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30s | 46,180 | 9,895 | 9,900 | 62.8 | 57.1 | 57.3 | 78.9 | 35.2 | 59.8 | 56.5 | 56.6 | 78.2 | 34.7 |
| 10m ← | 46,163 | 9,895 | 9,893 | 63.3 | 58.5 | 57.3 | 66.9 | 50.0 | 60.7 | 57.5 | 56.4 | 66.0 | 49.0 |
| 40m | 46,142 | 9,895 | 9,890 | 63.4 | 58.4 | 58.3 | 74.2 | 42.7 | 60.4 | 57.4 | 57.2 | 73.1 | 41.7 |

---
## cvd_binary  `seq2label` · 2 classes

### Head: `lstm`

| Context | N-train | N-val | N-test | V-AUROC | V-Bal-Acc | V-Macro-F1 | V-Rec-0 | V-Rec-1 | T-AUROC | T-Bal-Acc | T-Macro-F1 | T-Rec-0 | T-Rec-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30s | 40,620 | 8,705 | 8,710 | 63.7 | 60.0 | 57.5 | 65.1 | 55.0 | 64.2 | 59.8 | 57.6 | 66.2 | 53.5 |
| 10m | 40,620 | 8,705 | 8,710 | 64.9 | 59.6 | 58.4 | 71.7 | 47.5 | 64.7 | 59.8 | 59.1 | 74.4 | 45.2 |
| 40m ← | 40,620 | 8,705 | 8,710 | 65.9 | 61.6 | 58.8 | 66.0 | 57.2 | 65.5 | 60.8 | 58.6 | 67.3 | 54.3 |

### Head: `transformer`

| Context | N-train | N-val | N-test | V-AUROC | V-Bal-Acc | V-Macro-F1 | V-Rec-0 | V-Rec-1 | T-AUROC | T-Bal-Acc | T-Macro-F1 | T-Rec-0 | T-Rec-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40m ← | 40,620 | 8,705 | 8,710 | 65.5 | 61.4 | 58.1 | 63.7 | 59.1 | 65.5 | 61.0 | 58.3 | 65.4 | 56.6 |

---
## rested_morning  `seq2label` · 2 classes

### Head: `lstm`

| Context | N-train | N-val | N-test | V-AUROC | V-Bal-Acc | V-Macro-F1 | V-Rec-0 | V-Rec-1 | T-AUROC | T-Bal-Acc | T-Macro-F1 | T-Rec-0 | T-Rec-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30s | 13,755 | 2,945 | 2,950 | 55.7 | 53.2 | 50.8 | 36.4 | 70.0 | 53.1 | 52.7 | 50.2 | 36.3 | 69.1 |
| 10m ← | 13,755 | 2,945 | 2,950 | 55.2 | 52.9 | 49.8 | 33.2 | 72.6 | 54.0 | 53.1 | 49.6 | 32.5 | 73.7 |
| 40m | 13,755 | 2,945 | 2,950 | 53.7 | 52.3 | 51.0 | 41.5 | 63.1 | 51.9 | 52.1 | 50.7 | 41.5 | 62.7 |


---
# Subject-level Results (all-window aggregation)

_Segment-level metrics use K=5 windows (training eval). Subject-level metrics run inference on ALL available windows, then aggregate per subject._
_mean-prob: argmax of averaged softmax probabilities. maj-vote: mode of per-window hard predictions. AUROC always from mean-prob._

---
## apnea_binary

### Head: `lstm`

| Context | Method | N-subj | Avg-wins | AUROC | Bal-Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 10m | majority-vote | 1,863 | 56 | 76.1 | 67.4 | 67.0 |
| 10m | mean-prob | 1,863 | 56 | 76.1 | 68.0 | 67.7 |
