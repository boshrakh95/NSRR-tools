# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 63.1 | 70.1 | 64.3 | 61.2 | 63.1 | 70.1 | 64.3 | 61.2 | 63.1 | 70.1 | 64.3 | 61.2 |
| 5 | 1856 | 9280 | 63.4 | 71.6 | 65.2 | 61.8 | 67.7 | 76.1 | 69.5 | 66.0 | 67.6 | 76.1 | 69.2 | 65.8 |
| 10 | 1856 | 18514 | 63.8 | 72.3 | 65.8 | 62.2 | 67.8 | 76.8 | 69.5 | 66.1 | 69.2 | 76.8 | 69.7 | 67.0 |
| 20 | 1856 | 25227 | 63.7 | 72.0 | 65.4 | 61.9 | 67.7 | 76.6 | 69.3 | 65.9 | 67.8 | 76.6 | 68.9 | 65.9 |
| all | 1856 | 25261 | 63.7 | 71.9 | 65.3 | 61.9 | 67.7 | 76.6 | 69.3 | 65.9 | 67.8 | 76.6 | 68.9 | 65.9 |
