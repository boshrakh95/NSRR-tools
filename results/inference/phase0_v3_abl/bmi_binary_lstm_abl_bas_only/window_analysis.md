# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 63.6 | 69.8 | 64.1 | 61.5 | 63.6 | 69.8 | 64.1 | 61.5 | 63.6 | 69.8 | 64.1 | 61.5 |
| 5 | 1856 | 9280 | 66.2 | 69.4 | 64.3 | 62.9 | 72.2 | 73.4 | 68.0 | 67.7 | 71.5 | 73.4 | 67.7 | 67.2 |
| 10 | 1856 | 18514 | 67.5 | 70.0 | 65.1 | 63.9 | 72.9 | 74.2 | 68.3 | 68.2 | 72.8 | 74.2 | 67.4 | 67.6 |
| 20 | 1856 | 25227 | 67.8 | 69.7 | 64.7 | 63.7 | 73.0 | 74.1 | 68.4 | 68.3 | 72.4 | 74.1 | 67.7 | 67.6 |
| all | 1856 | 25261 | 67.8 | 69.7 | 64.7 | 63.7 | 73.0 | 74.2 | 68.4 | 68.3 | 72.4 | 74.2 | 67.7 | 67.6 |
