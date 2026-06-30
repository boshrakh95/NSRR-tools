# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 61.8 | 69.1 | 63.9 | 60.3 | 61.8 | 69.1 | 63.9 | 60.3 | 61.8 | 69.1 | 63.9 | 60.3 |
| 5 | 1856 | 9280 | 66.1 | 70.1 | 64.7 | 63.1 | 71.3 | 74.4 | 68.7 | 67.7 | 70.2 | 74.4 | 67.6 | 66.6 |
| 10 | 1856 | 18514 | 67.1 | 70.9 | 65.2 | 63.8 | 71.1 | 75.4 | 68.1 | 67.3 | 71.7 | 75.4 | 67.5 | 67.2 |
| 20 | 1856 | 25227 | 67.5 | 70.6 | 65.0 | 63.7 | 71.3 | 75.1 | 67.8 | 67.2 | 71.3 | 75.1 | 67.4 | 67.0 |
| all | 1856 | 25261 | 67.4 | 70.6 | 65.0 | 63.7 | 71.3 | 75.1 | 67.8 | 67.2 | 71.2 | 75.1 | 67.4 | 66.9 |
