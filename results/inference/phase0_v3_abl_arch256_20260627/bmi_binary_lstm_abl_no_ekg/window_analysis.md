# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 70.6 | 69.8 | 63.4 | 63.8 | 70.6 | 69.8 | 63.4 | 63.8 | 70.6 | 69.8 | 63.4 | 63.8 |
| 5 | 1856 | 9280 | 71.9 | 71.4 | 65.5 | 65.9 | 73.8 | 75.2 | 66.4 | 67.2 | 74.7 | 75.2 | 67.4 | 68.3 |
| 10 | 1856 | 18514 | 71.9 | 71.9 | 65.6 | 66.0 | 75.2 | 75.8 | 67.9 | 68.9 | 74.9 | 75.8 | 66.9 | 68.0 |
| 20 | 1856 | 25227 | 72.3 | 71.9 | 65.4 | 65.9 | 75.2 | 75.8 | 68.0 | 68.9 | 75.0 | 75.8 | 67.5 | 68.4 |
| all | 1856 | 25261 | 72.3 | 71.9 | 65.4 | 65.9 | 75.2 | 75.8 | 68.0 | 68.9 | 75.0 | 75.8 | 67.5 | 68.4 |
