# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 70.7 | 70.6 | 66.0 | 65.8 | 70.7 | 70.6 | 66.0 | 65.8 | 70.7 | 70.6 | 66.0 | 65.8 |
| 5 | 1856 | 9280 | 69.9 | 71.8 | 65.7 | 65.3 | 73.2 | 75.6 | 67.8 | 68.0 | 73.5 | 75.6 | 67.9 | 68.2 |
| 10 | 1856 | 18514 | 70.0 | 72.3 | 66.1 | 65.6 | 73.0 | 76.0 | 68.3 | 68.2 | 73.1 | 76.0 | 67.2 | 67.5 |
| 20 | 1856 | 25227 | 70.4 | 72.2 | 66.0 | 65.6 | 73.2 | 75.9 | 68.5 | 68.5 | 73.6 | 75.9 | 68.3 | 68.5 |
| all | 1856 | 25261 | 70.4 | 72.2 | 66.0 | 65.6 | 73.2 | 75.9 | 68.5 | 68.5 | 73.6 | 75.9 | 68.3 | 68.5 |
