# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 68.6 | 69.2 | 64.9 | 64.3 | 68.6 | 69.2 | 64.9 | 64.3 | 68.6 | 69.2 | 64.9 | 64.3 |
| 5 | 1856 | 9280 | 68.0 | 69.9 | 65.2 | 64.2 | 72.7 | 73.6 | 68.7 | 68.4 | 71.9 | 73.6 | 67.9 | 67.5 |
| 10 | 1856 | 18514 | 68.4 | 70.3 | 65.4 | 64.5 | 72.6 | 74.1 | 68.4 | 68.2 | 73.2 | 74.1 | 68.1 | 68.2 |
| 20 | 1856 | 25227 | 68.6 | 70.2 | 65.2 | 64.4 | 72.7 | 74.1 | 68.4 | 68.2 | 72.5 | 74.1 | 68.1 | 67.9 |
| all | 1856 | 25261 | 68.6 | 70.2 | 65.2 | 64.4 | 72.7 | 74.1 | 68.4 | 68.2 | 72.5 | 74.1 | 68.1 | 67.9 |
