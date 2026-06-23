# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 69.3 | 68.7 | 61.3 | 61.7 | 69.3 | 68.7 | 61.3 | 61.7 | 69.3 | 68.7 | 61.3 | 61.7 |
| 5 | 1856 | 9280 | 70.1 | 69.2 | 61.8 | 62.3 | 72.3 | 72.5 | 62.5 | 63.3 | 72.2 | 72.5 | 62.9 | 63.7 |
| 10 | 1856 | 18514 | 70.4 | 69.4 | 62.1 | 62.6 | 72.7 | 72.7 | 63.1 | 64.0 | 72.6 | 72.7 | 62.5 | 63.3 |
| 20 | 1856 | 25227 | 70.8 | 69.4 | 61.8 | 62.4 | 72.6 | 72.8 | 62.7 | 63.6 | 72.6 | 72.8 | 62.6 | 63.5 |
| all | 1856 | 25261 | 70.8 | 69.4 | 61.8 | 62.4 | 72.5 | 72.8 | 62.7 | 63.5 | 72.6 | 72.8 | 62.6 | 63.5 |
