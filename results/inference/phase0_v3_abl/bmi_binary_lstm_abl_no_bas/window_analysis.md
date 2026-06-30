# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 64.9 | 68.3 | 64.1 | 62.2 | 64.9 | 68.3 | 64.1 | 62.2 | 64.9 | 68.3 | 64.1 | 62.2 |
| 5 | 1856 | 9280 | 65.8 | 68.6 | 63.3 | 62.2 | 68.7 | 72.0 | 65.5 | 64.7 | 69.1 | 72.0 | 66.3 | 65.3 |
| 10 | 1856 | 18514 | 66.0 | 68.8 | 63.3 | 62.3 | 68.9 | 72.0 | 66.0 | 65.1 | 69.9 | 72.0 | 65.6 | 65.3 |
| 20 | 1856 | 25227 | 66.5 | 68.8 | 63.2 | 62.3 | 69.0 | 72.1 | 65.6 | 64.9 | 69.1 | 72.1 | 65.4 | 64.8 |
| all | 1856 | 25261 | 66.5 | 68.8 | 63.1 | 62.3 | 68.9 | 72.1 | 65.5 | 64.8 | 69.1 | 72.1 | 65.4 | 64.8 |
