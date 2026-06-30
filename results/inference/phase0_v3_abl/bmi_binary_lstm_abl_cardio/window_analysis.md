# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 67.0 | 65.7 | 60.8 | 60.9 | 67.0 | 65.7 | 60.8 | 60.9 | 67.0 | 65.7 | 60.8 | 60.9 |
| 5 | 1856 | 9280 | 66.8 | 65.0 | 60.5 | 60.6 | 69.2 | 67.5 | 61.3 | 61.7 | 69.3 | 67.5 | 61.4 | 61.8 |
| 10 | 1856 | 18514 | 67.0 | 64.9 | 60.4 | 60.5 | 69.6 | 67.3 | 61.7 | 62.2 | 69.6 | 67.3 | 60.5 | 61.0 |
| 20 | 1856 | 25227 | 67.4 | 64.7 | 60.1 | 60.3 | 69.8 | 67.5 | 61.7 | 62.2 | 69.5 | 67.5 | 61.2 | 61.7 |
| all | 1856 | 25261 | 67.4 | 64.7 | 60.1 | 60.3 | 69.9 | 67.5 | 61.7 | 62.2 | 69.4 | 67.5 | 61.1 | 61.6 |
