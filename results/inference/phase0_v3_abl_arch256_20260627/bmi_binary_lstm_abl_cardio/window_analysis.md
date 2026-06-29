# Window-count analysis: `bmi_binary` · `lstm`

_Window selection: **evenly-spaced**. Metrics in %._


---
# Split: TEST

_Window selection: **evenly-spaced**. Metrics in %. MP = mean-prob aggregation. MV = majority-vote._

## Context: `40m`

| K | N-subj | N-seg | Seg-Acc | Seg-AUROC | Seg-BalAcc | Seg-F1 | MP-Acc | MP-AUROC | MP-BalAcc | MP-F1 | MV-Acc | MV-AUROC | MV-BalAcc | MV-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1856 | 1856 | 64.0 | 65.5 | 60.4 | 59.7 | 64.0 | 65.5 | 60.4 | 59.7 | 64.0 | 65.5 | 60.4 | 59.7 |
| 5 | 1856 | 9280 | 63.8 | 64.6 | 60.1 | 59.4 | 65.4 | 66.2 | 61.0 | 60.5 | 65.4 | 66.2 | 60.9 | 60.5 |
| 10 | 1856 | 18514 | 64.3 | 64.8 | 60.4 | 59.8 | 66.2 | 66.3 | 61.6 | 61.2 | 66.6 | 66.3 | 61.2 | 61.1 |
| 20 | 1856 | 25227 | 64.7 | 64.5 | 60.2 | 59.7 | 66.1 | 66.3 | 61.3 | 61.0 | 66.3 | 66.3 | 61.5 | 61.2 |
| all | 1856 | 25261 | 64.7 | 64.5 | 60.1 | 59.7 | 66.1 | 66.3 | 61.3 | 61.0 | 66.3 | 66.3 | 61.5 | 61.2 |
