| Task                 | Head        | N_test | Best_L@K=5 | AUROC@K=5 | Best_L@K=all | AUROC@K=all |
| -------------------- | ----------- | ------ | ---------- | --------- | ------------ | ----------- |
| Sex                  | lstm        | 1430   | 120m       | 0.872     | 120m         | 0.872       |
| Sex                  | transformer | 1430   | 120m       | 0.905     | 240m         | 0.910       |
| Sex                  | mean_pool   | 1430   | 120m       | 0.815     | 240m         | 0.818       |
| Age group            | lstm        | 1859   | 120m       | 0.893     | 120m         | 0.893       |
| Age group            | transformer | 1859   | 120m       | 0.902     | 240m         | 0.905       |
| Age group            | mean_pool   | 1859   | 120m       | 0.846     | 240m         | 0.850       |
| BMI (obese)          | lstm        | 1856   | 80m        | 0.763     | 80m          | 0.767       |
| BMI (obese)          | transformer | 1856   | 80m        | 0.767     | 240m         | 0.777       |
| BMI (obese)          | mean_pool   | 1856   | 120m       | 0.744     | 240m         | 0.746       |
| Sleep efficiency     | lstm        | 2023   | 240m       | 0.788     | 240m         | 0.788       |
| Sleep efficiency     | transformer | 2023   | 240m       | 0.831     | 240m         | 0.831       |
| Sleep efficiency     | mean_pool   | 2020   | 240m       | 0.760     | 240m         | 0.760       |
| Sleep apnea (AHI≥15) | lstm        | 2054   | 120m       | 0.831     | 120m         | 0.832       |
| Sleep apnea (AHI≥15) | transformer | 2054   | 120m       | 0.856     | 120m         | 0.857       |
| Sleep apnea (AHI≥15) | mean_pool   | 2054   | 120m       | 0.764     | 240m         | 0.765       |
| Depression (extreme) | lstm        | 229    | 10m        | 0.776     | 10m          | 0.770       |
| Depression (extreme) | transformer | 229    | 30s        | 0.776     | 30s          | 0.756       |
| Depression (extreme) | mean_pool   | 229    | 120m       | 0.765     | 120m         | 0.765       |
| OSA (APPLES)         | lstm        | 161    | 10m        | 0.823     | 40m          | 0.834       |
| OSA (APPLES)         | transformer | 161    | 80m        | 0.892     | 80m          | 0.888       |
| OSA (APPLES)         | mean_pool   | 161    | 80m        | 0.845     | 240m         | 0.848       |