| Task                 | Head        | N_test | Best_L@K=5 | AUROC@K=5 | Best_L@K=all | AUROC@K=all |
| -------------------- | ----------- | ------ | ---------- | --------- | ------------ | ----------- |
| Sex                  | lstm        | 1430   | 40m        | 0.894     | 40m          | 0.906       |
| Sex                  | transformer | 1430   | 120m       | 0.929     | 120m         | 0.929       |
| Sex                  | mean_pool   | 1430   | 120m       | 0.829     | 240m         | 0.834       |
| Age group            | lstm        | 1859   | 80m        | 0.901     | 80m          | 0.901       |
| Age group            | transformer | 1859   | 120m       | 0.908     | 240m         | 0.911       |
| Age group            | mean_pool   | 1859   | 120m       | 0.847     | 240m         | 0.851       |
| BMI (obese)          | lstm        | 1856   | 80m        | 0.801     | 240m         | 0.802       |
| BMI (obese)          | transformer | 1856   | 120m       | 0.812     | 240m         | 0.816       |
| BMI (obese)          | mean_pool   | 1856   | 80m        | 0.775     | 240m         | 0.778       |
| Sleep efficiency     | lstm        | 2020   | 240m       | 0.810     | 240m         | 0.810       |
| Sleep efficiency     | transformer | 2020   | 240m       | 0.825     | 240m         | 0.825       |
| Sleep efficiency     | mean_pool   | 2020   | 240m       | 0.757     | 240m         | 0.757       |
| Sleep apnea (AHI≥15) | lstm        | 2054   | 120m       | 0.874     | 120m         | 0.874       |
| Sleep apnea (AHI≥15) | transformer | 2054   | 120m       | 0.900     | 240m         | 0.901       |
| Sleep apnea (AHI≥15) | mean_pool   | 2054   | 120m       | 0.818     | 240m         | 0.821       |
| Depression (extreme) | lstm        | 229    | 30s        | 0.755     | 120m         | 0.752       |
| Depression (extreme) | transformer | 229    | 80m        | 0.756     | 80m          | 0.756       |
| Depression (extreme) | mean_pool   | 229    | 30s        | 0.729     | 240m         | 0.744       |
| OSA (APPLES)         | lstm        | 161    | 80m        | 0.752     | 30s          | 0.772       |
| OSA (APPLES)         | transformer | 161    | 120m       | 0.814     | 240m         | 0.818       |
| OSA (APPLES)         | mean_pool   | 161    | 120m       | 0.798     | 240m         | 0.818       |