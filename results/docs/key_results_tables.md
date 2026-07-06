# Key Results Tables

## Stage summary

| stage | events | mother types | interpretation |
| --- | --- | --- | --- |
| 4-8 | 888 | 4 | early local-neighborhood stage |
| 8-12 | 872 | 4 | mixed local-global stage |
| 12-14 | 436 | 2 | neighbor-shape moment stage |
| 14-15 | 186 | 1 | short transition window |
| 15-24 | 1674 | 9 | late broad-structure dictionary stage |


## Main FixedTermOLS performance by stage

| stage | all-target mean R2 | mean-target mean R2 | half-target mean R2 | total selected terms |
| --- | --- | --- | --- | --- |
| 4-8 | 0.883 | 0.981 | 0.784 | 34 |
| 8-12 | 0.896 | 0.986 | 0.806 | 43 |
| 12-14 | 0.546 | 0.936 | 0.156 | 36 |
| 14-15 | 0.170 | 0.248 | 0.092 | 34 |
| 15-24 | 0.853 | 0.980 | 0.727 | 45 |


## Half-vector structural block share

| stage | stage | mother_coupling | mother_neighbor_moment | mother_polynomial | neighbor_moment | relative_spread |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 4-8 | 0.168 | 0.143 | 0.130 | 0.364 | 0.195 |
| 1 | 8-12 | 0.257 | 0.141 | 0.282 | 0.171 | 0.149 |
| 2 | 12-14 | 0.101 | 0.015 | 0.175 | 0.550 | 0.159 |
| 3 | 14-15 | 0.584 | 0.000 | 0.280 | 0.076 | 0.060 |
| 4 | 15-24 | 0.426 | 0.182 | 0.193 | 0.157 | 0.042 |


## Qy to x_half validation

| stage | n | Pearson r | Spearman r | univariate R2 | partial r | FixedTerm coef |
| --- | --- | --- | --- | --- | --- | --- |
| 4-8 | 888 | -0.094 | 0.051 | 0.009 | 0.151 | 0.109 |
| 8-12 | 872 | 0.405 | 0.288 | 0.164 | 0.124 | 0.120 |
| 12-14 | 436 | 0.437 | 0.399 | 0.191 | -0.122 | 0.031 |
| 14-15 | 186 | 0.148 | 0.147 | 0.022 | -0.121 | — |
| 15-24 | 1674 | 0.581 | 0.617 | 0.338 | 0.515 | 0.029 |


## Cross-stage transfer summary

| mode | mean_mean_R2 | median_mean_R2 | mean_half_R2 | median_half_R2 | mean_all_R2 | median_all_R2 |
| --- | --- | --- | --- | --- | --- | --- |
| direct coefficients | -7.263 | -1.117 | -137.451 | -8.638 | -72.357 | -8.118 |
| feature transfer + destination refit | 0.784 | 0.935 | 0.244 | 0.201 | 0.514 | 0.571 |


## RF-FixedTerm overlap

| stage | fixedterm_selected_count | rf_top20_count | overlap_count | overlap_fraction_selected |
| --- | --- | --- | --- | --- |
| 4-8 | 21 | 20 | 7 | 0.333 |
| 8-12 | 30 | 20 | 12 | 0.400 |
| 12-14 | 26 | 20 | 11 | 0.423 |
| 14-15 | 22 | 20 | 13 | 0.591 |
| 15-24 | 28 | 20 | 8 | 0.286 |


## Few-shot global ranking, n=20

| model | all R2 | mean R2 | half R2 | position RMSE | radius balance |
| --- | --- | --- | --- | --- | --- |
| RF quick | 0.458 | 0.627 | 0.288 | 4.202 | 0.461 |
| Prev-stage FixedTerm | 0.361 | 0.706 | 0.016 | 2.485 | 0.797 |
| Same-stage FixedTerm | 0.339 | 0.732 | -0.053 | 2.224 | 0.346 |
| 15-24 FixedTerm | 0.323 | 0.641 | 0.004 | 2.194 | 0.702 |
| Union FixedTerm | 0.108 | 0.649 | -0.433 | 2.521 | 0.397 |
| Full ridge | -0.082 | 0.373 | -0.536 | 3.014 | 0.515 |
| Mother copy | -0.127 | -0.187 | -0.067 | 3.328 | 0.000 |
| Full linear | -3724.407 | -316.961 | -7131.852 | 54.010 | -53.363 |


## Few-shot global ranking, n=40

| model | all R2 | mean R2 | half R2 | position RMSE | radius balance |
| --- | --- | --- | --- | --- | --- |
| Same-stage FixedTerm | 0.590 | 0.776 | 0.405 | 1.782 | 0.648 |
| RF quick | 0.536 | 0.725 | 0.347 | 2.861 | 0.381 |
| Union FixedTerm | 0.511 | 0.740 | 0.282 | 1.831 | 0.752 |
| Prev-stage FixedTerm | 0.465 | 0.748 | 0.181 | 2.195 | 0.756 |
| 15-24 FixedTerm | 0.445 | 0.712 | 0.178 | 1.946 | 0.643 |
| Full ridge | 0.341 | 0.610 | 0.072 | 2.057 | 0.795 |
| Mother copy | -0.091 | -0.145 | -0.036 | 3.307 | 0.000 |
| Full linear | -88.211 | -28.209 | -148.213 | 17.988 | -11.894 |


## Few-shot global ranking, n=80

| model | all R2 | mean R2 | half R2 | position RMSE | radius balance |
| --- | --- | --- | --- | --- | --- |
| Same-stage FixedTerm | 0.612 | 0.794 | 0.431 | 1.663 | 0.711 |
| RF quick | 0.609 | 0.773 | 0.446 | 2.005 | 0.530 |
| Union FixedTerm | 0.595 | 0.791 | 0.399 | 1.657 | 0.735 |
| Full ridge | 0.537 | 0.758 | 0.315 | 1.768 | 0.815 |
| Prev-stage FixedTerm | 0.510 | 0.766 | 0.254 | 2.088 | 0.760 |
| 15-24 FixedTerm | 0.482 | 0.726 | 0.237 | 1.855 | 0.719 |
| Mother copy | -0.062 | -0.107 | -0.017 | 3.295 | 0.000 |
| Full linear | -6.763 | -2.964 | -10.562 | 4.651 | -1.057 |
