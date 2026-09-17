# Paired Multi-Seed MOSEI H0 vs H0+B3 Report

## Material Passport

- Origin Mode: experiment run + paired descriptive validation
- Seeds: 42, 43, 44
- Verification Status: VERIFIED
- Statistical Scope: descriptive paired differences only; no significance claim

## Pairing Audit

H0 and H0+B3 use identical initial weights, train sample order, task masks, and task-forward dropout seeds within every seed. B3 consistency views use separate deterministic random streams.

| Seed | Init hash match | Order hash match | Task-mask hash match | Paired |
|---:|---|---|---|---|
| 42 | True | True | True | True |
| 43 | True | True | True | True |
| 44 | True | True | True | True |

## Primary Results

Values use sample standard deviation (`ddof=1`). Negative paired differences mean lower degradation/drift under B3.

| Metric | H0 Mean +/- Std | H0+B3 Mean +/- Std | Paired Difference Mean +/- Std | Improved Seeds |
|---|---:|---:|---:|---:|
| Mean D_cross | 0.624098 +/- 0.026125 | 0.623311 +/- 0.019753 | -0.000788 +/- 0.045804 | 2/3 |
| missAvg DeltaMAE | 0.071505 +/- 0.006680 | 0.072855 +/- 0.001259 | 0.001349 +/- 0.007591 | 2/3 |
| missAvg DeltaCorr | 0.177449 +/- 0.003213 | 0.180272 +/- 0.005273 | 0.002823 +/- 0.006952 | 1/3 |

## Seed-Level Primary Metrics

| Seed | Metric | H0 | H0+B3 | B3 - H0 |
|---:|---|---:|---:|---:|
| 42 | missAvg DeltaMAE | 0.074755 | 0.071470 | -0.003285 |
| 42 | missAvg DeltaCorr | 0.180692 | 0.181487 | 0.000795 |
| 42 | mean D_cross | 0.627154 | 0.623632 | -0.003522 |
| 43 | missAvg DeltaMAE | 0.075939 | 0.073162 | -0.002777 |
| 43 | missAvg DeltaCorr | 0.177387 | 0.174497 | -0.002890 |
| 43 | mean D_cross | 0.596580 | 0.642901 | 0.046322 |
| 44 | missAvg DeltaMAE | 0.063822 | 0.073932 | 0.010109 |
| 44 | missAvg DeltaCorr | 0.174267 | 0.184831 | 0.010564 |
| 44 | mean D_cross | 0.648562 | 0.603399 | -0.045163 |

## Interpretation Boundary

With only three paired seeds, this report does not compute p-values, confidence intervals, or standardized effect sizes. Seed-level directional consistency and paired mean +/- sample standard deviation are the authorized readouts. Lambda remains frozen and test-informed; this experiment is not lambda tuning.

Detailed fixed-condition task metrics and pattern-level D_cross results are stored in each seed directory and in `summary.csv`/`paired_detail.csv`.
