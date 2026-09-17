# CMRP v2 R1 Result

## Material Passport

- Mode: paired mechanism screen
- Verification: completed run with hash-based pairing audit
- Inference: descriptive only; n=3

## Pairing

| Seed | Paired |
|---:|---|
| 42 | True |
| 43 | True |
| 44 | True |

## REL vs H0 Primary Results

Negative paired differences indicate lower drift/degradation.

| Metric | H0 Mean +/- SD | REL Mean +/- SD | Paired Difference Mean +/- SD | Improved Seeds |
|---|---:|---:|---:|---:|
| Relational drift RMS | 0.396129 +/- 0.050133 | 0.168113 +/- 0.015464 | -0.228016 +/- 0.050008 | 3/3 |
| missAvg DeltaMAE | 0.071505 +/- 0.006680 | 0.077713 +/- 0.009850 | 0.006208 +/- 0.008840 | 1/3 |
| missAvg DeltaCorr | 0.177449 +/- 0.003213 | 0.172998 +/- 0.010643 | -0.004451 +/- 0.008059 | 2/3 |

## R1 Gate

- Automatic gate: `False`
- Decision: `STOP_BEFORE_R2`
- Manual pattern/seed heterogeneity review remains mandatory even if the automatic gate passes.

- `all_pairing_audits_pass`: `True`
- `relational_drift_improves_vs_h0_at_least_2_of_3`: `True`
- `relational_drift_improves_vs_b3_at_least_2_of_3`: `True`
- `mean_relational_drift_difference_vs_h0_negative`: `True`
- `mean_missAvg_DeltaMAE_not_worse_than_h0`: `False`
- `mean_missAvg_DeltaCorr_not_worse_than_h0`: `True`

R1 never authorizes R2 automatically. It does not tune lambda from test results.
