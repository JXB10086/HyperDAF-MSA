# CMRP v2 R1 Adjudication

- Date: 2026-09-17
- Status: `FROZEN_R1_GATE_FAILED`
- Decision: `STOP_BEFORE_R2`
- Dataset / protocol: MOSEI / Protocol B
- Seeds: 42, 43, 44
- Variants: H0, B3_POINT, REL
- Inference scope: descriptive paired evidence only (`n=3`)

## Pairing and Artifacts

All three seed-level pairing audits passed. The run produced nine
validation-MAE-selected checkpoints, three seed-level `results.json` files,
`summary.json`, `summary.csv`, and `R1_REPORT.md`. The complete runtime directory,
including checkpoints, is retained under the git-ignored
`r1_relational/results/`. The lightweight frozen evidence is tracked under
`r1_relational/frozen_r1/` and verified by `EVIDENCE_MANIFEST.json`.

## Primary Result

Negative paired differences mean lower drift or degradation.

| Metric | H0 mean +/- SD | REL mean +/- SD | REL - H0 mean +/- SD | Improved seeds |
|---|---:|---:|---:|---:|
| Relational drift RMS | 0.396129 +/- 0.050133 | 0.168113 +/- 0.015464 | -0.228016 +/- 0.050008 | 3/3 |
| missAvg DeltaMAE | 0.071505 +/- 0.006680 | 0.077713 +/- 0.009850 | +0.006208 +/- 0.008840 | 1/3 |
| missAvg DeltaCorr | 0.177449 +/- 0.003213 | 0.172998 +/- 0.010643 | -0.004451 +/- 0.008059 | 2/3 |

The preregistered automatic gate failed because mean missAvg DeltaMAE was worse
under REL than H0. The result therefore does not authorize R2.

## Scientific Conclusion

Ordinary relational consistency substantially stabilized cross-sample
representation geometry, with the relational-drift direction consistent across
all three seeds. That geometric improvement did not yield stable task-robustness
improvement: average DeltaMAE worsened and only one seed improved.

The supported boundary is:

> Preserving generic representation geometry is not sufficient evidence of
> preserving task robustness under irreversible modality loss.

This result does not establish that representation stability and robustness are
unrelated, that REL is ineffective for its geometric objective, or that REL
caused the task change. It also does not establish that a task-relevant
consistency objective will work.

## Locked Claims

Allowed:

- REL reduces relational drift in all three R1 seeds.
- The relational-drift improvement did not produce stable DeltaMAE improvement.
- R1 does not support ordinary relational consistency as an effective task-
  robustness intervention under the tested protocol.
- A future, separately preregistered hypothesis may study task-relevant
  cross-missing representation stability.

Forbidden:

- "Representation stability is unrelated to robustness."
- "REL is ineffective."
- "REL causes performance degradation."
- "Task-relevant consistency will work."

## Freeze Rules

- Do not start R2 from this result.
- Do not alter the R1 gate after observing the result.
- Do not tune `lambda_rel` or select REL variants using R1 test outcomes.
- Do not overwrite or regenerate `frozen_r1/` in place.
- Any future study must use a new name, preregistration, output directory, and
  validation-only development process.

## Program State

- CMRP v1 core: frozen `H0+B3`, `lambda_cons=0.005`.
- F2 Global Query Dynamic Fusion: auxiliary/ablation evidence only.
- F3 Hyper-Guided Dynamic Fusion: failure analysis only.
- CMRP v2 R1 REL: frozen mechanism result; geometry-task mismatch.
- Candidate next hypothesis: Task-Relevant Cross-Missing Representation
  Stability, not another generic consistency loss.
