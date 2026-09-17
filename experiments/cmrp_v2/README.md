# CMRP v2: Relational Consistency Research Track

## Purpose

This directory is an isolated research track. It does not revise or overwrite
CMRP v1, H0+B3, `lambda_cons=0.005`, or Evidence Package v1.

CMRP v1 established two robust observations and one bounded intervention result:

1. whole-modality missing causes task degradation;
2. whole-modality missing causes representation drift, especially for
   text-related patterns;
3. pointwise B3 has a small, seed-dependent intervention effect.

CMRP v2 tests a new falsifiable hypothesis:

> Matching complete and missing representations point by point may over-constrain
> information that is unavailable after modality loss. Preserving sample-to-sample
> representation relations may be a more appropriate consistency target.

R1 is now complete and frozen. It established that REL can preserve generic
relational geometry without producing stable task-robustness improvement. This
does not establish that representation stability is irrelevant; it shows that
generic relational geometry is not sufficient under the tested protocol.

## Project Map

```text
experiments/cmrp_v2/
|-- README.md                     # human entry point
|-- PROJECT_STATE.json            # machine-readable handoff state
|-- R1_PREREGISTRATION.md         # frozen R1 question, protocol, gates
|-- R1_ADJUDICATION.md            # frozen result, claim boundary, STOP decision
|-- cmrp_v2/
|   |-- __init__.py
|   |-- losses.py                 # pointwise and relational objectives
|   `-- metrics.py                # L2, cosine, relational and shared-scale drift
|-- r1_relational/
|   |-- __init__.py
|   |-- run_r1.py                 # strict paired H0/B3/REL experiment
|   `-- frozen_r1/                # tracked report, summaries, hashes, seed results
`-- tests/
    |-- test_losses_metrics.py             # CPU-only mathematical checks
    |-- test_r1_checkpoint.py              # checkpoint provenance round-trip
    `-- test_r1_checkpoint_integration.py  # training-loop wiring, bit-identical
```

Complete runtime results belong under `r1_relational/results/`, and the validation-selected
weights under `r1_relational/results/checkpoints/`. Neither replaces any v1
artifact. Both are git-ignored (`results/`, `checkpoints/`, `*.pth`). A curated
lightweight copy is frozen and tracked under `r1_relational/frozen_r1/`.

## R1 Adjudication

R1 completed for MOSEI Protocol B, seeds 42/43/44. All pairing audits passed.

| Metric | H0 mean +/- SD | REL mean +/- SD | REL - H0 | Improved seeds |
|---|---:|---:|---:|---:|
| Relational drift RMS | 0.396129 +/- 0.050133 | 0.168113 +/- 0.015464 | -0.228016 | 3/3 |
| missAvg DeltaMAE | 0.071505 +/- 0.006680 | 0.077713 +/- 0.009850 | +0.006208 | 1/3 |
| missAvg DeltaCorr | 0.177449 +/- 0.003213 | 0.172998 +/- 0.010643 | -0.004451 | 2/3 |

The preregistered automatic gate failed because mean DeltaMAE worsened. The
decision is `STOP_BEFORE_R2`. See `R1_ADJUDICATION.md` for the allowed and
forbidden claims.

## R1 Comparison

```text
H0
H0 + pointwise B3 (lambda=0.005, historical frozen comparator)
H0 + relational consistency (lambda=0.005, preregistered scale-matched pilot)
```

All variants use the same architecture, task objective, data split, optimizer,
epochs, sample order, task missing masks, task-forward dropout seeds, checkpoint
rule, and evaluation code. B3 and REL also share their consistency-view masks
and consistency-forward dropout seeds.

R1 intentionally excludes:

- balanced six-pattern consistency sampling;
- Group-DRO;
- F2/F3;
- attention, fusion, reconstruction, or imputation;
- HME/CMAD integration;
- test-driven lambda tuning.

## Checkpoint Artifacts

`run_r1.py` writes one file per variant and seed:

```text
results/checkpoints/<variant>_seed<seed>_best_val_mae.pth
```

Each file records `variant`, `seed`, `best_epoch`, `best_valid_mae`, `protocol`,
`lambda_point`, `lambda_rel`, the selection rule (`validation MAE only`) and the
`state_dict`. The same record plus byte size and SHA-256 is stored in
`train_info["checkpoint"]` and therefore reaches `results.json` and `summary.json`.

This is instrumentation added after the R1 protocol was frozen. It consumes no
randomness and does not affect validation-based selection:
`test_r1_checkpoint_integration.py` asserts that a run with checkpoints enabled
and a run without are bit-identical in parameters, digests and validation MAE.

It exists because the sealed CMRP v1 paired run never called `torch.save`, so
per-sample representation audits were impossible once a process exited. Keeping
the selected weights makes those audits repeatable.

## Verification Commands

CPU-safe code/config audit:

```bash
python experiments/cmrp_v2/r1_relational/run_r1.py --audit-only
```

CPU unit tests:

```bash
python -m unittest discover experiments/cmrp_v2/tests -v
```

The full GPU run is complete. Do not rerun it into `results/` or overwrite
`frozen_r1/`. Verify the evidence against `frozen_r1/EVIDENCE_MANIFEST.json`.

## Decision Rule

R1 is a mechanism screen, not a SOTA experiment. REL advances only if the
paired multi-seed evidence is directionally consistent for both:

- relational representation drift; and
- task robustness (`missAvg DeltaMAE` and `missAvg DeltaCorr`).

The observed outcome is exactly the preregistered mismatch case: relational
drift improved consistently, while task robustness did not. The generic REL
line is stopped before adding more modules. A separately preregistered future
study may ask which task-relevant representation structure should be preserved;
R1 does not establish that such a method will succeed.

## Existing Dependencies

The runner reuses the repository's tracked components:

- `configs/config.py`
- `datasets/mosei_dataset.py`
- `models/hyper_representation.py`
- `experiments/stage2_lib.py`
- `experiments/stage3_lib.py`
- `utils/missing_simulator.py`

No new dependency is required beyond the environment already used by this
project.
