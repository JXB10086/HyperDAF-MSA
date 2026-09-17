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

This is initially called **relational representation consistency**, not
task-relevant consistency. The current experiment does not establish that all
preserved relations encode sentiment.

## Project Map

```text
experiments/cmrp_v2/
|-- README.md                     # human entry point
|-- PROJECT_STATE.json            # machine-readable handoff state
|-- R1_PREREGISTRATION.md         # frozen R1 question, protocol, gates
|-- cmrp_v2/
|   |-- __init__.py
|   |-- losses.py                 # pointwise and relational objectives
|   `-- metrics.py                # L2, cosine, relational and shared-scale drift
|-- r1_relational/
|   |-- __init__.py
|   `-- run_r1.py                 # strict paired H0/B3/REL experiment
`-- tests/
    |-- test_losses_metrics.py             # CPU-only mathematical checks
    |-- test_r1_checkpoint.py              # checkpoint provenance round-trip
    `-- test_r1_checkpoint_integration.py  # training-loop wiring, bit-identical
```

Runtime results belong under `r1_relational/results/`, and the validation-selected
weights under `r1_relational/results/checkpoints/`. Neither replaces any v1
artifact. Both are git-ignored (`results/`, `checkpoints/`, `*.pth`).

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

## Commands

CPU-safe code/config audit:

```bash
python experiments/cmrp_v2/r1_relational/run_r1.py --audit-only
```

CPU unit tests:

```bash
python -m unittest discover experiments/cmrp_v2/tests -v
```

Full GPU run after explicit research GO:

```bash
python experiments/cmrp_v2/r1_relational/run_r1.py \
  --output-dir experiments/cmrp_v2/r1_relational/results
```

The full command runs seeds 42/43/44 and stops after R1. It does not launch R2,
R3, HME, CMAD, or IEMOCAP.

## Decision Rule

R1 is a mechanism screen, not a SOTA experiment. REL advances only if the
paired multi-seed evidence is directionally consistent for both:

- relational representation drift; and
- task robustness (`missAvg DeltaMAE` and `missAvg DeltaCorr`).

If relational drift improves without task robustness, the result does not
support the claim that preserving generic sample relations preserves sentiment.
If REL is not stable across seeds, the consistency research line stops before
adding more modules.

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
