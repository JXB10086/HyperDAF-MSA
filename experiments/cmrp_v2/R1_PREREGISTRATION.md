# R1 Preregistration: Pointwise vs Relational Consistency

## Status

- Stage: design frozen before any R1 result exists
- Dataset: CMU-MOSEI
- Seeds: 42, 43, 44
- Protocol: existing Protocol B, unchanged
- Purpose: mechanism screen
- Statistical scope: paired descriptive evidence; no significance claim at `n=3`

## Hypothesis

Pointwise B3 aligns each missing-view representation with the corresponding
complete-view vector. Under irreversible modality loss, this may also constrain
unavailable modality-specific information. R1 tests whether preserving the
within-batch relational geometry is a more stable target.

This mechanism is a hypothesis, not an established explanation of v1 results.

## Fixed Variants

1. `H0`: task loss only.
2. `B3_POINT`: frozen normalized stop-gradient pointwise consistency,
   `lambda=0.005`.
3. `REL`: normalized stop-gradient relational consistency, `lambda=0.005`.

For a batch of L2-normalized representations, define off-diagonal cosine
similarity matrices `S_full` and `S_miss`. The REL objective is:

```text
L_rel = sum_{i != j} (sg(S_full[i,j]) - S_miss[i,j])^2 / batch_size
```

The division by batch size gives a per-anchor relation error and keeps the loss
on a comparable order to the existing pointwise squared-distance objective.
The diagonal is excluded because self-similarity is identically one.

The REL weight is deliberately fixed before observing R1 test results. If its
scale is unsuitable, any later weight study must be a separately named,
validation-only R1b experiment; R1 test outcomes must not be used to tune it.

## Pairing Contract

Within each seed, all three variants must have matching hashes for:

- initial parameter state;
- complete train sample order;
- task missing-mask sequence;
- task-forward dropout seed sequence.

`B3_POINT` and `REL` must additionally share:

- single-missing consistency masks;
- full-view consistency-forward seeds;
- missing-view consistency-forward seeds.

Consistency forward passes use independent deterministic random streams so they
cannot change the next task batch, task mask, shuffle, or task-forward dropout.

## Model Selection and Test Boundary

- Checkpoint selection: existing validation MAE only.
- Test split: fixed automated evaluation after training; no adaptive decision
  occurs inside the run.
- No lambda selection, early stopping policy change, or architecture change is
  allowed after inspecting test output.

## Evaluation

Task metrics for `None`, `T`, `A`, `V`, `T+A`, `T+V`, `A+V`:

- MAE and Corr;
- DeltaMAE relative to `None`;
- DeltaCorr relative to `None`;
- six-pattern missAvg.

Representation metrics for every missing pattern:

1. mean absolute L2 drift;
2. mean cosine drift `1-cos(z_full,z_miss)`;
3. relational drift RMS over all off-diagonal sample pairs;
4. legacy model-specific `D_cross`;
5. shared-reference `D_cross`, using the H0 full-view median pair distance for
   all three variants within a seed.

The shared denominator prevents a change in each model's representation scale
from being silently absorbed into a cross-model drift comparison.

## R1 Decision Gate

REL may proceed to R2 only if all conditions hold:

1. pairing audit passes for every seed;
2. relational drift improves versus both H0 and B3_POINT in at least 2/3 seeds;
3. mean paired relational-drift difference is negative;
4. missAvg DeltaMAE does not worsen on average;
5. missAvg DeltaCorr does not worsen on average;
6. the direction is not driven solely by one extreme seed or one missing pattern.

Passing R1 supports further study, not a causal or SOTA claim. Failing R1 stops
the consistency line before balanced sampling, Group-DRO, or other modules.

## Amendments After Freeze

| Date | Change | Effect on the preregistered contrast |
| --- | --- | --- |
| 2026-09-17 | `run_r1.py` saves the validation-selected weights per variant and seed under `results/checkpoints/`, and records path, byte size and SHA-256 in `results.json` / `summary.json` | None. Instrumentation only: consumes no randomness, selection rule remains validation MAE alone |

Amendment policy: changes are recorded here, never applied silently. Any change
that touches seeds, sample order, masks, dropout draws, the optimizer, the
selection rule, the lambdas or the gate invalidates the frozen protocol and must
be dated together with its reason.

## Explicit Exclusions

R1 does not include all-pattern consistency training, Group-DRO, task-aware
relation weights, F2/F3, attention, fusion, reconstruction, imputation, HME,
CMAD, IEMOCAP, or a new data split.
