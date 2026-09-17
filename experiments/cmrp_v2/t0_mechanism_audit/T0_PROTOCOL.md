# T0 Cross-Missing Representation Mechanism Audit

Status: `PREREGISTERED_BEFORE_T0_RESULTS`

## Purpose

T0 is a post-hoc mechanism audit of the nine frozen MOSEI R1 checkpoints. It
does not train a model, introduce a loss, tune REL, modify R1, or authorize a
follow-up experiment. It asks why REL can reduce pairwise cosine Gram-matrix
drift without producing stable task-robustness improvement.

The competing explanations are:

1. **Coordinate misalignment:** task information and relational geometry are
   retained, but missing-modality representations occupy a rotated coordinate
   system that the frozen prediction head cannot read.
2. **Predictor incompatibility:** task information remains linearly readable,
   but not by the frozen nonlinear head at the missing representations.
3. **Task-relevant information loss:** neither an orthogonal alignment nor a
   split-disciplined linear readout recovers the missing-modality degradation.

## Frozen Inputs

- Dataset/protocol: MOSEI / Protocol B.
- Seeds: 42, 43, 44.
- Variants: H0, B3_POINT, REL (mutually exclusive R1 variants).
- Conditions: None, T, A, V, T+A, T+V, A+V missing.
- Weights: validation-MAE-selected R1 checkpoints only.
- Sample order and masks: deterministic split order and the existing fixed
  condition masks.

Checkpoint SHA-256 values are verified against the frozen R1 evidence before
inference. Model weights are loaded read-only and `model.eval()` is mandatory.

## Data-use Boundary

- Orthogonal Procrustes is fit on train only and then applied unchanged to
  valid and test.
- Ridge-probe parameters are fit on train only. Ridge alpha is selected by
  valid MAE. Test is evaluated once after selection.
- T0 test results are exploratory evidence. They may not be used to choose a
  future objective and then serve as confirmatory evidence for that objective.
- Samples are not experimental replicates. Evidence is reported at
  seed x condition x variant level; no sample-level significance test is used.

## Saved Per-sample Quantities

For every split, seed, variant, and condition, the audit saves sample IDs,
labels, full/missing representations and norms, pointwise L2 and cosine,
frozen-head predictions, absolute prediction drift, and full/missing absolute
errors. Gradient vectors are not retained; only per-sample projections and
norms needed by the registered audit are saved.

## Registered Audits

### A. Frozen-head output drift

Primary quantities are missing-minus-full MAE, correlation loss, and mean
absolute prediction drift. These determine whether geometric preservation is
accompanied by output stability.

### B. Task-gradient-aligned drift

For `delta = z_missing - z_full`, compute gradients of per-sample L1 task loss
with respect to `z` at both endpoints. Report absolute projection and cosine:

`abs(delta dot g)`, and `abs(delta dot g) / (||delta|| ||g||)`.

The full-endpoint result is primary; the missing endpoint is a sensitivity
check. Zero-norm cases are explicitly mapped to zero and counted.

### C. Train-fitted orthogonal Procrustes

For each seed, variant, and missing condition, fit the origin-preserving map

`Q = argmin ||Z_missing_train Q - Z_full_train||_F`, with `Q^T Q = I`.

Apply the same Q to valid/test and feed the aligned representation through the
unchanged frozen predictor. Report aligned MAE/correlation, excess-MAE recovery,
and representation alignment error. No centering, translation, scale, or
test-fitted transform is permitted in the primary audit.

### D. Split-disciplined linear probes

Standardized ridge probes use train statistics only. Alpha is selected from
`[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]` by valid MAE with the smaller
alpha breaking ties. Two diagnostics are reported:

- condition-specific probe: fit/read the same condition, measuring whether
  label information remains linearly readable;
- full-transfer probe: fit on full train and apply unchanged to missing
  valid/test, measuring shared-coordinate readability.

## Interpretation and Stop Rule

Results are summarized descriptively across three paired seeds and six missing
conditions. A mechanism is called **recurrent** only when its predicted
direction occurs in at least 2/3 seeds for at least 4/6 conditions. A recovery
is called **material** only when median excess-MAE recovery is at least 25%;
denominators at or below zero are reported as not applicable rather than forced
into a recovery ratio.

- recurrent material Procrustes recovery supports coordinate misalignment;
- recurrent condition-specific probe recovery without Procrustes recovery
  supports predictor incompatibility;
- recurrent failure of both recovery routes, together with degraded missing
  probe performance, supports information loss as a candidate;
- mixed/non-recurrent evidence triggers `STOP_CONSISTENCY_METHOD_LINE`.

T0 can nominate a mechanism for a separately preregistered T1. It cannot start
T1, create TASK_REL, or overturn the frozen `STOP_BEFORE_R2` decision.

