# HyperDAF-MSA / CMRP Research Handoff

## Read This First

This repository contains two distinct CMRP research tracks. Do not merge their
claims or overwrite their artifacts.

### CMRP v1: frozen evidence

- Frozen model: `H0+B3`
- Frozen pointwise consistency weight: `lambda_cons=0.005`
- Research question: Cross-Missing Representation Stability
- Evidence directory: `experiments/cmrp_evidence/`
- Paper audit: `experiments/cmrp_evidence/PAPER_EVIDENCE_AUDIT.md`
- Strict paired result:
  `experiments/cmrp_evidence/paired_multiseed/PAIRED_MULTISeed_REPORT.md`

The strict MOSEI paired seeds 42/43/44 show that pointwise B3 has a small,
seed-dependent effect. Average representation drift is nearly unchanged and
average task degradation is not consistently improved. This result must not be
rewritten as a stable multi-seed robustness gain.

### CMRP v2: active hypothesis, isolated code

- Entry document: `experiments/cmrp_v2/README.md`
- Machine-readable state: `experiments/cmrp_v2/PROJECT_STATE.json`
- Frozen R1 design: `experiments/cmrp_v2/R1_PREREGISTRATION.md`
- Runner: `experiments/cmrp_v2/r1_relational/run_r1.py`

R1 tests only whether relational representation consistency is a better target
than pointwise full-vector matching. It compares `H0`, frozen pointwise B3, and
REL under a strict paired protocol. No R1 GPU result exists at the time of this
handoff.

Do not add balanced pattern sampling, Group-DRO, fusion, attention,
reconstruction, imputation, HME, CMAD, or IEMOCAP before R1 is adjudicated.

## External Baseline Status

LNLN official/native reproduction completed for seeds 1111/1112/1113. It uses
the released token/frame erasure protocol and test-driven checkpoint selection:

```text
test_used_for_checkpoint_selection=true
leakage_free_fair_baseline=false
```

It is external protocol context, not a leakage-free head-to-head CMRP baseline.
See `experiments/fair_baseline/lnln_native/`.

## Data and Existing Code

R1 reuses the existing MOSEI data adapter, H0 architecture, missing simulator,
and evaluation functions. It does not copy or modify them:

- `configs/config.py`
- `datasets/mosei_dataset.py`
- `models/hyper_representation.py`
- `experiments/stage2_lib.py`
- `experiments/stage3_lib.py`
- `utils/missing_simulator.py`

Large datasets, checkpoints, caches, external repositories, and runtime results
must not be committed.

## Current Safe Actions

Without a GPU:

```bash
python -m unittest discover experiments/cmrp_v2/tests -v
python experiments/cmrp_v2/r1_relational/run_r1.py --audit-only
```

With a GPU and explicit research GO:

```bash
python experiments/cmrp_v2/r1_relational/run_r1.py \
  --output-dir experiments/cmrp_v2/r1_relational/results
```

After R1 completes, stop and inspect `R1_REPORT.md`, `summary.json`, pairing
hashes, seed directions, and pattern heterogeneity. Never advance automatically.
