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

### CMRP v2 R1: frozen failed gate

- Entry document: `experiments/cmrp_v2/README.md`
- Machine-readable state: `experiments/cmrp_v2/PROJECT_STATE.json`
- Frozen R1 design: `experiments/cmrp_v2/R1_PREREGISTRATION.md`
- Frozen adjudication: `experiments/cmrp_v2/R1_ADJUDICATION.md`
- Frozen evidence: `experiments/cmrp_v2/r1_relational/frozen_r1/`
- Runner: `experiments/cmrp_v2/r1_relational/run_r1.py`

R1 completed on MOSEI Protocol B for seeds 42/43/44. All pairing audits passed
and nine validation-selected checkpoints were saved. REL reduced relational
drift from `0.396129` to `0.168113` with improvement in 3/3 seeds, but mean
missAvg DeltaMAE worsened from `0.071505` to `0.077713` and improved in only
1/3 seeds. The preregistered automatic gate therefore failed.

The locked decision is `STOP_BEFORE_R2`. Do not change the gate, tune
`lambda_rel` from the R1 test results, launch R2, or add modules to rescue the
failed hypothesis. A future task-relevant stability study requires a new name,
preregistration, development boundary, and output directory.

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

Large datasets, checkpoints, caches, external repositories, and complete runtime
directories must not be committed. The curated lightweight R1 evidence under
`r1_relational/frozen_r1/` is intentionally tracked; checkpoints remain under
the git-ignored `r1_relational/results/` and are covered by the evidence manifest.

## Current Safe Actions

Without a GPU:

```bash
python -m unittest discover experiments/cmrp_v2/tests -v
python experiments/cmrp_v2/r1_relational/run_r1.py --audit-only
```

Read and verify the frozen decision:

```text
experiments/cmrp_v2/R1_ADJUDICATION.md
experiments/cmrp_v2/r1_relational/frozen_r1/R1_REPORT.md
experiments/cmrp_v2/r1_relational/frozen_r1/EVIDENCE_MANIFEST.json
```

Do not start R2. The next research question, if pursued, is task-relevant
cross-missing representation stability rather than another generic consistency
loss. It is an untested direction, not a conclusion from R1.
