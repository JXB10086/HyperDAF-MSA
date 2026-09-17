# CMRP Paper Evidence Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-16
- Verification Status: ANALYZED
- Version Label: paper_evidence_audit_v1
- Audit Scope: existing local artifacts only; no training or reproduction rerun

## 1. Scope and Frozen Boundary

This audit evaluates whether the current evidence supports the paper research question **Cross-Missing Representation Stability**. It does not authorize model development or new training.

The following remain frozen:

- Core model: `H0+B3`
- Consistency weight: `lambda_cons=0.005`
- B3 form: normalized stop-gradient consistency, `||sg(zhat_full)-zhat_miss||_2^2`
- Evidence Package v1: unchanged
- F2: auxiliary/ablation only
- F3: failure analysis only

No HME, CMAD, IEMOCAP, dependency installation, or server synchronization was performed.

## 2. LNLN Native Reproduction Seal

LNLN is retained only as **official/native reproduction with test-driven checkpoint selection**.

```text
test_used_for_checkpoint_selection=true
leakage_free_fair_baseline=false
```

The three 200-epoch seeds completed successfully. Standard deviations below are sample standard deviations (`ddof=1`). Metric-specific historical optima may come from different epochs.

| Metric | seed 1111 | seed 1112 | seed 1113 | Mean +/- Std |
|---|---:|---:|---:|---:|
| Best valid MAE | 1.0815 | 0.9892 | 1.0460 | 1.038900 +/- 0.046558 |
| Best valid Corr | 0.5700 | 0.6285 | 0.5665 | 0.588333 +/- 0.034829 |
| Best test MAE | 1.0687 | 1.0523 | 1.0975 | 1.072833 +/- 0.022882 |
| Best test Corr | 0.5212 | 0.5418 | 0.5003 | 0.521100 +/- 0.020750 |
| Training time (s) | 2801 | 2064 | 2432 | 2432.33 +/- 368.50* |

`*` Runtime aggregates mix RTX 3090 and RTX 4090 hardware and are descriptive only.

Native robustness evaluation, using the metric-specific `best_MAE_<seed>.pth` checkpoint and token/frame-level erasure, is:

| Missing rate | MAE Mean +/- Std | Corr Mean +/- Std |
|---:|---:|---:|
| 0.0 | 0.759333 +/- 0.044370 | 0.769000 +/- 0.032006 |
| 0.1 | 0.840100 +/- 0.025847 | 0.704967 +/- 0.028327 |
| 0.2 | 0.913333 +/- 0.015713 | 0.651333 +/- 0.018592 |
| 0.3 | 0.962800 +/- 0.013852 | 0.604833 +/- 0.018375 |
| 0.4 | 1.002233 +/- 0.019720 | 0.570867 +/- 0.024229 |
| 0.5 | 1.075033 +/- 0.032368 | 0.504767 +/- 0.037273 |
| 0.6 | 1.164200 +/- 0.032284 | 0.418033 +/- 0.030420 |
| 0.7 | 1.244500 +/- 0.012985 | 0.340500 +/- 0.013781 |
| 0.8 | 1.329633 +/- 0.011225 | 0.248367 +/- 0.012529 |
| 0.9 | 1.400700 +/- 0.017880 | 0.138000 +/- 0.022358 |

CMRP and LNLN must not be placed in a single leakage-free fair SOTA ranking. CMRP uses whole-modality fixed/random missing masks and validation-MAE checkpoint selection; LNLN native uses token/frame erasure and test-driven metric-specific checkpoint selection. Their shared MAE/Corr metric names do not make the protocols directly comparable.

Sources: `experiments/fair_baseline/lnln_native/summary.json`, `robust_summary.csv`, and `PROTOCOL_COMPARISON.md`.

## 3. Evidence Chain Verdict

Status meanings:

- `PASS`: current artifacts directly support the bounded claim.
- `PARTIAL`: directional evidence exists, but pairing, seeds, dataset coverage, or selection independence is insufficient for a stronger claim.
- `MISSING`: no usable evidence exists for the claim.

| Evidence item | Status | Audit basis | Limitation |
|---|---|---|---|
| Missing -> performance degradation | **PASS** | On paired MOSEI seed 42 H0, missAvg degradation is `DeltaMAE=0.0758` and `DeltaCorr=0.1970`; text-related missing patterns are substantially worse. | Supports existence and pattern dependence, not a population-level effect size. |
| Missing -> representation drift | **PASS** | MOSEI H0 mean `D_cross=0.6978`; the largest values are T, T+A, and T+V missing. MOSI H0 shows the same text-related ordering. | Primary paired diagnostic is MOSEI seed 42. |
| B3 -> drift reduction | **PARTIAL** | On paired MOSEI seed 42, mean `D_cross` changes `0.6978 -> 0.6785`; T missing `0.8996 -> 0.7879`; T+V `0.9948 -> 0.8937`. | Single seed; A, T+A, and A+V drift increase; MOSI H0+B3 `D_cross` is absent. |
| B3 -> robustness improvement | **PARTIAL** | MOSEI missAvg degradation changes `DeltaMAE 0.0758 -> 0.0746` and `DeltaCorr 0.1970 -> 0.1837`. | Improvement is small and pattern-dependent; no paired multi-seed H0 comparison; MOSI runs are not fully paired. |
| Text-related missing effect | **PASS** | Text-related patterns dominate H0 performance degradation and drift on MOSEI and MOSI; B3's clearest joint gains occur for T+V, with the largest drift reduction for T missing. | B3 does not improve every text-related metric or every missing pattern. |
| Multi-seed stability | **PARTIAL** | H0+B3 seeds 42/43/44: MOSEI missAvg MAE `0.7335 +/- 0.0012`, MOSEI randMean MAE `0.7212 +/- 0.0013`, MOSI missAvg MAE `1.2755 +/- 0.0250`. | Shows stability of H0+B3 outcomes, but lacks a multi-seed H0 comparator and multi-seed `D_cross`. |
| Lambda neighborhood | **PARTIAL** | MOSEI seed 42 tested `{0, 0.005, 0.01, 0.02}`; `0.005` has the lowest test missAvg MAE, while `0.01` has lower mean `D_cross`. | Checkpoints use valid MAE, but project-level lambda selection used test fixed-condition metrics/diagnostics. It is test-informed, not clean validation-only tuning. |
| B3 ablation | **PARTIAL** | A same-run paired MOSEI seed 42 comparison isolates `lambda=0` H0 versus `lambda=0.005` H0+B3 with unchanged architecture and training protocol. | No paired multi-seed ablation; MOSI H0 and H0+B3 come from different runners and lack paired `D_cross`. |
| Inference-time full-reference independence | **PASS** | The full view is constructed only in the training consistency branch. Evaluation performs one masked forward pass under `torch.no_grad()` and does not construct or require `z_full`. | Code-level verification; no additional deployment experiment was run. |

Overall: **4 PASS, 5 PARTIAL, 0 MISSING**. No single item is evidence-free, but the central intervention claim remains bounded by single-seed paired evidence.

## 4. RQ-Level Synthesis

The current evidence supports the following bounded chain on MOSEI:

```text
whole-modality missing
    -> performance degradation and cross-missing representation drift
    -> normalized stop-gradient B3
    -> lower average D_cross and slightly smaller average degradation
```

The chain is strongest for text-related missing patterns, especially T and T+V. It is not uniform: MOSEI `D_cross` worsens for A, T+A, and A+V under B3. The evidence therefore supports average and pattern-dependent improvement, not universal drift reduction.

MOSI currently serves as trend/stability verification only. Its H0 and H0+B3 task results are not fully paired, and the H0+B3 `D_cross` artifact is missing.

The code audit also supports inference-time independence:

- `experiments/stage3c_lib.py` constructs `[1,1,1]` only for `z_full` inside the enabled training consistency branch.
- `z_full.detach()` applies stop-gradient to the full-view consistency anchor.
- The task loss remains on the masked task view, and shared parameters continue to learn through that task path.
- `experiments/stage3_lib.py` evaluation applies the supplied mask and performs a single model forward pass; inference does not consume a paired full view.

## 5. Genuine Missing Experiments

These are evidence gaps required to strengthen the existing RQ. They are not proposals for new modules, and this audit does not authorize running them.

1. **Paired multi-seed MOSEI H0 versus H0+B3.** Use identical seeds, data splits, masks, runner, and evaluation. Report fixed-pattern `DeltaMAE/DeltaCorr`, pattern-level and mean `D_cross`, and paired uncertainty/effect estimates. This is the highest-priority missing evidence because it tests whether the observed intervention effect survives beyond seed 42.
2. **Paired MOSI H0 versus H0+B3 representation drift.** If MOSI remains a stability-verification dataset, rerun both variants under one identical runner and record pattern-level/mean `D_cross`. Otherwise, keep MOSI claims explicitly limited to unpaired performance trends and H0-only drift.
3. **Independent lambda confirmation or explicit test-informed disclosure.** Confirm `lambda=0.005` using a validation-only rule independent of test diagnostics, or state transparently that the frozen value was selected with test-informed evidence. The existing artifacts do not support the claim that lambda was tuned solely on validation data.

Secondary, non-blocking gap: the MOSEI random-missing per-rate curve is not stored; only a multi-seed aggregate is available. This is less important than the paired multi-seed intervention evidence.

No new inference experiment is required for the current deployment-independence claim unless a reviewer explicitly requests runtime verification; the present code path is sufficient for a code-level claim.

## 6. Paper-Safe Claims

Allowed primary wording:

> On paired MOSEI seed-42 experiments, B3 reduces average cross-missing representation drift and slightly alleviates average performance degradation, with the clearest improvements under text-related missing patterns. Multi-seed H0+B3 results are stable, but a paired multi-seed H0 comparison is still required to establish the intervention effect across seeds.

Allowed implementation wording:

> The full-modality view is used only as a training-time representation reference. Inference uses the available masked modalities and does not require a full-reference input.

Allowed LNLN wording:

> LNLN results are an official/native reproduction under the released token/frame-erasure protocol, with test-driven checkpoint selection, and are reported as external protocol context rather than a leakage-free head-to-head comparison.

## 7. Claims Not Supported

Do not claim any of the following from the current package:

- B3 uniformly reduces representation drift across all missing patterns.
- B3 causally proves robustness improvement across seeds or datasets.
- `lambda=0.005` was selected solely using validation data.
- MOSI provides a fully paired H0 versus H0+B3 drift ablation.
- CMRP outperforms LNLN, HME, CMAD, EASE, or RECAP in a fair SOTA comparison.
- LNLN native results are leakage-free or directly protocol-equivalent to CMRP.

## 8. Final Verdict

The frozen Evidence Package v1 is sufficient to support a **limited, mechanism-oriented paper claim**: missing-pattern shifts are associated with representation drift, and B3 produces a small average co-improvement in drift and task robustness on the primary paired MOSEI run. It is not yet sufficient for a strong cross-seed causal intervention claim or a leakage-free external SOTA claim.

The most consequential missing evidence is the paired multi-seed MOSEI H0 versus H0+B3 comparison. No model redesign is indicated by this audit.

