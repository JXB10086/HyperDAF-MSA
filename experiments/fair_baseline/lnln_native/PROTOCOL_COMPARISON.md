# CMRP vs LNLN Protocol Comparison

This document is a protocol-alignment audit. It is not a head-to-head SOTA table.

| Item | CMRP-MSA | LNLN |
|---|---|---|
| Missing unit | Whole modality | Token/frame |
| Missing construction | Entire T/A/V streams are zero-masked | Tokens/frames are randomly erased |
| Missing pattern | Seven fixed availability conditions plus modality-level random missing masks | Native missing rate `r=0.0...0.9` applied at token/frame level |
| Training protocol | CMRP Protocol A/B; frozen core uses H0+B3 | Released LNLN native pipeline |
| Test protocol | CMRP fixed-pattern and random modality-mask evaluations | LNLN native robust evaluation |
| Metrics | MAE, Corr | MAE, Corr |
| Checkpoint selection | Validation MAE | Test metrics trigger metric-specific best checkpoints |
| Reproduction label | Ours / frozen CMRP protocol | Official/native reproduction with test-driven checkpoint selection |
| Direct numerical comparability | Within the CMRP protocol only | Within the LNLN native protocol only |

## Audit Verdict

- Do not place CMRP and LNLN native values in a single leakage-free fair SOTA table.
- The same metric names do not make the protocols equivalent.
- LNLN is external/native baseline context for robustness trends.
- A direct head-to-head claim requires a separately specified unified re-run.
- The LNLN native result must retain `test_used_for_checkpoint_selection = true`.

## CMRP Frozen Boundary

- Core model: H0+B3.
- `lambda_cons = 0.005`.
- Evidence Package v1 remains frozen.
- No H0/B3 redesign or additional fusion, attention, imputation, or reconstruction is authorized by this audit.
