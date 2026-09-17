# LNLN Native Three-Seed Reproduction

Status: complete.

This directory seals the MOSI LNLN native reproduction for seeds 1111, 1112,
and 1113. Every run completed 200 epochs. Training and robust evaluation exited
with code 0, and no NaN, Inf, OOM, or traceback was found in the training logs.

## Aggregate Results

Values are mean +/- sample standard deviation over three seeds.

| Metric | Mean +/- std |
|---|---:|
| Best valid MAE | 1.038900 +/- 0.046558 |
| Best valid Corr | 0.588333 +/- 0.034829 |
| Best test MAE | 1.072833 +/- 0.022882 |
| Best test Corr | 0.521100 +/- 0.020750 |

The metric-specific best values are independently tracked historical optima and
may occur at different epochs. They must not be interpreted as one joint
checkpoint result.

## Required Caveat

Label this result as:

`official/native reproduction with test-driven checkpoint selection`

The released training code uses test evaluation to save metric-specific best
checkpoints. Therefore this result is not a leakage-free fair baseline.

The robust curves use each seed's test-selected `best_MAE_<seed>.pth` checkpoint
and the native token/frame erase protocol at `r=0.0...0.9`.

## Runtime Caveat

Seed 1111 ran on an RTX 3090; seeds 1112 and 1113 ran on an RTX 4090. Runtime is
reported per seed. Its cross-seed mean is descriptive only and is not a hardware
efficiency comparison.

## Files

- `summary.json`: machine-readable run and aggregate evidence.
- `summary.csv`: per-seed scalar metrics and aggregate rows.
- `robust_summary.csv`: per-rate, per-seed values and aggregate statistics.
- `PROTOCOL_COMPARISON.md`: CMRP-LNLN protocol-alignment audit.
- `seed*/train.log`: complete training logs.
- `seed*/gpu_monitor.csv`: GPU telemetry.
- `seed*/robust.json`: native robust-evaluation rows.

Large checkpoints remain on the server data disk under
`/root/autodl-fs/HyperDAF-MSA-runs/lnln_native_200_seed<seed>/ckpt/mosi/`.
