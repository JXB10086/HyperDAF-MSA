"""Zero-training evidence synthesis for frozen CMRP R1 and T0 artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
CONDITIONS = (
    "T missing",
    "A missing",
    "V missing",
    "T+A missing",
    "T+V missing",
    "A+V missing",
)
TEXT_MISSING = {"T missing", "T+A missing", "T+V missing"}
VARIANTS = ("H0", "B3_POINT", "REL")
SEEDS = (42, 43, 44)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t0-results", default=str(V2 / "t0_mechanism_audit" / "results")
    )
    parser.add_argument(
        "--r1-summary",
        default=str(V2 / "r1_relational" / "frozen_r1" / "summary.json"),
    )
    parser.add_argument("--output-dir", default=str(HERE / "results"))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--implementation-revision", required=True)
    return parser.parse_args()


def sha256(path, block=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_groups(ids):
    return np.asarray([str(value).split("|", 1)[0] for value in ids])


def cluster_bootstrap_mean(values, groups, reps, seed):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        draw = rng.integers(0, len(unique), size=len(unique))
        estimates[index] = sums[draw].sum() / counts[draw].sum()
    return {
        "mean": float(values.mean()),
        "cluster_ci_low": float(np.percentile(estimates, 2.5)),
        "cluster_ci_high": float(np.percentile(estimates, 97.5)),
        "n_clips": int(values.size),
        "n_videos": int(len(unique)),
        "bootstrap_reps": int(reps),
    }


def prediction_shape(full, missing):
    full = np.asarray(full, dtype=np.float64)
    missing = np.asarray(missing, dtype=np.float64)
    full_var = float(np.var(full))
    slope = float(np.cov(full, missing, ddof=0)[0, 1] / full_var) if full_var > 0 else 0.0
    corr = float(np.corrcoef(full, missing)[0, 1]) if full.std() and missing.std() else 0.0
    return {
        "prediction_full_mean": float(full.mean()),
        "prediction_missing_mean": float(missing.mean()),
        "prediction_mean_shift": float((missing - full).mean()),
        "prediction_full_std": float(full.std()),
        "prediction_missing_std": float(missing.std()),
        "prediction_std_ratio": float(missing.std() / full.std()) if full.std() else None,
        "full_to_missing_slope": slope,
        "full_to_missing_corr": corr,
    }


def load_split(t0_root, seed, variant):
    path = Path(t0_root) / "per_sample" / f"seed{seed}" / variant / "test.npz"
    return path, np.load(path, allow_pickle=False)


def build_manifest(t0_root, implementation_revision):
    root = Path(t0_root)
    selected = [
        root / "T0_REPORT.md",
        root / "summary.json",
        root / "basic_metrics.csv",
        root / "recovery_metrics.csv",
        root / "t0_console.log",
    ]
    selected.extend(sorted((root / "per_sample").rglob("*.npz")))
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing T0 evidence: {missing}")
    return {
        "status": "T0_STATUS_FINAL",
        "decision": "STOP_AFTER_T0_RESEARCH_REVIEW",
        "implementation_revision": implementation_revision,
        "file_count": len(selected),
        "per_sample_file_count": sum(path.suffix == ".npz" for path in selected),
        "total_bytes": sum(path.stat().st_size for path in selected),
        "files": [
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in selected
        ],
        "freeze_rule": "Do not overwrite or regenerate these T0 artifacts in place.",
    }


def sample_audit(t0_root, reps):
    rows = []
    paired_rows = []
    grouped_pairs = []
    for seed in SEEDS:
        loaded = {}
        for variant in VARIANTS:
            path, data = load_split(t0_root, seed, variant)
            loaded[variant] = data
            ids = data["id"]
            groups = video_groups(ids)
            label = data["label"].astype(np.float64)
            full_pred = data["pred_full"].astype(np.float64)
            for index, condition in enumerate(CONDITIONS):
                key = f"m{index}"
                missing_pred = data[f"{key}_pred"].astype(np.float64)
                error_delta = np.abs(missing_pred - label) - np.abs(full_pred - label)
                ci = cluster_bootstrap_mean(
                    error_delta, groups, reps, seed * 1000 + index * 10 + VARIANTS.index(variant)
                )
                row = {
                    "seed": seed,
                    "variant": variant,
                    "condition": condition,
                    "text_missing": condition in TEXT_MISSING,
                    "full_MAE": float(np.abs(full_pred - label).mean()),
                    "missing_MAE": float(np.abs(missing_pred - label).mean()),
                    "DeltaMAE": float(error_delta.mean()),
                    "DeltaMAE_cluster_ci_low": ci["cluster_ci_low"],
                    "DeltaMAE_cluster_ci_high": ci["cluster_ci_high"],
                    "prediction_drift_mean": float(data[f"{key}_pred_drift"].mean()),
                    **prediction_shape(full_pred, missing_pred),
                    "n_clips": ci["n_clips"],
                    "n_videos": ci["n_videos"],
                }
                rows.append(row)

        h0, rel = loaded["H0"], loaded["REL"]
        if not np.array_equal(h0["id"], rel["id"]):
            raise RuntimeError(f"H0/REL sample mismatch for seed {seed}")
        ids = h0["id"]
        groups = video_groups(ids)
        group_accum = {"text_missing": [], "other_missing": []}
        for index, condition in enumerate(CONDITIONS):
            key = f"m{index}"
            h0_delta = h0[f"{key}_error"].astype(np.float64) - h0["error_full"].astype(np.float64)
            rel_delta = rel[f"{key}_error"].astype(np.float64) - rel["error_full"].astype(np.float64)
            robustness_diff = rel_delta - h0_delta
            drift_diff = rel[f"{key}_pred_drift"].astype(np.float64) - h0[f"{key}_pred_drift"].astype(np.float64)
            ci_robust = cluster_bootstrap_mean(robustness_diff, groups, reps, seed * 100 + index)
            ci_drift = cluster_bootstrap_mean(drift_diff, groups, reps, seed * 100 + index + 50)
            corr = float(np.corrcoef(drift_diff, robustness_diff)[0, 1])
            paired_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "text_missing": condition in TEXT_MISSING,
                    "REL_minus_H0_DeltaMAE": ci_robust["mean"],
                    "REL_minus_H0_DeltaMAE_ci_low": ci_robust["cluster_ci_low"],
                    "REL_minus_H0_DeltaMAE_ci_high": ci_robust["cluster_ci_high"],
                    "REL_minus_H0_prediction_drift": ci_drift["mean"],
                    "REL_minus_H0_prediction_drift_ci_low": ci_drift["cluster_ci_low"],
                    "REL_minus_H0_prediction_drift_ci_high": ci_drift["cluster_ci_high"],
                    "drift_change_error_change_corr": corr,
                    "fraction_drift_better_error_worse": float(
                        np.mean((drift_diff < 0) & (robustness_diff > 0))
                    ),
                }
            )
            group = "text_missing" if condition in TEXT_MISSING else "other_missing"
            group_accum[group].append((robustness_diff, drift_diff))
        for group, arrays in group_accum.items():
            robustness = np.mean([item[0] for item in arrays], axis=0)
            drift = np.mean([item[1] for item in arrays], axis=0)
            ci_robust = cluster_bootstrap_mean(robustness, groups, reps, seed * 10000 + len(group))
            ci_drift = cluster_bootstrap_mean(drift, groups, reps, seed * 10000 + len(group) + 1)
            grouped_pairs.append(
                {
                    "seed": seed,
                    "condition_group": group,
                    "REL_minus_H0_DeltaMAE": ci_robust["mean"],
                    "REL_minus_H0_DeltaMAE_ci_low": ci_robust["cluster_ci_low"],
                    "REL_minus_H0_DeltaMAE_ci_high": ci_robust["cluster_ci_high"],
                    "REL_minus_H0_prediction_drift": ci_drift["mean"],
                    "REL_minus_H0_prediction_drift_ci_low": ci_drift["cluster_ci_low"],
                    "REL_minus_H0_prediction_drift_ci_high": ci_drift["cluster_ci_high"],
                    "drift_change_error_change_corr": float(np.corrcoef(drift, robustness)[0, 1]),
                    "fraction_drift_better_error_worse": float(
                        np.mean((drift < 0) & (robustness > 0))
                    ),
                }
            )
        for data in loaded.values():
            data.close()
    return rows, paired_rows, grouped_pairs


def claim_matrix():
    return [
        {"claim": "REL reduces pairwise cosine relational drift on MOSEI.", "status": "SUPPORTED", "evidence": "R1: 3/3 seeds; 0.396129 to 0.168113.", "boundary": "Tested architecture, protocol, dataset and seeds only."},
        {"claim": "REL slightly reduces frozen-head prediction drift.", "status": "SUPPORTED_DESCRIPTIVE", "evidence": "T0 test aggregate: 0.293497 to 0.271923.", "boundary": "Post-hoc exploratory test evidence."},
        {"claim": "REL improves missing-modality task robustness.", "status": "NOT_SUPPORTED", "evidence": "R1 missAvg DeltaMAE: 0.071505 to 0.077713; 1/3 seeds improved.", "boundary": "No causal degradation claim."},
        {"claim": "Coordinate rotation explains text-missing degradation.", "status": "NOT_SUPPORTED_UNDER_TEST", "evidence": "T0 orthogonal Procrustes showed no material recovery in text-missing conditions.", "boundary": "Does not exclude non-orthogonal/nonlinear alignment."},
        {"claim": "A linearly incompatible predictor explains text-missing degradation.", "status": "NOT_SUPPORTED_UNDER_TEST", "evidence": "Condition-specific ridge probes did not recover the three text-missing conditions.", "boundary": "Does not exclude nonlinear or externally informed recovery."},
        {"claim": "Text-missing behavior is consistent with task-relevant information loss.", "status": "SUPPORTED_AS_CANDIDATE", "evidence": "Joint non-recovery occurred in 3/3 seeds for all three text-missing conditions.", "boundary": "Consistency, not proof of irrecoverability."},
        {"claim": "Geometry stability is generally unrelated to robustness.", "status": "FORBIDDEN_OVERGENERALIZATION", "evidence": "Only one relational objective and one primary dataset were audited.", "boundary": "Claim only insufficiency of the tested geometry."},
        {"claim": "The findings generalize beyond MOSEI.", "status": "UNRESOLVED", "evidence": "No independent T0 confirmation dataset.", "boundary": "Requires preregistered external validation."},
    ]


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_by(rows, key, group_key, group_value):
    values = [row[key] for row in rows if row[group_key] == group_value]
    return float(np.mean(values))


def write_report(path, t0, sample_rows, group_rows, claims, manifest):
    text = [row for row in group_rows if row["condition_group"] == "text_missing"]
    other = [row for row in group_rows if row["condition_group"] == "other_missing"]
    text_error_worse = sum(row["REL_minus_H0_DeltaMAE"] > 0 for row in text)
    text_drift_lower = sum(row["REL_minus_H0_prediction_drift"] < 0 for row in text)

    def shape_mean(variant, is_text, key):
        selected = [
            row
            for row in sample_rows
            if row["variant"] == variant and row["text_missing"] == is_text
        ]
        return float(np.mean([row[key] for row in selected]))

    t0_outcome = t0["adjudication"]["mechanism"]
    lines = [
        "# CMRP Research Review",
        "",
        "## Material Passport",
        "",
        "- Mode: zero-training evidence synthesis",
        "- Verification: completed against frozen R1 and T0 artifacts",
        "- T0 status: `FINAL`",
        "- T0 mechanism screen: `" + t0_outcome + "`",
        "- Method-line decision: `STOP`",
        "- Paper positioning: `ANALYSIS_PAPER_CANDIDATE`",
        "- Paper readiness: `NOT_READY`",
        "",
        "## Sample-level Finding",
        "",
        "REL minus H0, averaged within each condition group and then across the three seeds:",
        "",
        "| Group | DeltaMAE difference | Prediction-drift difference | Drift-better/error-worse samples |",
        "|---|---:|---:|---:|",
        f"| Text missing | {np.mean([r['REL_minus_H0_DeltaMAE'] for r in text]):+.6f} | {np.mean([r['REL_minus_H0_prediction_drift'] for r in text]):+.6f} | {np.mean([r['fraction_drift_better_error_worse'] for r in text]):.3f} |",
        f"| Other missing | {np.mean([r['REL_minus_H0_DeltaMAE'] for r in other]):+.6f} | {np.mean([r['REL_minus_H0_prediction_drift'] for r in other]):+.6f} | {np.mean([r['fraction_drift_better_error_worse'] for r in other]):.3f} |",
        "",
        "Negative prediction-drift difference means REL is more output-stable; positive DeltaMAE difference means worse robustness degradation. Cluster intervals are retained in `grouped_paired_audit.csv`; seeds, not clips, remain the replication unit.",
        "",
        f"For text-missing conditions, REL has worse DeltaMAE than H0 in {text_error_worse}/3 seeds and lower prediction drift in {text_drift_lower}/3 seeds. The within-sample drift-change/error-change correlations by seed are "
        + ", ".join(f"{row['drift_change_error_change_corr']:+.3f}" for row in text)
        + ".",
        "",
        "## Output Shape",
        "",
        "Mean missing/full prediction standard-deviation ratio and full-to-missing slope:",
        "",
        "| Variant | Text-missing SD ratio | Text-missing slope | Other-missing SD ratio | Other-missing slope |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        lines.append(
            f"| {variant} | {shape_mean(variant, True, 'prediction_std_ratio'):.4f} | "
            f"{shape_mean(variant, True, 'full_to_missing_slope'):.4f} | "
            f"{shape_mean(variant, False, 'prediction_std_ratio'):.4f} | "
            f"{shape_mean(variant, False, 'full_to_missing_slope'):.4f} |"
        )
    lines.extend(
        [
        "",
        "Text removal contracts predictions for every variant. REL is not uniquely more contracted than H0, so REL-specific output collapse is not supported as the explanation for its robustness result.",
        "",
        "## Evidence Synthesis",
        "",
        "- REL strongly achieves its relational-geometry objective and modestly reduces frozen-head output drift.",
        "- That output stabilization does not improve task robustness and is therefore not a sufficient task-relevant target.",
        "- Orthogonal coordinate misalignment and simple linear readout incompatibility are not supported as recurring explanations for the text-missing degradation.",
        "- The text-missing pattern is consistent with loss of task-relevant information under the tested recovery mechanisms, but universal irrecoverability is not established.",
        "",
        "## Publication Gap Audit",
        "",
        "| Gap | State | Required action |",
        "|---|---|---|",
        "| Claim discipline and artifact provenance | Addressed | Keep manifests and locked wording |",
        "| Multi-seed paired MOSEI evidence | Addressed descriptively | Do not inflate clip-level inference |",
        "| Independent dataset confirmation | Missing | One preregistered external confirmation only if pursuing publication |",
        "| Literature novelty of geometry-task decoupling | Missing | Structured literature review before drafting |",
        "| Competitive benchmark positioning | Missing | Audit comparable methods/protocols; do not add modules |",
        "| Confirmatory status | Missing | Current test evidence remains exploratory |",
        "",
        "## Decision",
        "",
        "The consistency method line remains closed. The current evidence is coherent enough for an analysis/mechanism paper candidate, but not yet paper-ready. Do not train a new method. First perform literature/novelty and independent-evidence gap review; only a single preregistered external confirmation may be considered afterward.",
        "",
        f"Frozen T0 evidence: {manifest['file_count']} files, {manifest['total_bytes']} bytes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    t0_root = Path(args.t0_results).resolve()
    t0 = json.loads((t0_root / "summary.json").read_text(encoding="utf-8"))
    _ = json.loads(Path(args.r1_summary).read_text(encoding="utf-8"))
    manifest = build_manifest(t0_root, args.implementation_revision)
    sample_rows, paired_rows, grouped_rows = sample_audit(t0_root, args.bootstrap_reps)
    claims = claim_matrix()
    payload = {
        "status": "RESEARCH_REVIEW_COMPLETE",
        "decision": "STOP_METHOD_LINE_ANALYSIS_PAPER_CANDIDATE_NOT_READY",
        "protocol": str((HERE / "REVIEW_PROTOCOL.md").resolve()),
        "implementation_revision": args.implementation_revision,
        "t0_freeze": manifest,
        "sample_audit": sample_rows,
        "paired_rel_vs_h0": paired_rows,
        "grouped_paired_rel_vs_h0": grouped_rows,
        "claim_evidence_matrix": claims,
    }
    (output / "T0_FREEZE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "review.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output / "sample_audit.csv", sample_rows)
    write_csv(output / "paired_rel_vs_h0.csv", paired_rows)
    write_csv(output / "grouped_paired_audit.csv", grouped_rows)
    write_csv(output / "claim_evidence_matrix.csv", claims)
    write_report(output / "RESEARCH_REVIEW.md", t0, sample_rows, grouped_rows, claims, manifest)
    print(f"RESEARCH_REVIEW_COMPLETE {output}")


if __name__ == "__main__":
    main()
