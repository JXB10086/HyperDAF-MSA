"""T0: read-only mechanism audit of the nine frozen R1 checkpoints."""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R1_ROOT = HERE.parent / "r1_relational"
sys.path[:0] = [str(ROOT), str(HERE), str(R1_ROOT)]

from audit_lib import (  # noqa: E402
    drift_projection,
    excess_mae_recovery,
    orthogonal_procrustes,
    regression_metrics,
    select_ridge_alpha,
)
from configs.config import Config  # noqa: E402
from datasets import build_mosei_datasets, mosei_dims  # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS  # noqa: E402
from experiments.stage3_lib import forward_model_s3  # noqa: E402
from run_r1 import VARIANTS, build_model, file_sha256  # noqa: E402
from utils import generate_missing_mask  # noqa: E402


SEEDS = (42, 43, 44)
SPLITS = ("train", "valid", "test")
FULL = "None"
CONDITIONS = tuple(name for name, _ in FIXED_TEST_CONDITIONS)
MISSING = CONDITIONS[1:]
PATTERNS = dict(FIXED_TEST_CONDITIONS)
ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", default=str(R1_ROOT / "results" / "checkpoints")
    )
    parser.add_argument("--output-dir", default=str(HERE / "results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def checkpoint_path(checkpoint_dir, variant, seed):
    return Path(checkpoint_dir) / f"{variant}_seed{seed}_best_val_mae.pth"


def load_checkpoint(path, model, variant, seed):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("variant") != variant or int(payload.get("seed")) != int(seed):
        raise ValueError(f"checkpoint identity mismatch: {path}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return payload


def frozen_manifest_hashes():
    manifest = R1_ROOT / "frozen_r1" / "EVIDENCE_MANIFEST.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    hashes = {}

    def visit(value):
        if isinstance(value, dict):
            if "sha256" in value and "path" in value:
                hashes[Path(value["path"]).name] = value["sha256"]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    hashes.update(data.get("checkpoint_sha256", {}))
    return hashes


def verify_checkpoints(checkpoint_dir, seeds):
    expected = frozen_manifest_hashes()
    rows = []
    for seed in seeds:
        for variant in VARIANTS:
            path = checkpoint_path(checkpoint_dir, variant, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = file_sha256(path)
            registered = expected.get(path.name)
            if registered is None:
                raise ValueError(f"checkpoint absent from frozen manifest: {path.name}")
            if actual != registered:
                raise ValueError(f"frozen hash mismatch: {path.name}")
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "path": str(path.resolve()),
                    "sha256": actual,
                    "frozen_hash_available": True,
                    "frozen_hash_match": actual == registered,
                }
            )
    return rows


def extract_condition(model, loader, mask_all, device):
    model.eval()
    reps, preds, labels, ids, grads = [], [], [], [], []
    offset = 0
    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        label = batch["label"].to(device)
        size = label.shape[0]
        mask = torch.as_tensor(
            mask_all[offset : offset + size], dtype=torch.float32, device=device
        )
        offset += size
        with torch.enable_grad():
            pred, rep = forward_model_s3(
                model, "hyper", text, audio, vision, mask, return_rep=True
            )
            grad = torch.autograd.grad(
                torch.abs(pred - label).sum(), rep, retain_graph=False
            )[0]
        reps.append(rep.detach().cpu().numpy().astype(np.float32))
        preds.append(pred.detach().cpu().numpy().astype(np.float32))
        labels.append(label.detach().cpu().numpy().astype(np.float32))
        grads.append(grad.detach().cpu().numpy().astype(np.float32))
        ids.extend(str(item) for item in batch["id"])
    return {
        "id": np.asarray(ids),
        "label": np.concatenate(labels),
        "z": np.concatenate(reps),
        "pred": np.concatenate(preds),
        "grad": np.concatenate(grads),
    }


def assert_aligned_ids(reference, candidate, condition):
    if not np.array_equal(reference, candidate):
        raise RuntimeError(f"sample alignment failure for {condition}")


def extract_split(model, dataset, cfg, device):
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    masks = {
        name: generate_missing_mask(len(dataset), missing_pattern=pattern)
        for name, pattern in FIXED_TEST_CONDITIONS
    }
    full = extract_condition(model, loader, masks[FULL], device)
    out = {
        "id": full["id"],
        "label": full["label"],
        "z_full": full["z"],
        "pred_full": full["pred"],
        "norm_full": np.linalg.norm(full["z"], axis=1).astype(np.float32),
        "error_full": np.abs(full["pred"] - full["label"]).astype(np.float32),
    }
    for index, condition in enumerate(MISSING):
        miss = extract_condition(model, loader, masks[condition], device)
        assert_aligned_ids(full["id"], miss["id"], condition)
        zf, zm = full["z"], miss["z"]
        l2 = np.linalg.norm(zm - zf, axis=1)
        denom = np.linalg.norm(zf, axis=1) * np.linalg.norm(zm, axis=1)
        cosine = np.zeros_like(l2)
        valid = denom > 1e-12
        cosine[valid] = np.einsum("ij,ij->i", zf[valid], zm[valid]) / denom[valid]
        prefix = f"m{index}"
        full_proj = drift_projection(zf, zm, full["grad"])
        miss_proj = drift_projection(zf, zm, miss["grad"])
        out.update(
            {
                f"{prefix}_condition": np.asarray(condition),
                f"{prefix}_z": zm,
                f"{prefix}_pred": miss["pred"],
                f"{prefix}_norm": np.linalg.norm(zm, axis=1).astype(np.float32),
                f"{prefix}_l2": l2.astype(np.float32),
                f"{prefix}_cosine": cosine.astype(np.float32),
                f"{prefix}_pred_drift": np.abs(miss["pred"] - full["pred"]).astype(np.float32),
                f"{prefix}_error": np.abs(miss["pred"] - full["label"]).astype(np.float32),
                f"{prefix}_grad_full_projection": full_proj["absolute_projection"].astype(np.float32),
                f"{prefix}_grad_full_cosine": full_proj["absolute_cosine"].astype(np.float32),
                f"{prefix}_grad_missing_projection": miss_proj["absolute_projection"].astype(np.float32),
                f"{prefix}_grad_missing_cosine": miss_proj["absolute_cosine"].astype(np.float32),
                f"{prefix}_grad_full_zero": full_proj["zero_denominator"],
                f"{prefix}_grad_missing_zero": miss_proj["zero_denominator"],
            }
        )
    return out


def save_split(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def predictor_numpy(model, z, device, batch_size):
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            batch = torch.as_tensor(
                z[start : start + batch_size], dtype=torch.float32, device=device
            )
            output.append(model.predictor(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(output)


def condition_key(index):
    return f"m{index}"


def summarize_basic(data_by_split, seed, variant):
    rows = []
    for split, data in data_by_split.items():
        full_metrics = regression_metrics(data["pred_full"], data["label"])
        for index, condition in enumerate(MISSING):
            key = condition_key(index)
            miss_metrics = regression_metrics(data[f"{key}_pred"], data["label"])
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "split": split,
                    "condition": condition,
                    "family": "fixed_head",
                    "full_MAE": full_metrics["MAE"],
                    "missing_MAE": miss_metrics["MAE"],
                    "DeltaMAE": miss_metrics["MAE"] - full_metrics["MAE"],
                    "full_Corr": full_metrics["Corr"],
                    "missing_Corr": miss_metrics["Corr"],
                    "DeltaCorr": full_metrics["Corr"] - miss_metrics["Corr"],
                    "prediction_drift_mean": float(data[f"{key}_pred_drift"].mean()),
                    "pointwise_l2_mean": float(data[f"{key}_l2"].mean()),
                    "pointwise_cosine_mean": float(data[f"{key}_cosine"].mean()),
                    "norm_full_mean": float(data["norm_full"].mean()),
                    "norm_missing_mean": float(data[f"{key}_norm"].mean()),
                    "gradient_full_projection_mean": float(data[f"{key}_grad_full_projection"].mean()),
                    "gradient_full_cosine_mean": float(data[f"{key}_grad_full_cosine"].mean()),
                    "gradient_missing_projection_mean": float(data[f"{key}_grad_missing_projection"].mean()),
                    "gradient_missing_cosine_mean": float(data[f"{key}_grad_missing_cosine"].mean()),
                    "gradient_full_zero_count": int(data[f"{key}_grad_full_zero"].sum()),
                    "gradient_missing_zero_count": int(data[f"{key}_grad_missing_zero"].sum()),
                }
            )
    return rows


def run_recovery_audits(model, data, seed, variant, device, batch_size):
    rows = []
    train, valid, test = (data[name] for name in SPLITS)
    full_probe, full_trace = select_ridge_alpha(
        train["z_full"], train["label"], valid["z_full"], valid["label"], ALPHAS
    )
    full_probe_metrics = {
        split: regression_metrics(full_probe.predict(data[split]["z_full"]), data[split]["label"])
        for split in ("valid", "test")
    }
    for index, condition in enumerate(MISSING):
        key = condition_key(index)
        q = orthogonal_procrustes(train[f"{key}_z"], train["z_full"])
        probe, probe_trace = select_ridge_alpha(
            train[f"{key}_z"], train["label"], valid[f"{key}_z"], valid["label"], ALPHAS
        )
        for split in ("valid", "test"):
            item = data[split]
            raw = regression_metrics(item[f"{key}_pred"], item["label"])
            full = regression_metrics(item["pred_full"], item["label"])
            aligned_z = np.asarray(item[f"{key}_z"], dtype=np.float64) @ q
            aligned_pred = predictor_numpy(model, aligned_z, device, batch_size)
            aligned = regression_metrics(aligned_pred, item["label"])
            condition_probe = regression_metrics(
                probe.predict(item[f"{key}_z"]), item["label"]
            )
            transfer_probe = regression_metrics(
                full_probe.predict(item[f"{key}_z"]), item["label"]
            )
            raw_delta = raw["MAE"] - full["MAE"]
            probe_delta = condition_probe["MAE"] - full_probe_metrics[split]["MAE"]
            probe_recovery = (
                float(1.0 - probe_delta / raw_delta) if raw_delta > 0 else None
            )
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "split": split,
                    "condition": condition,
                    "family": "recovery",
                    "raw_full_MAE": full["MAE"],
                    "raw_missing_MAE": raw["MAE"],
                    "procrustes_MAE": aligned["MAE"],
                    "procrustes_Corr": aligned["Corr"],
                    "procrustes_excess_MAE_recovery": excess_mae_recovery(
                        full["MAE"], raw["MAE"], aligned["MAE"]
                    ),
                    "procrustes_representation_RMSE": float(
                        np.sqrt(np.mean((aligned_z - item["z_full"]) ** 2))
                    ),
                    "condition_probe_alpha": probe.alpha,
                    "condition_probe_MAE": condition_probe["MAE"],
                    "condition_probe_Corr": condition_probe["Corr"],
                    "full_probe_alpha": full_probe.alpha,
                    "full_probe_MAE": full_probe_metrics[split]["MAE"],
                    "full_probe_Corr": full_probe_metrics[split]["Corr"],
                    "full_transfer_probe_MAE": transfer_probe["MAE"],
                    "full_transfer_probe_Corr": transfer_probe["Corr"],
                    "probe_DeltaMAE": probe_delta,
                    "probe_relative_degradation_recovery": probe_recovery,
                    "condition_probe_valid_trace": json.dumps(probe_trace),
                    "full_probe_valid_trace": json.dumps(full_trace),
                }
            )
    return rows


def adjudicate(recovery_rows, variant="REL", split="test"):
    selected = [
        row
        for row in recovery_rows
        if row["variant"] == variant and row["split"] == split
    ]
    by_condition = []
    for condition in MISSING:
        rows = [row for row in selected if row["condition"] == condition]
        proc = sum(
            row["procrustes_excess_MAE_recovery"] is not None
            and row["procrustes_excess_MAE_recovery"] >= 0.25
            for row in rows
        )
        probe = sum(
            row["probe_relative_degradation_recovery"] is not None
            and row["probe_relative_degradation_recovery"] >= 0.25
            for row in rows
        )
        joint_failure = sum(
            (
                row["procrustes_excess_MAE_recovery"] is None
                or row["procrustes_excess_MAE_recovery"] < 0.25
            )
            and (
                row["probe_relative_degradation_recovery"] is None
                or row["probe_relative_degradation_recovery"] < 0.25
            )
            and row["probe_DeltaMAE"] > 0
            for row in rows
        )
        by_condition.append(
            {
                "condition": condition,
                "n_seeds": len(rows),
                "material_procrustes_seeds": proc,
                "material_probe_seeds": probe,
                "joint_failure_with_probe_degradation_seeds": joint_failure,
                "recurrent_procrustes": proc >= 2,
                "recurrent_probe": probe >= 2,
                "recurrent_joint_failure": joint_failure >= 2,
            }
        )
    proc_conditions = sum(row["recurrent_procrustes"] for row in by_condition)
    probe_conditions = sum(row["recurrent_probe"] for row in by_condition)
    failure_conditions = sum(row["recurrent_joint_failure"] for row in by_condition)
    if proc_conditions >= 4:
        mechanism = "COORDINATE_MISALIGNMENT_CANDIDATE"
    elif probe_conditions >= 4:
        mechanism = "PREDICTOR_INCOMPATIBILITY_CANDIDATE"
    elif failure_conditions >= 4:
        mechanism = "TASK_INFORMATION_LOSS_CANDIDATE"
    else:
        mechanism = "NO_RECURRENT_MECHANISM"
    return {
        "variant": variant,
        "split": split,
        "thresholds": {
            "material_recovery": 0.25,
            "seeds_per_condition": 2,
            "conditions_required": 4,
        },
        "condition_evidence": by_condition,
        "recurrent_condition_counts": {
            "procrustes": proc_conditions,
            "probe": probe_conditions,
            "joint_failure": failure_conditions,
        },
        "mechanism": mechanism,
        "next_action": "STOP_AFTER_T0_RESEARCH_REVIEW",
        "does_not_authorize_T1": True,
    }


def write_report(path, adjudication):
    counts = adjudication["recurrent_condition_counts"]
    lines = [
        "# T0 Mechanism Audit Result",
        "",
        "- Status: `EXPLORATORY_MECHANISM_EVIDENCE`",
        f"- Registered mechanism outcome: `{adjudication['mechanism']}`",
        "- Decision: `STOP_AFTER_T0_RESEARCH_REVIEW`",
        "- T1/R2 authorization: `FALSE`",
        "",
        "## Registered Recurrence Counts",
        "",
        "A signal must recur in at least 2/3 seeds for at least 4/6 missing conditions.",
        "",
        f"- Procrustes material recovery: {counts['procrustes']}/6 conditions",
        f"- Condition-probe material recovery: {counts['probe']}/6 conditions",
        f"- Joint non-recovery with probe degradation: {counts['joint_failure']}/6 conditions",
        "",
        "| Condition | Procrustes seeds | Probe seeds | Joint-failure seeds |",
        "|---|---:|---:|---:|",
    ]
    for row in adjudication["condition_evidence"]:
        lines.append(
            f"| {row['condition']} | {row['material_procrustes_seeds']}/3 | "
            f"{row['material_probe_seeds']}/3 | "
            f"{row['joint_failure_with_probe_degradation_seeds']}/3 |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report is a threshold-based descriptive screen of three frozen seeds. "
            "It nominates at most a candidate mechanism; it is not confirmatory evidence, "
            "does not alter the frozen R1 decision, and does not authorize a new loss or T1.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = verify_checkpoints(args.checkpoint_dir, args.seeds)
    if args.audit_only:
        print(json.dumps({"checkpoint_audit": checkpoint_rows}, indent=2))
        print("T0_AUDIT_ONLY_PASS")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    cfg = Config()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    datasets = build_mosei_datasets(
        cfg.mosei_data_path, cfg.mosei_conv_dir, inf_policy=cfg.mosei_inf_policy
    )
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = mosei_dims(datasets)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    started = time.time()
    basic_rows, recovery_rows = [], []
    for seed in args.seeds:
        for variant in VARIANTS:
            path = checkpoint_path(args.checkpoint_dir, variant, seed)
            model = build_model(cfg, device)
            load_checkpoint(path, model, variant, seed)
            split_data = {}
            for split in SPLITS:
                print(f"[T0] seed={seed} variant={variant} split={split}", flush=True)
                split_data[split] = extract_split(model, datasets[split], cfg, device)
                save_split(
                    output_dir / "per_sample" / f"seed{seed}" / variant / f"{split}.npz",
                    split_data[split],
                )
            basic_rows.extend(summarize_basic(split_data, seed, variant))
            recovery_rows.extend(
                run_recovery_audits(
                    model, split_data, seed, variant, device, cfg.batch_size
                )
            )
            del split_data, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    payload = {
        "status": "T0_EXPLORATORY_COMPLETE",
        "protocol": str((HERE / "T0_PROTOCOL.md").resolve()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "checkpoint_audit": checkpoint_rows,
        "config": {
            "seeds": args.seeds,
            "variants": list(VARIANTS),
            "conditions": list(CONDITIONS),
            "splits": {name: len(dataset) for name, dataset in datasets.items()},
            "ridge_alphas": list(ALPHAS),
            "batch_size": cfg.batch_size,
        },
        "runtime_sec": time.time() - started,
        "basic": basic_rows,
        "recovery": recovery_rows,
        "adjudication": adjudicate(recovery_rows),
        "boundary": "Exploratory mechanism evidence only; does not authorize T1 or R2.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_csv(output_dir / "basic_metrics.csv", basic_rows)
    write_csv(output_dir / "recovery_metrics.csv", recovery_rows)
    write_report(output_dir / "T0_REPORT.md", payload["adjudication"])
    print(f"T0_COMPLETE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
