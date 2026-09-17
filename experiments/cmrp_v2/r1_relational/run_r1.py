"""R1: strict paired H0 vs pointwise B3 vs relational consistency on MOSEI."""

import argparse
import copy
import csv
import hashlib
import json
import platform
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(V2_ROOT)]

from cmrp_v2.losses import (  # noqa: E402
    pointwise_consistency_loss,
    relational_consistency_loss,
)
from cmrp_v2.metrics import (  # noqa: E402
    median_pair_distance,
    representation_drift_metrics,
)
from configs.config import Config  # noqa: E402
from datasets import build_mosei_datasets, mosei_dims  # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS, valid_mask_for  # noqa: E402
from experiments.stage3_lib import (  # noqa: E402
    count_params,
    evaluate_with_mask_s3,
    extract_representation_s3,
    forward_model_s3,
)
from models import HyperRepresentationModel  # noqa: E402
from utils import generate_missing_mask  # noqa: E402


SEEDS = (42, 43, 44)
VARIANTS = ("H0", "B3_POINT", "REL")
LABELS = {
    "H0": "H0",
    "B3_POINT": "H0 + pointwise B3 (lambda=0.005)",
    "REL": "H0 + relational consistency (lambda=0.005)",
}
PROTOCOL = "B"
LAMBDA_POINT = 0.005
LAMBDA_REL = 0.005
CONS_SEED_OFF = 777_777
ORDER_SEED_OFF = 100_000
TASK_FORWARD_SEED_OFF = 200_000
FULL_FORWARD_SEED_OFF = 300_000
MISS_FORWARD_SEED_OFF = 400_000
SINGLE_MISSING_MASKS = np.array(
    [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int64
)
CONDITIONS = [name for name, _ in FIXED_TEST_CONDITIONS]
MISS_CONDITIONS = [name for name in CONDITIONS if name != "None"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--metric-block-size", type=int, default=512)
    return parser.parse_args()


def seed_torch(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def mask_tensor(mask, device):
    return torch.as_tensor(mask, dtype=torch.float32, device=device)


def single_missing_mask(batch_size, seed):
    rng = np.random.default_rng(seed)
    return SINGLE_MISSING_MASKS[
        rng.integers(0, SINGLE_MISSING_MASKS.shape[0], size=batch_size)
    ]


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path, chunk_size=1 << 20):
    """SHA-256 of a file on disk, recorded as checkpoint provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_id_hash(digest, ids):
    for sample_id in ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")


def update_seed_hash(digest, seed):
    digest.update(struct.pack("<q", int(seed)))


def build_model(cfg, device):
    return HyperRepresentationModel(
        text_dim=cfg.text_dim,
        audio_dim=cfg.audio_dim,
        vision_dim=cfg.vision_dim,
        hyper_dim=cfg.hyper_dim,
        hidden_dim=cfg.hyper_hidden_dim,
        dropout=cfg.dropout,
    ).to(device)


def save_best_checkpoint(checkpoint_dir, variant, seed, best_epoch, best_mae, best_state):
    """Persist the validation-selected weights and return their provenance record.

    This is instrumentation only. It consumes no randomness and does not influence
    model selection, which remains validation MAE alone.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{variant}_seed{seed}_best_val_mae.pth"
    torch.save(
        {
            "variant": variant,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_valid_mae": best_mae,
            "protocol": PROTOCOL,
            "selection": "validation MAE only",
            "lambda_point": LAMBDA_POINT,
            "lambda_rel": LAMBDA_REL,
            "state_dict": {
                key: value.detach().cpu() for key, value in best_state.items()
            },
        },
        checkpoint_path,
    )
    return {
        "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": file_sha256(checkpoint_path),
    }


def train_variant(variant, seed, cfg, datasets, device, checkpoint_dir=None):
    seed_torch(seed)
    np.random.seed(seed)
    model = build_model(cfg, device)
    initial_state_sha256 = state_hash(model)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    order_generator = torch.Generator().manual_seed(seed + ORDER_SEED_OFF)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        generator=order_generator,
    )
    valid_loader = DataLoader(
        datasets["valid"],
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    valid_mask = valid_mask_for(len(datasets["valid"]), PROTOCOL, cfg)

    digests = {
        name: hashlib.sha256()
        for name in (
            "train_order",
            "task_mask",
            "task_forward_seed",
            "cons_mask",
            "full_forward_seed",
            "miss_forward_seed",
        )
    }
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    global_step = 0
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        task_total = 0.0
        cons_total = 0.0
        sample_count = 0
        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["label"].to(device)
            batch_size = label.size(0)
            update_id_hash(digests["train_order"], batch["id"])

            task_mask_np = generate_missing_mask(
                batch_size,
                missing_probability=cfg.protocol_b_train_missing_prob,
                allow_all_missing=False,
                seed=seed + global_step,
            )
            digests["task_mask"].update(
                np.asarray(task_mask_np, dtype=np.int8).tobytes()
            )
            task_forward_seed = seed + TASK_FORWARD_SEED_OFF + global_step
            update_seed_hash(digests["task_forward_seed"], task_forward_seed)

            optimizer.zero_grad()
            seed_torch(task_forward_seed)
            prediction = forward_model_s3(
                model,
                "hyper",
                text,
                audio,
                vision,
                mask_tensor(task_mask_np, device),
            )
            task_loss = criterion(prediction, label)

            if variant == "H0":
                consistency_loss = task_loss.new_zeros(())
                total_loss = task_loss
            else:
                cons_mask_np = single_missing_mask(
                    batch_size, seed + CONS_SEED_OFF + global_step
                )
                full_seed = seed + FULL_FORWARD_SEED_OFF + global_step
                miss_seed = seed + MISS_FORWARD_SEED_OFF + global_step
                digests["cons_mask"].update(cons_mask_np.astype(np.int8).tobytes())
                update_seed_hash(digests["full_forward_seed"], full_seed)
                update_seed_hash(digests["miss_forward_seed"], miss_seed)

                seed_torch(full_seed)
                _, z_full = forward_model_s3(
                    model,
                    "hyper",
                    text,
                    audio,
                    vision,
                    mask_tensor(np.ones((batch_size, 3)), device),
                    return_rep=True,
                )
                seed_torch(miss_seed)
                _, z_miss = forward_model_s3(
                    model,
                    "hyper",
                    text,
                    audio,
                    vision,
                    mask_tensor(cons_mask_np, device),
                    return_rep=True,
                )
                if variant == "B3_POINT":
                    consistency_loss = pointwise_consistency_loss(z_full, z_miss)
                    weight = LAMBDA_POINT
                else:
                    consistency_loss = relational_consistency_loss(z_full, z_miss)
                    weight = LAMBDA_REL
                total_loss = task_loss + weight * consistency_loss

            total_loss.backward()
            optimizer.step()
            task_total += float(task_loss.item()) * batch_size
            cons_total += float(consistency_loss.item()) * batch_size
            sample_count += batch_size
            global_step += 1

        valid_loss, valid_metrics = evaluate_with_mask_s3(
            model,
            "hyper",
            valid_loader,
            valid_mask,
            device,
            criterion,
        )
        row = {
            "epoch": epoch,
            "train_task": task_total / sample_count,
            "train_consistency": cons_total / sample_count,
            "valid_loss": valid_loss,
            "valid_MAE": float(valid_metrics["MAE"]),
            "valid_Corr": float(valid_metrics["Corr"]),
        }
        history.append(row)
        if row["valid_MAE"] < best_mae:
            best_mae = row["valid_MAE"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"seed={seed} variant={variant} epoch={epoch:02d}/{cfg.epochs} "
            f"task={row['train_task']:.5f} cons={row['train_consistency']:.5f} "
            f"valid_MAE={row['valid_MAE']:.5f} valid_Corr={row['valid_Corr']:.5f}",
            flush=True,
        )

    model.load_state_dict(best_state)

    checkpoint_info = None
    if checkpoint_dir is not None:
        checkpoint_info = save_best_checkpoint(
            checkpoint_dir, variant, seed, best_epoch, best_mae, best_state
        )

    return model, {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_valid_mae": best_mae,
        "n_params": count_params(model),
        "initial_state_sha256": initial_state_sha256,
        **{f"{name}_sha256": digest.hexdigest() for name, digest in digests.items()},
        "runtime_sec": time.time() - started,
        "checkpoint": checkpoint_info,
        "history": history,
    }


def evaluate_task_and_representations(model, test_loader, condition_masks, device):
    fixed = {}
    representations = {}
    for condition in CONDITIONS:
        _, metrics = evaluate_with_mask_s3(
            model, "hyper", test_loader, condition_masks[condition], device
        )
        fixed[condition] = {
            "MAE": float(metrics["MAE"]),
            "Corr": float(metrics["Corr"]),
        }
        representations[condition] = extract_representation_s3(
            model, "hyper", test_loader, condition_masks[condition], device
        )["rep"]

    none = fixed["None"]
    for condition in CONDITIONS:
        fixed[condition]["DeltaMAE"] = fixed[condition]["MAE"] - none["MAE"]
        fixed[condition]["DeltaCorr"] = none["Corr"] - fixed[condition]["Corr"]
    fixed["missAvg"] = {
        metric: float(np.mean([fixed[c][metric] for c in MISS_CONDITIONS]))
        for metric in ("MAE", "Corr", "DeltaMAE", "DeltaCorr")
    }
    return fixed, representations


def add_drift_metrics(models, block_size):
    shared_scale = median_pair_distance(
        models["H0"]["representations"]["None"], block_size=block_size
    )
    for variant in VARIANTS:
        reps = models[variant].pop("representations")
        model_scale = median_pair_distance(reps["None"], block_size=block_size)
        drift = {}
        for condition in MISS_CONDITIONS:
            drift[condition] = representation_drift_metrics(
                reps["None"],
                reps[condition],
                model_pair_scale=model_scale,
                shared_reference_scale=shared_scale,
                block_size=block_size,
            )
        drift["mean(6)"] = {
            metric: float(np.mean([drift[c][metric] for c in MISS_CONDITIONS]))
            for metric in (
                "absolute_l2",
                "cosine_drift",
                "relational_drift_rms",
                "legacy_D_cross",
                "shared_reference_D_cross",
            )
        }
        drift["mean(6)"]["model_pair_scale"] = model_scale
        drift["mean(6)"]["shared_reference_scale"] = shared_scale
        models[variant]["drift"] = drift
    return shared_scale


def pairing_audit(models):
    common_fields = (
        "initial_state_sha256",
        "train_order_sha256",
        "task_mask_sha256",
        "task_forward_seed_sha256",
    )
    audit = {
        field.replace("_sha256", "_match"): len(
            {models[v]["train"][field] for v in VARIANTS}
        )
        == 1
        for field in common_fields
    }
    consistency_fields = (
        "cons_mask_sha256",
        "full_forward_seed_sha256",
        "miss_forward_seed_sha256",
    )
    audit.update(
        {
            field.replace("_sha256", "_match"): models["B3_POINT"]["train"][field]
            == models["REL"]["train"][field]
            for field in consistency_fields
        }
    )
    audit["paired_pass"] = all(audit.values())
    return audit


def flatten_seed(seed_result):
    rows = []
    seed = seed_result["seed"]
    for variant in VARIANTS:
        for condition in CONDITIONS + ["missAvg"]:
            for metric, value in seed_result["models"][variant]["fixed"][condition].items():
                rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "family": "task",
                        "condition": condition,
                        "metric": metric,
                        "value": value,
                    }
                )
        for condition in MISS_CONDITIONS + ["mean(6)"]:
            for metric, value in seed_result["models"][variant]["drift"][condition].items():
                if metric.endswith("scale"):
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "family": "representation",
                        "condition": condition,
                        "metric": metric,
                        "value": value,
                    }
                )
    return rows


def aggregate(long_rows):
    summary = []
    keys = sorted(
        {(r["family"], r["condition"], r["metric"]) for r in long_rows}
    )
    for family, condition, metric in keys:
        by_variant = {}
        for variant in VARIANTS:
            selected = [
                r["value"]
                for r in long_rows
                if r["variant"] == variant
                and (r["family"], r["condition"], r["metric"])
                == (family, condition, metric)
            ]
            by_variant[variant] = np.asarray(selected, dtype=np.float64)
        for variant in VARIANTS:
            values = by_variant[variant]
            for reference in ("H0", "B3_POINT"):
                if variant == reference:
                    continue
                diff = values - by_variant[reference]
                summary.append(
                    {
                        "family": family,
                        "condition": condition,
                        "metric": metric,
                        "variant": variant,
                        "reference": reference,
                        "n_seeds": len(values),
                        "variant_mean": float(values.mean()),
                        "variant_sample_std": float(values.std(ddof=1)),
                        "reference_mean": float(by_variant[reference].mean()),
                        "reference_sample_std": float(
                            by_variant[reference].std(ddof=1)
                        ),
                        "paired_difference_mean": float(diff.mean()),
                        "paired_difference_sample_std": float(diff.std(ddof=1)),
                        "improved_seeds": int((diff < 0).sum()),
                    }
                )
    return summary


def find_summary(summary, family, condition, metric, variant, reference):
    return next(
        row
        for row in summary
        if (
            row["family"],
            row["condition"],
            row["metric"],
            row["variant"],
            row["reference"],
        )
        == (family, condition, metric, variant, reference)
    )


def evaluate_gate(summary, seed_results):
    rel_h0 = find_summary(
        summary, "representation", "mean(6)", "relational_drift_rms", "REL", "H0"
    )
    rel_b3 = find_summary(
        summary,
        "representation",
        "mean(6)",
        "relational_drift_rms",
        "REL",
        "B3_POINT",
    )
    dmae = find_summary(summary, "task", "missAvg", "DeltaMAE", "REL", "H0")
    dcorr = find_summary(summary, "task", "missAvg", "DeltaCorr", "REL", "H0")
    checks = {
        "all_pairing_audits_pass": all(
            result["pairing_audit"]["paired_pass"] for result in seed_results
        ),
        "relational_drift_improves_vs_h0_at_least_2_of_3": rel_h0[
            "improved_seeds"
        ]
        >= 2,
        "relational_drift_improves_vs_b3_at_least_2_of_3": rel_b3[
            "improved_seeds"
        ]
        >= 2,
        "mean_relational_drift_difference_vs_h0_negative": rel_h0[
            "paired_difference_mean"
        ]
        < 0,
        "mean_missAvg_DeltaMAE_not_worse_than_h0": dmae["paired_difference_mean"]
        <= 0,
        "mean_missAvg_DeltaCorr_not_worse_than_h0": dcorr["paired_difference_mean"]
        <= 0,
    }
    return {
        "checks": checks,
        "automatic_gate_pass": all(checks.values()),
        "manual_heterogeneity_review_required": True,
        "decision": "AWAIT_RESEARCH_REVIEW" if all(checks.values()) else "STOP_BEFORE_R2",
    }


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir, payload):
    def row(metric, family, condition, reference="H0"):
        return find_summary(
            payload["summary"], family, condition, metric, "REL", reference
        )

    primary = [
        ("Relational drift RMS", row("relational_drift_rms", "representation", "mean(6)")),
        ("missAvg DeltaMAE", row("DeltaMAE", "task", "missAvg")),
        ("missAvg DeltaCorr", row("DeltaCorr", "task", "missAvg")),
    ]
    lines = [
        "# CMRP v2 R1 Result",
        "",
        "## Material Passport",
        "",
        "- Mode: paired mechanism screen",
        "- Verification: completed run with hash-based pairing audit",
        "- Inference: descriptive only; n=3",
        "",
        "## Pairing",
        "",
        "| Seed | Paired |",
        "|---:|---|",
    ]
    for result in payload["seed_results"]:
        lines.append(f"| {result['seed']} | {result['pairing_audit']['paired_pass']} |")
    lines.extend(
        [
            "",
            "## REL vs H0 Primary Results",
            "",
            "Negative paired differences indicate lower drift/degradation.",
            "",
            "| Metric | H0 Mean +/- SD | REL Mean +/- SD | Paired Difference Mean +/- SD | Improved Seeds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, item in primary:
        lines.append(
            f"| {label} | {item['reference_mean']:.6f} +/- "
            f"{item['reference_sample_std']:.6f} | {item['variant_mean']:.6f} +/- "
            f"{item['variant_sample_std']:.6f} | "
            f"{item['paired_difference_mean']:.6f} +/- "
            f"{item['paired_difference_sample_std']:.6f} | "
            f"{item['improved_seeds']}/{item['n_seeds']} |"
        )
    lines.extend(
        [
            "",
            "## R1 Gate",
            "",
            f"- Automatic gate: `{payload['gate']['automatic_gate_pass']}`",
            f"- Decision: `{payload['gate']['decision']}`",
            "- Manual pattern/seed heterogeneity review remains mandatory even if the automatic gate passes.",
            "",
        ]
    )
    for name, passed in payload["gate"]["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(
        [
            "",
            "R1 never authorizes R2 automatically. It does not tune lambda from test results.",
            "",
        ]
    )
    (output_dir / "R1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.audit_only:
        raise RuntimeError("The full R1 run requires CUDA; use --audit-only without a GPU")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    datasets = build_mosei_datasets(
        cfg.mosei_data_path, cfg.mosei_conv_dir, inf_policy=cfg.mosei_inf_policy
    )
    dims = mosei_dims(datasets)
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = dims
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    config = {
        "dataset": "MOSEI",
        "protocol": PROTOCOL,
        "seeds": args.seeds,
        "variants": list(VARIANTS),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "optimizer": "Adam",
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "lambda_point": LAMBDA_POINT,
        "lambda_rel": LAMBDA_REL,
        "checkpoint_selection": "validation MAE only",
        "checkpoint_artifacts": "best validation-MAE weights saved per variant/seed under checkpoints/",
        "split_sizes": {key: len(value) for key, value in datasets.items()},
        "dims": list(dims),
        "metric_block_size": args.metric_block_size,
    }
    run_config = {"environment": environment, "config": config}
    (output_dir / "RUN_CONFIG.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_config, indent=2), flush=True)
    if args.audit_only:
        print("R1_AUDIT_ONLY_PASS", flush=True)
        return

    test_loader = DataLoader(
        datasets["test"],
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    condition_masks = {
        name: generate_missing_mask(len(datasets["test"]), missing_pattern=pattern)
        for name, pattern in FIXED_TEST_CONDITIONS
    }

    checkpoint_dir = output_dir / "checkpoints"
    run_started = time.time()
    seed_results = []
    long_rows = []
    for seed in args.seeds:
        cfg.seed = seed
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        models = {}
        for variant in VARIANTS:
            model, train_info = train_variant(
                variant, seed, cfg, datasets, device, checkpoint_dir
            )
            fixed, representations = evaluate_task_and_representations(
                model, test_loader, condition_masks, device
            )
            models[variant] = {
                "label": LABELS[variant],
                "train": train_info,
                "fixed": fixed,
                "representations": representations,
            }
            del model
            torch.cuda.empty_cache()

        shared_scale = add_drift_metrics(models, args.metric_block_size)
        audit = pairing_audit(models)
        if not audit["paired_pass"]:
            raise RuntimeError(f"Pairing audit failed for seed {seed}: {audit}")
        seed_result = {
            "seed": seed,
            "shared_reference_scale": shared_scale,
            "pairing_audit": audit,
            "models": models,
        }
        seed_results.append(seed_result)
        rows = flatten_seed(seed_result)
        long_rows.extend(rows)
        (seed_dir / "results.json").write_text(
            json.dumps(seed_result, indent=2), encoding="utf-8"
        )
        write_csv(seed_dir / "metrics_long.csv", rows)

    summary = aggregate(long_rows)
    gate = evaluate_gate(summary, seed_results)
    payload = {
        "status": "complete",
        "label": "CMRP v2 R1 paired relational consistency screen",
        "environment": environment,
        "config": config,
        "runtime_sec": time.time() - run_started,
        "seed_results": seed_results,
        "metrics_long": long_rows,
        "summary": summary,
        "gate": gate,
        "inference_policy": "descriptive paired evidence only; n=3",
        "next_action": "stop and request research adjudication",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "metrics_long.csv", long_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_report(output_dir, payload)
    print(f"R1_COMPLETE decision={gate['decision']}", flush=True)


if __name__ == "__main__":
    main()
