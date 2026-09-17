"""Paired MOSEI H0 vs H0+B3 multi-seed evidence run.

This runner does not modify either model. It isolates the random streams used
for training order, task masks, task-forward dropout, and B3 consistency views
so the H0/B3 comparison shares every stochastic realization that can be shared.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from configs.config import Config  # noqa: E402
from datasets import build_mosei_datasets, mosei_dims  # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS, valid_mask_for  # noqa: E402
from experiments.stage3_lib import (  # noqa: E402
    count_params,
    evaluate_with_mask_s3,
    extract_representation_s3,
    forward_model_s3,
)
from experiments.stage3c_lib import (  # noqa: E402
    CONS_SEED_OFF,
    _single_missing_cons_mask,
)
from models import HyperRepresentationModel  # noqa: E402
from utils import generate_missing_mask  # noqa: E402


SEEDS = (42, 43, 44)
PROTOCOL = "B"
LAM_B3 = 0.005
CONS_NORM = "l2"
ORDER_SEED_OFF = 100_000
TASK_FORWARD_SEED_OFF = 200_000
FULL_FORWARD_SEED_OFF = 300_000
MISS_FORWARD_SEED_OFF = 400_000
MISS_CONDITIONS = [name for name, _ in FIXED_TEST_CONDITIONS if name != "None"]
MODEL_LABELS = {"H0": "H0", "B3": "H0+B3 (lambda=0.005)"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def seed_torch(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def mask_tensor(mask_np, device):
    return torch.as_tensor(np.asarray(mask_np), dtype=torch.float32, device=device)


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        arr = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def update_order_hash(digest, ids):
    for sample_id in ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")


def train_paired_variant(variant, seed, cfg, datasets, device):
    use_cons = variant == "B3"
    seed_torch(seed)
    np.random.seed(seed)

    model = HyperRepresentationModel(
        text_dim=cfg.text_dim,
        audio_dim=cfg.audio_dim,
        vision_dim=cfg.vision_dim,
        hyper_dim=cfg.hyper_dim,
        hidden_dim=cfg.hyper_hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    initial_sha256 = state_hash(model)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    order_generator = torch.Generator()
    order_generator.manual_seed(seed + ORDER_SEED_OFF)
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

    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    global_step = 0
    order_digest = hashlib.sha256()
    task_mask_digest = hashlib.sha256()
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        task_sum = 0.0
        cons_sum = 0.0
        n = 0
        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["label"].to(device)
            batch_size = label.size(0)

            update_order_hash(order_digest, batch["id"])
            task_mask_np = generate_missing_mask(
                batch_size,
                missing_probability=cfg.protocol_b_train_missing_prob,
                allow_all_missing=False,
                seed=seed + global_step,
            )
            task_mask_digest.update(np.asarray(task_mask_np, dtype=np.int8).tobytes())
            task_mask = mask_tensor(task_mask_np, device)

            optimizer.zero_grad()
            seed_torch(seed + TASK_FORWARD_SEED_OFF + global_step)
            prediction = forward_model_s3(
                model, "hyper", text, audio, vision, task_mask
            )
            task_loss = criterion(prediction, label)

            if use_cons:
                full_mask = mask_tensor(np.ones((batch_size, 3), dtype=np.int64), device)
                cons_mask_np = _single_missing_cons_mask(
                    batch_size, seed + CONS_SEED_OFF + global_step
                )
                cons_mask = mask_tensor(cons_mask_np, device)

                seed_torch(seed + FULL_FORWARD_SEED_OFF + global_step)
                _, z_full = forward_model_s3(
                    model,
                    "hyper",
                    text,
                    audio,
                    vision,
                    full_mask,
                    return_rep=True,
                )
                seed_torch(seed + MISS_FORWARD_SEED_OFF + global_step)
                _, z_miss = forward_model_s3(
                    model,
                    "hyper",
                    text,
                    audio,
                    vision,
                    cons_mask,
                    return_rep=True,
                )
                z_anchor = z_full.detach()
                z_anchor = z_anchor / (z_anchor.norm(dim=1, keepdim=True) + 1e-8)
                z_miss = z_miss / (z_miss.norm(dim=1, keepdim=True) + 1e-8)
                cons_loss = ((z_anchor - z_miss) ** 2).sum(dim=1).mean()
                loss = task_loss + LAM_B3 * cons_loss
                cons_value = float(cons_loss.item())
            else:
                loss = task_loss
                cons_value = 0.0

            loss.backward()
            optimizer.step()
            task_sum += float(task_loss.item()) * batch_size
            cons_sum += cons_value * batch_size
            n += batch_size
            global_step += 1

        valid_loss, valid_metrics = evaluate_with_mask_s3(
            model,
            "hyper",
            valid_loader,
            valid_mask,
            device,
            criterion,
        )
        entry = {
            "epoch": epoch,
            "train_task": task_sum / n,
            "train_cons": cons_sum / n,
            "valid_loss": valid_loss,
            "valid_MAE": valid_metrics["MAE"],
            "valid_Corr": valid_metrics["Corr"],
        }
        history.append(entry)
        if valid_metrics["MAE"] < best_mae:
            best_mae = float(valid_metrics["MAE"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"seed={seed} variant={variant} epoch={epoch:02d}/{cfg.epochs} "
            f"task={entry['train_task']:.5f} cons={entry['train_cons']:.5f} "
            f"valid_MAE={entry['valid_MAE']:.5f} "
            f"valid_Corr={entry['valid_Corr']:.5f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    return model, {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_valid_mae": best_mae,
        "n_params": count_params(model),
        "initial_state_sha256": initial_sha256,
        "train_order_sha256": order_digest.hexdigest(),
        "task_mask_sha256": task_mask_digest.hexdigest(),
        "runtime_sec": time.time() - started,
        "history": history,
    }


def evaluate_variant(model, test_loader, condition_masks, device):
    fixed = {}
    reps = {}
    for condition, _ in FIXED_TEST_CONDITIONS:
        _, metrics = evaluate_with_mask_s3(
            model, "hyper", test_loader, condition_masks[condition], device
        )
        fixed[condition] = {
            "MAE": float(metrics["MAE"]),
            "Corr": float(metrics["Corr"]),
        }
        reps[condition] = extract_representation_s3(
            model, "hyper", test_loader, condition_masks[condition], device
        )["rep"]

    none = fixed["None"]
    for condition in fixed:
        fixed[condition]["DeltaMAE"] = fixed[condition]["MAE"] - none["MAE"]
        fixed[condition]["DeltaCorr"] = none["Corr"] - fixed[condition]["Corr"]
    fixed["missAvg"] = {
        metric: float(np.mean([fixed[c][metric] for c in MISS_CONDITIONS]))
        for metric in ("MAE", "Corr", "DeltaMAE", "DeltaCorr")
    }

    z_full = reps["None"]
    median_pair_dist = float(np.median(cdist(z_full, z_full)))
    dcross = {}
    for condition in MISS_CONDITIONS:
        raw = float(np.linalg.norm(reps[condition] - z_full, axis=1).mean())
        dcross[condition] = {
            "raw_drift_L2": raw,
            "D_cross": raw / (median_pair_dist + 1e-9),
        }
    dcross["mean(6)"] = {
        "raw_drift_L2": float(
            np.mean([dcross[c]["raw_drift_L2"] for c in MISS_CONDITIONS])
        ),
        "D_cross": float(np.mean([dcross[c]["D_cross"] for c in MISS_CONDITIONS])),
    }
    return fixed, dcross, median_pair_dist


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(seed_results):
    rows = []
    detail = []
    for seed_result in seed_results:
        seed = seed_result["seed"]
        for condition in [name for name, _ in FIXED_TEST_CONDITIONS] + ["missAvg"]:
            for metric in ("MAE", "Corr", "DeltaMAE", "DeltaCorr"):
                if condition == "None" and metric.startswith("Delta"):
                    continue
                h0 = seed_result["models"]["H0"]["fixed"][condition][metric]
                b3 = seed_result["models"]["B3"]["fixed"][condition][metric]
                detail.append(
                    {
                        "seed": seed,
                        "family": "task",
                        "condition": condition,
                        "metric": metric,
                        "H0": h0,
                        "H0_B3": b3,
                        "difference_B3_minus_H0": b3 - h0,
                        "relative_change": (b3 - h0) / (abs(h0) + 1e-12),
                    }
                )
        for condition in MISS_CONDITIONS + ["mean(6)"]:
            h0 = seed_result["models"]["H0"]["dcross"][condition]["D_cross"]
            b3 = seed_result["models"]["B3"]["dcross"][condition]["D_cross"]
            detail.append(
                {
                    "seed": seed,
                    "family": "representation",
                    "condition": condition,
                    "metric": "D_cross",
                    "H0": h0,
                    "H0_B3": b3,
                    "difference_B3_minus_H0": b3 - h0,
                    "relative_change": (b3 - h0) / (abs(h0) + 1e-12),
                }
            )

    keys = sorted({(r["family"], r["condition"], r["metric"]) for r in detail})
    for family, condition, metric in keys:
        selected = [
            r
            for r in detail
            if (r["family"], r["condition"], r["metric"])
            == (family, condition, metric)
        ]
        h0 = np.array([r["H0"] for r in selected], dtype=np.float64)
        b3 = np.array([r["H0_B3"] for r in selected], dtype=np.float64)
        diff = b3 - h0
        rel = np.array([r["relative_change"] for r in selected], dtype=np.float64)
        rows.append(
            {
                "family": family,
                "condition": condition,
                "metric": metric,
                "n_seeds": len(selected),
                "H0_mean": float(h0.mean()),
                "H0_sample_std": float(h0.std(ddof=1)),
                "H0_B3_mean": float(b3.mean()),
                "H0_B3_sample_std": float(b3.std(ddof=1)),
                "paired_difference_mean": float(diff.mean()),
                "paired_difference_sample_std": float(diff.std(ddof=1)),
                "relative_change_mean": float(rel.mean()),
                "B3_improved_seeds": int((diff < 0).sum()),
            }
        )
    return detail, rows


def primary_summary(summary_rows):
    targets = {
        "mean_D_cross": ("representation", "mean(6)", "D_cross"),
        "missAvg_DeltaMAE": ("task", "missAvg", "DeltaMAE"),
        "missAvg_DeltaCorr": ("task", "missAvg", "DeltaCorr"),
    }
    result = {}
    for name, key in targets.items():
        result[name] = next(
            row
            for row in summary_rows
            if (row["family"], row["condition"], row["metric"]) == key
        )
    return result


def build_report(output_dir, payload):
    primary = payload["primary_metrics"]
    lines = [
        "# Paired Multi-Seed MOSEI H0 vs H0+B3 Report",
        "",
        "## Material Passport",
        "",
        "- Origin Mode: experiment run + paired descriptive validation",
        f"- Seeds: {', '.join(str(x) for x in payload['seeds'])}",
        "- Verification Status: VERIFIED",
        "- Statistical Scope: descriptive paired differences only; no significance claim",
        "",
        "## Pairing Audit",
        "",
        "H0 and H0+B3 use identical initial weights, train sample order, task masks, "
        "and task-forward dropout seeds within every seed. B3 consistency views use "
        "separate deterministic random streams.",
        "",
        "| Seed | Init hash match | Order hash match | Task-mask hash match | Paired |",
        "|---:|---|---|---|---|",
    ]
    for result in payload["seed_results"]:
        audit = result["pairing_audit"]
        lines.append(
            f"| {result['seed']} | {audit['initial_state_match']} | "
            f"{audit['train_order_match']} | {audit['task_mask_match']} | "
            f"{audit['paired_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Primary Results",
            "",
            "Values use sample standard deviation (`ddof=1`). Negative paired "
            "differences mean lower degradation/drift under B3.",
            "",
            "| Metric | H0 Mean +/- Std | H0+B3 Mean +/- Std | Paired Difference Mean +/- Std | Improved Seeds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, key in [
        ("Mean D_cross", "mean_D_cross"),
        ("missAvg DeltaMAE", "missAvg_DeltaMAE"),
        ("missAvg DeltaCorr", "missAvg_DeltaCorr"),
    ]:
        row = primary[key]
        lines.append(
            f"| {label} | {row['H0_mean']:.6f} +/- {row['H0_sample_std']:.6f} | "
            f"{row['H0_B3_mean']:.6f} +/- {row['H0_B3_sample_std']:.6f} | "
            f"{row['paired_difference_mean']:.6f} +/- "
            f"{row['paired_difference_sample_std']:.6f} | "
            f"{row['B3_improved_seeds']}/{row['n_seeds']} |"
        )
    lines.extend(
        [
            "",
            "## Seed-Level Primary Metrics",
            "",
            "| Seed | Metric | H0 | H0+B3 | B3 - H0 |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    target_keys = {
        ("representation", "mean(6)", "D_cross"),
        ("task", "missAvg", "DeltaMAE"),
        ("task", "missAvg", "DeltaCorr"),
    }
    for row in payload["paired_detail"]:
        if (row["family"], row["condition"], row["metric"]) in target_keys:
            label = (
                "mean D_cross"
                if row["metric"] == "D_cross"
                else f"missAvg {row['metric']}"
            )
            lines.append(
                f"| {row['seed']} | {label} | {row['H0']:.6f} | "
                f"{row['H0_B3']:.6f} | {row['difference_B3_minus_H0']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "With only three paired seeds, this report does not compute p-values, "
            "confidence intervals, or standardized effect sizes. Seed-level "
            "directional consistency and paired mean +/- sample standard deviation "
            "are the authorized readouts. Lambda remains frozen and test-informed; "
            "this experiment is not lambda tuning.",
            "",
            "Detailed fixed-condition task metrics and pattern-level D_cross results "
            "are stored in each seed directory and in `summary.csv`/`paired_detail.csv`.",
            "",
        ]
    )
    (output_dir / "PAIRED_MULTISeed_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.audit_only:
        raise RuntimeError("CUDA is required for the full paired multi-seed run")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    datasets = build_mosei_datasets(
        cfg.mosei_data_path, cfg.mosei_conv_dir, inf_policy=cfg.mosei_inf_policy
    )
    dims = mosei_dims(datasets)
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = dims
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

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
    }
    config = {
        "dataset": "MOSEI",
        "protocol": PROTOCOL,
        "seeds": args.seeds,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "optimizer": "Adam",
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "dropout": cfg.dropout,
        "train_missing_probability": cfg.protocol_b_train_missing_prob,
        "lambda_cons": LAM_B3,
        "consistency": "normalized stop-gradient L2",
        "split_sizes": {k: len(v) for k, v in datasets.items()},
        "dims": list(dims),
        "sample_std_ddof": 1,
        "random_streams": {
            "order_seed": "seed + 100000",
            "task_mask_seed": "seed + global_step",
            "task_forward_seed": "seed + 200000 + global_step",
            "cons_mask_seed": "seed + 777777 + global_step",
            "full_forward_seed": "seed + 300000 + global_step",
            "miss_forward_seed": "seed + 400000 + global_step",
        },
    }
    (output_dir / "RUN_CONFIG.json").write_text(
        json.dumps({"environment": environment, "config": config}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"environment": environment, "config": config}, indent=2), flush=True)
    if args.audit_only:
        print("AUDIT_ONLY_PASS", flush=True)
        return

    seed_results = []
    run_started = time.time()
    for seed in args.seeds:
        cfg.seed = seed
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        models = {}
        for variant in ("H0", "B3"):
            model, train_info = train_paired_variant(
                variant, seed, cfg, datasets, device
            )
            fixed, dcross, median_pair_dist = evaluate_variant(
                model, test_loader, condition_masks, device
            )
            models[variant] = {
                "label": MODEL_LABELS[variant],
                "train": train_info,
                "fixed": fixed,
                "dcross": dcross,
                "median_pair_dist_full": median_pair_dist,
            }
            del model
            torch.cuda.empty_cache()

        audit = {
            "initial_state_match": models["H0"]["train"]["initial_state_sha256"]
            == models["B3"]["train"]["initial_state_sha256"],
            "train_order_match": models["H0"]["train"]["train_order_sha256"]
            == models["B3"]["train"]["train_order_sha256"],
            "task_mask_match": models["H0"]["train"]["task_mask_sha256"]
            == models["B3"]["train"]["task_mask_sha256"],
        }
        audit["paired_pass"] = all(audit.values())
        if not audit["paired_pass"]:
            raise RuntimeError(f"Pairing audit failed for seed {seed}: {audit}")

        seed_result = {"seed": seed, "pairing_audit": audit, "models": models}
        seed_results.append(seed_result)
        (seed_dir / "results.json").write_text(
            json.dumps(seed_result, indent=2), encoding="utf-8"
        )

        fixed_rows = []
        for condition in [name for name, _ in FIXED_TEST_CONDITIONS] + ["missAvg"]:
            row = {"condition": condition}
            for variant in ("H0", "B3"):
                for metric, value in models[variant]["fixed"][condition].items():
                    row[f"{variant}_{metric}"] = value
            for metric in ("MAE", "Corr", "DeltaMAE", "DeltaCorr"):
                if f"H0_{metric}" in row:
                    row[f"difference_{metric}_B3_minus_H0"] = (
                        row[f"B3_{metric}"] - row[f"H0_{metric}"]
                    )
            fixed_rows.append(row)
        write_csv(
            seed_dir / "fixed_metrics.csv",
            fixed_rows,
            list(dict.fromkeys(key for row in fixed_rows for key in row)),
        )

        dcross_rows = []
        for condition in MISS_CONDITIONS + ["mean(6)"]:
            h0 = models["H0"]["dcross"][condition]
            b3 = models["B3"]["dcross"][condition]
            diff = b3["D_cross"] - h0["D_cross"]
            dcross_rows.append(
                {
                    "condition": condition,
                    "H0_D_cross": h0["D_cross"],
                    "B3_D_cross": b3["D_cross"],
                    "difference_B3_minus_H0": diff,
                    "relative_change": diff / (abs(h0["D_cross"]) + 1e-12),
                    "H0_raw_drift_L2": h0["raw_drift_L2"],
                    "B3_raw_drift_L2": b3["raw_drift_L2"],
                }
            )
        write_csv(
            seed_dir / "dcross.csv", dcross_rows, list(dcross_rows[0].keys())
        )

    paired_detail, summary_rows = aggregate_rows(seed_results)
    primary = primary_summary(summary_rows)
    detail_fields = list(paired_detail[0].keys())
    summary_fields = list(summary_rows[0].keys())
    write_csv(output_dir / "paired_detail.csv", paired_detail, detail_fields)
    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)

    payload = {
        "status": "complete",
        "label": "paired multi-seed MOSEI H0 vs H0+B3",
        "environment": environment,
        "config": config,
        "seeds": args.seeds,
        "runtime_sec": time.time() - run_started,
        "seed_results": seed_results,
        "paired_detail": paired_detail,
        "summary": summary_rows,
        "primary_metrics": primary,
        "inference_policy": "descriptive only; n=3; no p-values or confidence intervals",
        "lambda_caveat": "lambda=0.005 is frozen and test-informed; no tuning performed",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    build_report(output_dir, payload)
    print("PAIRED_MULTISeed_COMPLETE", flush=True)
    for key, row in primary.items():
        print(
            f"{key}: H0={row['H0_mean']:.6f}+/-{row['H0_sample_std']:.6f} "
            f"B3={row['H0_B3_mean']:.6f}+/-{row['H0_B3_sample_std']:.6f} "
            f"diff={row['paired_difference_mean']:.6f}+/-"
            f"{row['paired_difference_sample_std']:.6f} "
            f"improved={row['B3_improved_seeds']}/{row['n_seeds']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
