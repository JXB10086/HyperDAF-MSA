"""
Dimension-Matched MOSEI 对照实验  (p7 · 用户裁决 ③乙)
--------------------------------------------------
目的: 检验 "MOSEI 上 H0 vs Mask 的 T-missing 结论" 是否受 A/V 信息维度差异影响。
  MOSI      : audio=5  vision=20 (A+V=25)
  Standard MOSEI : audio=74 vision=35 (A+V=109, 4.36x)  <- run_mosei.py 已完成 -> mosei_results.json
  DimMatch MOSEI : audio=5  vision=20 (A+V=25, PCA 仅 train 拟合)  <- 本脚本

做法: 用 Dimension-Matched 数据训练【同样】M0/M1 Baseline + M2 MaskAware + M3 H0 Hyper × Protocol A/B,
      评估 7 固定条件, 重点输出 T-missing 的 Standard vs DimMatch 对照, 回答用户的 结果1/结果2 分支。
边界: 不替代标准主实验; 不做 H1/H2/B3/DynamicFusion/MCAC/CCL/KL; 不做重诊断/t-SNE (仅 MAE/Corr 对照)。
运行:  python experiments/run_mosei_dimmatch.py   (需在 run_mosei.py 之后, 以便对照)
"""
import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets import build_mosei_dimmatch_datasets, mosei_dims              # noqa: E402
from configs.config import Config                                          # noqa: E402
from utils import generate_missing_mask, ExperimentRecorder                # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS                   # noqa: E402
from experiments.stage3_lib import train_stage3_model, evaluate_with_mask_s3  # noqa: E402

PROTOCOLS = ("A", "B")
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
SPECS = [
    {"key": "baseline", "type": "baseline", "label": "M0/M1 Baseline"},
    {"key": "mask_aware", "type": "mask_aware", "label": "M2 MaskAware"},
    {"key": "hyper", "type": "hyper", "label": "M3 H0 Hyper"},
]
LABEL = {s["key"]: s["label"] for s in SPECS}
DATASET_TAG = "MOSEI-DimMatch"


def _avg(seq):
    return float(np.mean(seq))


def build_table(results, keys, conditions, metric):
    rows = []
    for k in keys:
        for pr in PROTOCOLS:
            row = {"Model": LABEL[k], "Protocol": pr}
            for cond in conditions:
                row[str(cond)] = round(results[(k, pr, cond)][metric], 4)
            rows.append(row)
    return pd.DataFrame(rows)


def print_table(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def load_standard(mosei_dir):
    p = os.path.join(mosei_dir, "mosei_results.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def tmiss(std_fixed, dm_fixed, key, pr, metric):
    """取 (model,protocol) 在 T-missing 上的 metric; std_fixed 来自 json 的 'k|pr|c' 键。"""
    s = std_fixed.get(f"{key}|{pr}|T missing", {}).get(metric) if std_fixed else None
    d = dm_fixed.get((key, pr, "T missing"), {}).get(metric)
    return s, d


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.mosei_dir, exist_ok=True)

    dm, pca = build_mosei_dimmatch_datasets(inf_policy=cfg.mosei_inf_policy)
    Dt, Da, Dv, L = mosei_dims(dm)
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = Dt, Da, Dv, L

    print("=" * 100)
    print("Dimension-Matched MOSEI 对照 (p7): audio 74->5 / vision 35->20 (PCA 仅 train 拟合)")
    print(f"Device={device} | seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size}")
    print(f"dims: text={Dt} audio={Da} vision={Dv} L={L} | A+V={Da + Dv} (对齐 MOSI 25)")
    print(f"PCA EVR: audio={pca['audio']['evr'].sum():.4f} (各{np.round(pca['audio']['evr'], 4).tolist()}) "
          f"vision={pca['vision']['evr'].sum():.4f}")
    print(f"split: train={len(dm['train'])} valid={len(dm['valid'])} test={len(dm['test'])}")
    print("=" * 100)

    rec = ExperimentRecorder(cfg.mosei_dir)     # 追加进统一 experiment_records.csv/jsonl
    test_loader = DataLoader(dm["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(dm["test"])

    trained, infos = {}, {}
    for s in SPECS:
        for pr in PROTOCOLS:
            print(f"\n>>> [DimMatch] 训练 {s['label']} + Protocol {pr} ...", flush=True)
            hd = cfg.hyper_dim if s["type"] == "hyper" else None
            model, info = train_stage3_model(s["type"], pr, cfg, dm, device, hyper_dim=hd)
            trained[(s["key"], pr)] = model
            infos[(s["key"], pr)] = info
            print(f"    best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f} "
                  f"| #params={info['n_params']:,}", flush=True)
            torch.save({"model_state_dict": model.state_dict(), "spec_key": s["key"],
                        "model_type": s["type"], "protocol": pr, "variant": "dimmatch",
                        "best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                        "n_params": info["n_params"],
                        "pca_evr": {"audio": pca["audio"]["evr"].tolist(),
                                    "vision": pca["vision"]["evr"].tolist()}},
                       os.path.join(cfg.mosei_dir, f"dimmatch_{s['key']}_protocol{pr}.pt"))

    fixed = {}
    for s in SPECS:
        for pr in PROTOCOLS:
            info = infos[(s["key"], pr)]
            for cond_name, pattern in FIXED_TEST_CONDITIONS:
                mask = generate_missing_mask(n_test, missing_pattern=pattern)
                _, m = evaluate_with_mask_s3(trained[(s["key"], pr)], s["type"],
                                             test_loader, mask, device)
                fixed[(s["key"], pr, cond_name)] = m
                rec.log(dataset=DATASET_TAG, protocol=pr, model=LABEL[s["key"]], seed=cfg.seed,
                        missing_pattern=cond_name,
                        missing_probability=(0.0 if pr == "A" else cfg.protocol_b_train_missing_prob),
                        best_epoch=info["best_epoch"], n_params=info["n_params"],
                        MAE=m["MAE"], Corr=m["Corr"])

    keys = [s["key"] for s in SPECS]
    df_mae = build_table(fixed, keys, COND_NAMES, "MAE")
    df_corr = build_table(fixed, keys, COND_NAMES, "Corr")
    print_table("[DimMatch-MAE] 3 模型 × 7 固定条件 × Protocol A/B  (MAE ↓)", df_mae)
    print_table("[DimMatch-Corr] 3 模型 × 7 固定条件 × Protocol A/B  (Corr ↑)", df_corr)

    # ---------------- Standard vs DimMatch: T-missing 对照 ----------------
    std = load_standard(cfg.mosei_dir)
    std_fixed = std["fixed"] if std else None
    print("\n" + "=" * 100)
    print("[关键对照] Standard MOSEI (A+V=109) vs Dimension-Matched (A+V=25) —— T-missing")
    print("=" * 100)
    if std_fixed is None:
        print("  [警告] 未找到 mosei_results.json (标准主实验尚未完成) -> 仅输出 DimMatch 数值, 跳过对照。")
    cmp_rows = []
    for key in ("mask_aware", "hyper"):
        for pr in PROTOCOLS:
            for metric in ("MAE", "Corr"):
                sv, dv = tmiss(std_fixed, fixed, key, pr, metric)
                cmp_rows.append({"Model": LABEL[key], "Protocol": pr, "Metric": metric,
                                 "Standard": (round(sv, 4) if sv is not None else None),
                                 "DimMatch": (round(dv, 4) if dv is not None else None),
                                 "Δ(DM-Std)": (round(dv - sv, 4) if (sv is not None and dv is not None) else None)})
    dfc = pd.DataFrame(cmp_rows)
    print(dfc.to_string(index=False))

    # H0 vs Mask 的 T-missing 优势 (在两个版本上分别算), 回答 结果1/结果2
    verdict = {}
    for pr in PROTOCOLS:
        for metric, better in (("MAE", "lower"), ("Corr", "higher")):
            sm, shm = tmiss(std_fixed, fixed, "mask_aware", pr, metric)
            sh, dhh = tmiss(std_fixed, fixed, "hyper", pr, metric)
            dm_m = fixed[("mask_aware", pr, "T missing")][metric]
            dm_h = fixed[("hyper", pr, "T missing")][metric]
            # gap>0 表示 hyper 更优 (MAE: mask-hyper>0; Corr: hyper-mask>0)
            gap_dm = (dm_m - dm_h) if better == "lower" else (dm_h - dm_m)
            gap_std = None
            if sm is not None and sh is not None:
                gap_std = (sm - sh) if better == "lower" else (sh - sm)
            verdict[f"{pr}|{metric}"] = {"gap_standard": gap_std, "gap_dimmatch": gap_dm,
                                         "hyper_better_standard": (gap_std > 0 if gap_std is not None else None),
                                         "hyper_better_dimmatch": bool(gap_dm > 0)}
    print("\n[H0 vs Mask 的 T-missing 优势 gap>0 表示 H0 更优]")
    for k, v in verdict.items():
        gs = f"{v['gap_standard']:+.4f}" if v["gap_standard"] is not None else "NA"
        print(f"  {k:10s}: gap_Standard={gs} gap_DimMatch={v['gap_dimmatch']:+.4f} | "
              f"H0更优? Std={v['hyper_better_standard']} DM={v['hyper_better_dimmatch']}")

    if std_fixed is not None:
        print("\n[用户分支判定]")
        for pr in PROTOCOLS:
            vm, vc = verdict[f"{pr}|MAE"], verdict[f"{pr}|Corr"]
            std_win = (vm["hyper_better_standard"] or vc["hyper_better_standard"])
            dm_win = (vm["hyper_better_dimmatch"] or vc["hyper_better_dimmatch"])
            if std_win and dm_win:
                tag = "结果1: 标准与降维下 H0 均优于 Mask -> 优势不来自 A/V 信息量 (强结论)"
            elif std_win and not dm_win:
                tag = "结果2: 仅标准下 H0 优于 Mask, 降维后消失 -> 优势部分来自 A/V 容量"
            elif not std_win and dm_win:
                tag = "反常: 标准下 H0 不优, 降维后反而优 -> 需细看指标"
            else:
                tag = "两版本 H0 均不优于 Mask -> 与 MOSI 一致的 H0 弱势"
            print(f"  Protocol {pr}: {tag}")

    # ---------------- 落盘 ----------------
    df_mae.to_csv(os.path.join(cfg.mosei_dir, "fixed_mae_dimmatch.csv"), index=False)
    df_corr.to_csv(os.path.join(cfg.mosei_dir, "fixed_corr_dimmatch.csv"), index=False)
    dfc.to_csv(os.path.join(cfg.mosei_dir, "tmiss_standard_vs_dimmatch.csv"), index=False)
    dump = {
        "variant": "dimension_matched", "target": {"audio": Da, "vision": Dv, "text": Dt, "L": L},
        "pca_evr": {"audio": pca["audio"]["evr"].tolist(), "vision": pca["vision"]["evr"].tolist()},
        "pca_fit": "train_valid_frames_only (no leakage)", "inf_policy": cfg.mosei_inf_policy,
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size},
        "fixed": {f"{k}|{pr}|{c}": fixed[(k, pr, c)] for (k, pr, c) in fixed},
        "train_info": {f"{k}|{pr}": {"best_epoch": infos[(k, pr)]["best_epoch"],
                                     "best_valid_mae": infos[(k, pr)]["best_valid_mae"],
                                     "n_params": infos[(k, pr)]["n_params"]} for (k, pr) in infos},
        "tmiss_comparison": cmp_rows, "verdict": verdict,
    }
    with open(os.path.join(cfg.mosei_dir, "dimmatch_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 100)
    print(f"DimMatch 结果   : {os.path.join(cfg.mosei_dir, 'dimmatch_results.json')}")
    print(f"T-missing 对照表: {os.path.join(cfg.mosei_dir, 'tmiss_standard_vs_dimmatch.csv')}")
    print(f"统一实验记录    : {rec.csv} / {rec.jsonl}")
    print("=" * 100)


if __name__ == "__main__":
    main()
