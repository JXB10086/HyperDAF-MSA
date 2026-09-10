"""
Stage 4 · Round-2 (A 干预实验) · Query L2 Normalization on F3 (MOSEI · Protocol B · 30ep)
--------------------------------------------------
用户 2026-09-06 裁决: C 诊断完成 -> 进入 A。【只】对 F3 的 Query 做 L2 归一化:
    Q = W_Q ẑ_hyper,  ẑ = z_hyper/(‖z_hyper‖₂+ε);   融合残差仍用原始 z_hyper (h_fuse = z + Σ α_i V_i)。
单变量干预: 与 F3 raw 唯一差异 = query 是否归一化; 不引入 LayerNorm/MCAC/CCL/KL/MoE/Reliability。

本 runner 【只训练 F3n】, F0/F1/F2/F3_raw 全部沿用 stage4_results.json / stage4_alpha_F*.npy 已有结果,
不重跑 -> 干净的 F3_raw vs F3_norm 单变量对照 (+ F2 参考)。

逐 epoch 记录 (经 epoch_diag_hook, 仅 eval 前向不耗 RNG): ‖z‖ mean/max、z_std、H(α)、H_norm、max(α)、饱和率,
加上训练循环自带的 ‖zf‖/valid_MAE/valid_Corr -> 可画 图1(‖z‖) / 图2(H(α)) / 图3(MAE) raw vs norm。

三层通过标准:
  L1 数值层: ‖z‖ 不再持续爆炸; H(α) 不退化为 0; max(α)<1; 不再近 one-hot。
  L2 表征层: probe_corr / drift_T / eff_rank 不明显劣于 F3 raw。
  L3 任务层: F3n > F3_raw (尺度假设直接支持); 进一步 F3n >= F2 则核心创新2 成立。
运行:  python experiments/run_mosei_stage4_r2.py
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

from datasets import build_mosei_datasets, mosei_dims                        # noqa: E402
from configs.config import Config                                            # noqa: E402
from utils import generate_missing_mask, ExperimentRecorder                 # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS                    # noqa: E402
from experiments.stage3_lib import (evaluate_with_mask_s3,                  # noqa: E402
                                    extract_representation_s3)
from experiments.stage3c_lib import train_stage3c_consistency               # noqa: E402
from experiments.stage4_lib import (extract_alpha_s4, alpha_missing_audit,  # noqa: E402
                                    alpha_entropy_stats, fusion_epoch_diag,
                                    plot_alpha_heatmap, plot_r2_curves)
from experiments.run_mosei import diag_hyper                                # noqa: E402
from models.stage4_fusion import HyperGuidedNormFusion                      # noqa: E402

PROTOCOL = "B"
LAM_CONS = 0.005
SG = True
CONS_NORM = "l2"
KEY = "F3n"
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
AVAIL = {  # 每条件可用模态 (T,A,V), 与 COND_NAMES 顺序一致
    "None": [1, 1, 1], "T missing": [0, 1, 1], "A missing": [1, 0, 1],
    "V missing": [1, 1, 0], "T+A missing": [0, 0, 1],
    "T+V missing": [0, 1, 0], "A+V missing": [1, 0, 0],
}
REF_KEYS = ["F0", "F1", "F2", "F3"]          # 沿用已有结果的对照
VLABEL = {"F0": "F0 H0", "F1": "F1 MaskGate", "F2": "F2 GlobalQuery",
          "F3": "F3_raw HyperGuided", "F3n": "F3_norm QNorm"}
DATASET_TAG = "MOSEI-Stage4-R2-F3norm"


def _avg(seq):
    return float(np.mean(seq))


def print_table(title, df):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print(df.to_string(index=False))


def entropy_table_from_npy(path, key):
    """(7,N,3) α.npy -> 每条件熵/饱和行。"""
    arr = np.load(path)
    rows = []
    for j, cond in enumerate(COND_NAMES):
        st = alpha_entropy_stats(arr[j], AVAIL[cond])
        rows.append({"variant": key, "cond": cond,
                     "H_norm": round(st["H_norm_mean"], 4),
                     "max_α": round(st["max_alpha_mean"], 4),
                     "min_avail_α": round(st["min_avail_alpha_mean"], 4),
                     "frac_sat": round(st["frac_sat"], 4)})
    return rows


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.mosei_dir, exist_ok=True)
    prior_path = os.path.join(cfg.mosei_dir, "stage4_results.json")
    with open(prior_path, encoding="utf-8") as f:
        R = json.load(f)

    datasets = build_mosei_datasets(cfg.mosei_data_path, cfg.mosei_conv_dir,
                                    inf_policy=cfg.mosei_inf_policy)
    Dt, Da, Dv, L = mosei_dims(datasets)
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = Dt, Da, Dv, L

    print("=" * 104)
    print("Stage 4 · Round-2 (A): Query L2 Normalization on F3 · 只训 F3n · Protocol B · 30ep")
    print("  Q=W_Q ẑ, ẑ=z/(‖z‖₂+ε); 残差仍用原始 z; 不重跑 F0/F1/F2/F3_raw (沿用 stage4_results.json)")
    print(f"  consistency 封版保留: λ={LAM_CONS}, sg={SG}, norm={CONS_NORM} | seed={cfg.seed} "
          f"epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} | hyper D={cfg.hyper_dim}")
    print("=" * 104)

    rec = ExperimentRecorder(cfg.mosei_dir)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    valid_loader = DataLoader(datasets["valid"], batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)
    n_test, n_valid = len(datasets["test"]), len(datasets["valid"])
    none_mask = generate_missing_mask(n_test, missing_pattern="T+A+V")
    valid_none_mask = generate_missing_mask(n_valid, missing_pattern="T+A+V")   # 钩子用全可用 mask
    cond_mask = {name: generate_missing_mask(n_test, missing_pattern=pat)
                 for name, pat in FIXED_TEST_CONDITIONS}

    common = dict(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim, vision_dim=cfg.vision_dim,
                  hyper_dim=cfg.hyper_dim, hidden_dim=cfg.hyper_hidden_dim, dropout=cfg.dropout)
    hook = lambda m, l, d: fusion_epoch_diag(m, l, d, valid_none_mask)          # noqa: E731

    print(f"\n>>> [R2] 训练 {VLABEL[KEY]} + consistency(λ={LAM_CONS},sg,norm={CONS_NORM}) "
          f"+ Protocol {PROTOCOL} + 逐epoch诊断钩子 ...", flush=True)
    model, info = train_stage3c_consistency(PROTOCOL, cfg, datasets, device,
                                            lam_cons=LAM_CONS, hyper_dim=cfg.hyper_dim,
                                            stop_grad_full=SG, cons_norm=CONS_NORM,
                                            model_factory=lambda: HyperGuidedNormFusion(**common),
                                            epoch_diag_hook=hook, verbose=True)
    print(f"    best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f} "
          f"| #params={info['n_params']:,}", flush=True)
    torch.save({"model_state_dict": model.state_dict(), "variant": KEY,
                "model_type": "hyper_stage4_fusion_qnorm", "protocol": PROTOCOL,
                "lam_cons": LAM_CONS, "stop_grad_full": SG, "cons_norm": CONS_NORM,
                "best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                "n_params": info["n_params"]},
               os.path.join(cfg.mosei_dir, f"stage4_{KEY}_protocolB.pt"))

    # ---- 7 条件评估 + 记录 ----
    fixed = {}
    for cond_name, _ in FIXED_TEST_CONDITIONS:
        _, m = evaluate_with_mask_s3(model, "hyper", test_loader, cond_mask[cond_name], device)
        fixed[cond_name] = m
        rec.log(dataset=DATASET_TAG, protocol=PROTOCOL, model=VLABEL[KEY], seed=cfg.seed,
                missing_pattern=cond_name, missing_probability=cfg.protocol_b_train_missing_prob,
                best_epoch=info["best_epoch"], n_params=info["n_params"],
                MAE=m["MAE"], Corr=m["Corr"])

    # ---- z 诊断 + per-sample α ----
    zA = extract_representation_s3(model, "hyper", test_loader, none_mask, device)
    Zs = [extract_representation_s3(model, "hyper", test_loader, cond_mask[c], device)["rep"]
          for c in COND_NAMES]
    diag_n = diag_hyper(zA["rep"], np.stack(Zs, axis=0), zA["label"], COND_NAMES)
    zstd_n = float(np.std(zA["rep"]))
    print(f"    [diag {KEY}] probe_corr={diag_n['probe_corr']:+.4f} eff_rank={diag_n['eff_rank']:.3f} "
          f"z_std={zstd_n:.4f} drift_T={diag_n['per_pattern']['T missing']['drift_ratio']:.4f} "
          f"transfer_T={diag_n['per_pattern']['T missing']['transfer']:+.4f}", flush=True)

    alpha_by, mean_mat, std_mat, audit_rows = {}, [], [], []
    for cond_name, _ in FIXED_TEST_CONDITIONS:
        out = extract_alpha_s4(model, test_loader, cond_mask[cond_name], device)
        a = out["alpha"]
        alpha_by[cond_name] = a
        mean_mat.append(a.mean(axis=0)); std_mat.append(a.std(axis=0))
        aud = alpha_missing_audit(a, np.asarray(cond_mask[cond_name][0]))
        audit_rows.append({"variant": KEY, "cond": cond_name,
                           "max_missing_α": round(aud["max_missing_alpha"], 8),
                           "mean_α_T": round(float(a[:, 0].mean()), 4),
                           "mean_α_A": round(float(a[:, 1].mean()), 4),
                           "mean_α_V": round(float(a[:, 2].mean()), 4)})
    alpha_mean_n = np.array(mean_mat); alpha_std_n = np.array(std_mat)
    np.save(os.path.join(cfg.mosei_dir, f"stage4_alpha_{KEY}.npy"),
            np.stack([alpha_by[c] for c in COND_NAMES], axis=0))
    worst = max(r["max_missing_α"] for r in audit_rows)
    print(f"    [α {KEY}] None: α_T/A/V={alpha_mean_n[0].round(3)} | 缺失归零 max_α_missing={worst:.2e}")

    # ---- 载入 prior (F3_raw / F2 / F0 / F1) ----
    hist_raw = R["train_info"]["F3"]["history"]
    hist_n = info["history"]
    fixed_prior = {k: {c: R["fixed"][f"{k}|{c}"] for c in COND_NAMES} for k in REF_KEYS}
    diag_prior = R["diag"]; zstd_prior = R["z_std"]
    missavg_prior = {k: R["missavg"][k] for k in REF_KEYS}
    raw_npy = os.path.join(cfg.mosei_dir, "stage4_alpha_F3.npy")
    raw_ent = entropy_table_from_npy(raw_npy, "F3") if os.path.exists(raw_npy) else []
    n_ent = entropy_table_from_npy(os.path.join(cfg.mosei_dir, f"stage4_alpha_{KEY}.npy"), KEY)
    # raw best-epoch H(α) (None 条件, 原始熵) 作为图2 参考线
    raw_ref_H = alpha_entropy_stats(np.load(raw_npy)[0], AVAIL["None"])["H_mean"]

    # ---- 表1: 7 条件 MAE/Corr (F3n vs F3raw vs F2 vs F0/F1) ----
    def row_mae_corr():
        rows = []
        for k in REF_KEYS:
            rows.append({"variant": VLABEL[k],
                         **{c: round(fixed_prior[k][c]["MAE"], 4) for c in COND_NAMES},
                         "missAvg": round(missavg_prior[k]["MAE"], 4)})
        rows.append({"variant": VLABEL[KEY],
                     **{c: round(fixed[c]["MAE"], 4) for c in COND_NAMES},
                     "missAvg": round(_avg([fixed[c]["MAE"] for c in MISS]), 4)})
        return pd.DataFrame(rows)
    print_table("[R2 MAE] Protocol B × 7 条件 (↓, 含 missAvg): F3_norm vs F3_raw vs 对照", row_mae_corr())

    def row_corr():
        rows = []
        for k in REF_KEYS:
            rows.append({"variant": VLABEL[k],
                         **{c: round(fixed_prior[k][c]["Corr"], 4) for c in COND_NAMES},
                         "missAvg": round(missavg_prior[k]["Corr"], 4)})
        rows.append({"variant": VLABEL[KEY],
                     **{c: round(fixed[c]["Corr"], 4) for c in COND_NAMES},
                     "missAvg": round(_avg([fixed[c]["Corr"] for c in MISS]), 4)})
        return pd.DataFrame(rows)
    print_table("[R2 Corr] Protocol B × 7 条件 (↑, 含 missAvg)", row_corr())

    # ---- 表2: 汇总/诊断 (F3n vs F3raw vs F2) ----
    srows = []
    for k in ["F2", "F3"]:
        d = diag_prior[k]; pt = d["per_pattern"]["T missing"]
        srows.append({"variant": VLABEL[k], "best_ep": R["train_info"][k]["best_epoch"],
                      "validMAE": round(R["train_info"][k]["best_valid_mae"], 4),
                      "probe_corr": round(d["probe_corr"], 4), "eff_rank": round(d["eff_rank"], 3),
                      "z_std": round(zstd_prior[k], 4), "drift_T": round(pt["drift_ratio"], 4),
                      "missMAE": round(missavg_prior[k]["MAE"], 4),
                      "missCorr": round(missavg_prior[k]["Corr"], 4)})
    ptn = diag_n["per_pattern"]["T missing"]
    srows.append({"variant": VLABEL[KEY], "best_ep": info["best_epoch"],
                  "validMAE": round(info["best_valid_mae"], 4),
                  "probe_corr": round(diag_n["probe_corr"], 4), "eff_rank": round(diag_n["eff_rank"], 3),
                  "z_std": round(zstd_n, 4), "drift_T": round(ptn["drift_ratio"], 4),
                  "missMAE": round(_avg([fixed[c]["MAE"] for c in MISS]), 4),
                  "missCorr": round(_avg([fixed[c]["Corr"] for c in MISS]), 4)})
    df_sum = pd.DataFrame(srows)
    print_table("[R2 汇总] F3_norm vs F3_raw vs F2 · 选模/表征诊断/缺失均值", df_sum)

    # ---- 表3: best-epoch α 熵/饱和 (F3n vs F3raw, 7 条件) ----
    df_ent = pd.DataFrame(raw_ent + n_ent)
    print_table("[R2 α 熵] best-epoch H_norm/max(α)/饱和率 (0=one-hot): F3_raw vs F3_norm", df_ent)

    # ---- 表4: 逐 epoch 数值层 (F3n) + 与 raw 对照 ----
    zf_n = np.array([h["norm_full"] for h in hist_n])
    zf_r = np.array([h["norm_full"] for h in hist_raw])
    be_n = info["best_epoch"]
    hbe = hist_n[be_n - 1]
    t4 = pd.DataFrame([
        {"量": "‖z‖ ep1-9 mean", "F3_raw": round(zf_r[:9].mean(), 2), "F3_norm": round(zf_n[:9].mean(), 2)},
        {"量": "‖z‖ 全程 max", "F3_raw": round(zf_r.max(), 2), "F3_norm": round(zf_n.max(), 2)},
        {"量": "‖z ep30", "F3_raw": round(zf_r[-1], 2), "F3_norm": round(zf_n[-1], 2)},
        {"量": "max/ep1-9 倍数", "F3_raw": round(zf_r.max() / zf_r[:9].mean(), 1),
         "F3_norm": round(zf_n.max() / zf_n[:9].mean(), 1)},
        {"量": "best-ep z_std(valid,None)", "F3_raw": round(zstd_prior["F3"], 4), "F3_norm": round(hbe.get("diag_z_std", float('nan')), 4)},
        {"量": "best-ep H(α)(valid,None)", "F3_raw": round(raw_ref_H, 4), "F3_norm": round(hbe.get("diag_H_alpha", float('nan')), 4)},
        {"量": "best-ep max(α)", "F3_raw": round(next(r['max_α'] for r in raw_ent if r['cond'] == 'None'), 4),
         "F3_norm": round(hbe.get("diag_max_alpha", float('nan')), 4)},
        {"量": "best-ep 饱和率", "F3_raw": round(next(r['frac_sat'] for r in raw_ent if r['cond'] == 'None'), 4),
         "F3_norm": round(hbe.get("diag_frac_sat", float('nan')), 4)},
    ])
    print_table("[R2 数值层] F3_norm 是否消除尺度爆炸/α 饱和 (vs F3_raw)", t4)

    # ---- 三层判定 ----
    miss_n = _avg([fixed[c]["MAE"] for c in MISS]); miss_r = missavg_prior["F3"]["MAE"]
    miss_f2 = missavg_prior["F2"]["MAE"]
    corr_n = _avg([fixed[c]["Corr"] for c in MISS]); corr_r = missavg_prior["F3"]["Corr"]
    corr_f2 = missavg_prior["F2"]["Corr"]
    z_bounded = (zf_n.max() / zf_n[:9].mean()) < 5.0 and zf_n.max() < 0.5 * zf_r.max()
    h_ok = hbe.get("diag_H_alpha", 0.0) > 0.05 and hbe.get("diag_max_alpha", 1.0) < 0.99
    sat_ok = hbe.get("diag_frac_sat", 1.0) < 0.9
    probe_ok = diag_n["probe_corr"] >= 0.95 * diag_prior["F3"]["probe_corr"]
    drift_ok = ptn["drift_ratio"] <= 1.10 * diag_prior["F3"]["per_pattern"]["T missing"]["drift_ratio"]
    rank_ok = diag_n["eff_rank"] >= 0.90 * diag_prior["F3"]["eff_rank"]
    task_gt_raw = miss_n < miss_r
    task_ge_f2 = miss_n <= miss_f2
    print("\n" + "=" * 104)
    print("[R2 三层判定]")
    print("=" * 104)
    print(f"  L1 数值层: ‖z有界(倍数<5 且 <0.5×raw峰值):{'✓' if z_bounded else '✗'} "
          f"(F3n max={zf_n.max():.1f}, 倍数={zf_n.max()/zf_n[:9].mean():.1f} | raw max={zf_r.max():.1f})")
    print(f"             H(α)>0.05 且 max(α)<0.99 :{'✓' if h_ok else '✗'} "
          f"(H={hbe.get('diag_H_alpha',0):.4f}, maxα={hbe.get('diag_max_alpha',1):.4f}) | "
          f"饱和率<0.9 :{'✓' if sat_ok else '✗'} ({hbe.get('diag_frac_sat',1):.3f})")
    print(f"  L2 表征层: probe≥0.95×raw:{'✓' if probe_ok else '✗'} "
          f"({diag_n['probe_corr']:.4f} vs {diag_prior['F3']['probe_corr']:.4f}) | "
          f"drift≤1.1×raw:{'✓' if drift_ok else '✗'} | eff_rank≥0.9×raw:{'✓' if rank_ok else '✗'} "
          f"({diag_n['eff_rank']:.3f} vs {diag_prior['F3']['eff_rank']:.3f})")
    print(f"  L3 任务层: F3n>F3_raw(missMAE↓):{'✓' if task_gt_raw else '✗'} "
          f"({miss_n:.4f} vs {miss_r:.4f}) | F3n≥F2:{'✓' if task_ge_f2 else '✗'} ({miss_n:.4f} vs {miss_f2:.4f})")
    print(f"             missCorr: F3n={corr_n:.4f} raw={corr_r:.4f} F2={corr_f2:.4f}")

    # ---- 可视化 ----
    png_curves = os.path.join(cfg.mosei_dir, "stage4_r2_curves.png")
    plot_r2_curves(hist_raw, hist_n, png_curves, raw_ref_H=raw_ref_H)
    print(f"\n  三联曲线图: 已保存 {png_curves}")
    png_hm = os.path.join(cfg.mosei_dir, "stage4_r2_alpha_heatmap.png")
    ok = plot_alpha_heatmap({"F3_raw": np.array(R["alpha_mean"]["F3"]),
                             "F2": np.array(R["alpha_mean"]["F2"]),
                             "F3_norm": alpha_mean_n},
                            {"F3_raw": np.array(R["alpha_std"]["F3"]),
                             "F2": np.array(R["alpha_std"]["F2"]),
                             "F3_norm": alpha_std_n}, COND_NAMES, png_hm)
    print(f"  α 热力图(F3_raw/F2/F3_norm): {'已保存 ' + png_hm if ok else '跳过'}")

    # ---- 落盘 ----
    row_mae_corr().to_csv(os.path.join(cfg.mosei_dir, "stage4_r2_mae.csv"), index=False, encoding="utf-8-sig")
    row_corr().to_csv(os.path.join(cfg.mosei_dir, "stage4_r2_corr.csv"), index=False, encoding="utf-8-sig")
    df_sum.to_csv(os.path.join(cfg.mosei_dir, "stage4_r2_summary.csv"), index=False, encoding="utf-8-sig")
    df_ent.to_csv(os.path.join(cfg.mosei_dir, "stage4_r2_alpha_entropy.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(hist_n).to_csv(os.path.join(cfg.mosei_dir, "stage4_r2_F3n_history.csv"),
                                index=False, encoding="utf-8-sig")
    dump = {
        "stage": "4-round2", "variant": KEY, "intervention": "Q=W_Q(z/‖z‖₂+ε), residual=raw z",
        "protocol": PROTOCOL, "lam_cons": LAM_CONS, "stop_grad_full": SG, "cons_norm": CONS_NORM,
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr,
                   "batch_size": cfg.batch_size, "hyper_dim": cfg.hyper_dim},
        "fixed_F3n": {c: fixed[c] for c in COND_NAMES},
        "missavg_F3n": {"MAE": miss_n, "Corr": corr_n},
        "diag_F3n": diag_n, "z_std_F3n": zstd_n,
        "alpha_mean_F3n": alpha_mean_n.tolist(), "alpha_audit_F3n": audit_rows,
        "alpha_entropy_F3n": n_ent, "alpha_entropy_F3raw": raw_ent,
        "verdict": {"L1_z_bounded": bool(z_bounded), "L1_H_ok": bool(h_ok), "L1_sat_ok": bool(sat_ok),
                    "L2_probe_ok": bool(probe_ok), "L2_drift_ok": bool(drift_ok), "L2_rank_ok": bool(rank_ok),
                    "L3_gt_raw": bool(task_gt_raw), "L3_ge_F2": bool(task_ge_f2)},
        "train_info_F3n": {"best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                           "n_params": info["n_params"], "history": hist_n},
    }
    with open(os.path.join(cfg.mosei_dir, "stage4_r2_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)
    print("\n" + "=" * 104)
    print(f"R2 结果 : {os.path.join(cfg.mosei_dir, 'stage4_r2_results.json')}")
    print(f"曲线/热力: stage4_r2_curves.png / stage4_r2_alpha_heatmap.png | 历史: stage4_r2_F3n_history.csv")
    print(f"统一记录: {rec.csv} / {rec.jsonl}")
    print("=" * 104)


if __name__ == "__main__":
    main()
