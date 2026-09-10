"""
Stage 3C 重跑 (MOSEI · Stop-Gradient 一致性 · 仅 Protocol B · λ 邻域 {0.005,0.01,0.02})
--------------------------------------------------
用户 2026-09-05 裁决 (方案1 Normalized Stop-Gradient): L_cons = ‖ sg(ẑ_full) − ẑ_miss ‖₂², ẑ=z/(‖z‖₂+ε)。
演进: ① 原始对称 ‖z_full−z_miss‖² 被 "z→0" 平凡最小化 (z_std 1.22→0.05, eff_rank→1.0);
      ② 未归一化 sg 版 ‖sg(z_full)−z_miss‖² 又因 "共享在线编码器使目标棘轮上抬" 发散爆炸 (实测 ‖z‖→3.5e9, cons→1e18, Corr→0);
      ③ 最终 = 停梯度(只更新 z_miss) + L2 单位化(去幅度投机, SimSiam 式), 实测稳定 (cons 0.44→0.28, Corr≈0.39, z_std≈0.62)。
  full 表征作稳定锚点(只由任务损失维护), 一致性梯度只把 z_miss 往锚点方向拉, 杜绝两者一起缩崩或爆炸。

本轮范围 (刻意最小化, 减少变量):
  - 仅 Protocol B (PA 训练不见缺失, 一致性必坍缩; 且 PB 是 Hyper 更可信的战场)。
  - λ_cons ∈ {0.005,0.01,0.02} 邻域 (对照 λ=0=H0); 单点 0.01 已通过五门槛, 本轮做超参敏感性确认。
  - View1=完整[1,1,1]; View2=单模态缺失(T/A/V 均匀采样); all-missing/双模态不参与。
  - H0 结构 / seed=42 / epochs=30 / bs=32 / lr=1e-3 / 划分 / 仅 valid MAE 选模 / test 仅评估, 全部不变。

必报 (用户指令): MAE/Corr + probe Corr/R² + sentiment silhouette + effective rank + z_std
  + T/A/V missing drift + None→missing transfer Corr + 单独保存 ‖z_full‖/‖z_miss‖/‖z_full−z_miss‖。
五道硬门槛 (对照本轮 λ=0 的 H0):
  ① eff_rank ≥ 1.5   ② z_std 不出现数量级坍缩 (≥ 0.1×H0)   ③ drift_T < H0(0.90)
  ④ transfer_T > H0(0.139)   ⑤ 缺失均值 MAE ≤ H0 + ε
运行:  python experiments/run_mosei_b3_sg.py
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
from utils import generate_missing_mask, ExperimentRecorder                  # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS                     # noqa: E402
from experiments.stage3_lib import (evaluate_with_mask_s3,                   # noqa: E402
                                    extract_representation_s3)
from experiments.stage3c_lib import train_stage3c_consistency               # noqa: E402
from experiments.run_mosei import diag_hyper                                # noqa: E402

PROTOCOL = "B"                       # 本轮仅 Protocol B
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
SINGLE_MISS = ["T missing", "A missing", "V missing"]
LAMBDAS = [0.0, 0.005, 0.01, 0.02]   # λ=0 为 H0 对照(锚点); 0.005/0.01/0.02 为 normalized sg-L2 邻域
SG_FOR = {0.0: False, 0.005: True, 0.01: True, 0.02: True}   # λ>0 启用 stop-gradient
CONS_NORM = "l2"                     # 单位化 ẑ=z/‖z‖ (raw sg-L2 会发散爆炸, 见 docstring); λ=0 不启用一致性, 无影响
EPS_MAE = 0.005                      # 门槛⑤容差
RANK_FLOOR = 1.5                     # 门槛①
ZSTD_FLOOR_RATIO = 0.1               # 门槛②: z_std ≥ 0.1×H0 (不得缩超 1 个数量级)
DATASET_TAG = "MOSEI-B3-sg"


def lab(lam):
    return "H0 (w/o cons)" if lam == 0.0 else f"H0+nsgL2 λ={lam}"


def _avg(seq):
    return float(np.mean(seq))


def three_norm_table(Z_by_pat):
    """从 PB 模型的 7 模式表征计算 ‖z_full‖/‖z_miss‖/‖z_full−z_miss‖ (逐样本 L2, 再对样本求均值)。
    Z_by_pat[0]=None=z_full。用于判断一致性靠 '真正靠近'(‖diff‖↓ 而 ‖z‖ 保持) 还是 '整体缩小'(三者齐↓)。"""
    Znone = Z_by_pat[0]
    n_full = float(np.linalg.norm(Znone, axis=1).mean())
    rows = []
    for p, name in enumerate(COND_NAMES):
        if name == "None":
            continue
        Zp = Z_by_pat[p]
        rows.append({"pattern": name,
                     "‖z_full‖": round(n_full, 4),
                     "‖z_miss‖": round(float(np.linalg.norm(Zp, axis=1).mean()), 4),
                     "‖z_full−z_miss‖": round(float(np.linalg.norm(Znone - Zp, axis=1).mean()), 4)})
    return pd.DataFrame(rows)


def print_table(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.mosei_dir, exist_ok=True)

    datasets = build_mosei_datasets(cfg.mosei_data_path, cfg.mosei_conv_dir,
                                    inf_policy=cfg.mosei_inf_policy)
    Dt, Da, Dv, L = mosei_dims(datasets)
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = Dt, Da, Dv, L

    print("=" * 100)
    print("Stage 3C 重跑 (MOSEI): Normalized Stop-Gradient 一致性 L_cons=‖sg(ẑ_full)−ẑ_miss‖² (ẑ=z/‖z‖) · 仅 Protocol B · λ 邻域 {0.005,0.01,0.02}")
    print(f"Device={device} | seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} | hyper D={cfg.hyper_dim}")
    print(f"dims: text={Dt} audio={Da} vision={Dv} L={L} | split train={len(datasets['train'])} "
          f"valid={len(datasets['valid'])} test={len(datasets['test'])}")
    print(f"λ={LAMBDAS} (λ=0=H0对照) | 一致性视图=单模态缺失(T/A/V) | 门槛: eff_rank≥{RANK_FLOOR}, "
          f"z_std≥{ZSTD_FLOOR_RATIO}×H0, drift_T↓, transfer_T↑, missMAE≤H0+{EPS_MAE}")
    print("=" * 100)

    rec = ExperimentRecorder(cfg.mosei_dir)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])
    none_mask = generate_missing_mask(n_test, missing_pattern="T+A+V")

    models, infos, fixed, diags, zstds, zby = {}, {}, {}, {}, {}, {}
    for lam in LAMBDAS:
        sg = SG_FOR[lam]
        print(f"\n>>> [B3-sg] 训练 H0 + {'nsg-' if sg else ''}L_cons (λ={lam}, norm={CONS_NORM}) + Protocol {PROTOCOL} ...", flush=True)
        model, info = train_stage3c_consistency(PROTOCOL, cfg, datasets, device,
                                                lam_cons=lam, hyper_dim=cfg.hyper_dim,
                                                stop_grad_full=sg, cons_norm=CONS_NORM, verbose=True)
        models[lam] = model
        infos[lam] = info
        print(f"    best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f} "
              f"| #params={info['n_params']:,}", flush=True)
        torch.save({"model_state_dict": model.state_dict(), "model_type": "hyper_b3",
                    "protocol": PROTOCOL, "lam_cons": lam, "stop_grad_full": sg,
                    "cons_norm": CONS_NORM, "variant": "b3_sg",
                    "best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                    "n_params": info["n_params"]},
                   os.path.join(cfg.mosei_dir, f"b3sg_lam{lam}_protocolB.pt"))

        # ---- 7 固定条件评估 (PB) + 记录 ----
        for cond_name, pattern in FIXED_TEST_CONDITIONS:
            mask = generate_missing_mask(n_test, missing_pattern=pattern)
            _, m = evaluate_with_mask_s3(model, "hyper", test_loader, mask, device)
            fixed[(lam, cond_name)] = m
            rec.log(dataset=DATASET_TAG, protocol=PROTOCOL, model=lab(lam), seed=cfg.seed,
                    missing_pattern=cond_name, missing_probability=cfg.protocol_b_train_missing_prob,
                    best_epoch=info["best_epoch"], n_params=info["n_params"],
                    MAE=m["MAE"], Corr=m["Corr"])

        # ---- PB 模型表征诊断 (质量 + drift/transfer 全部来自同一 PB 模型, 修正此前 PA 盲区) ----
        zA = extract_representation_s3(model, "hyper", test_loader, none_mask, device)
        zs = []
        for cond_name, pattern in FIXED_TEST_CONDITIONS:
            mm = generate_missing_mask(n_test, missing_pattern=pattern)
            o = extract_representation_s3(model, "hyper", test_loader, mm, device)
            zs.append(o["rep"])
        Z_by_pat = np.stack(zs, axis=0)
        y = zA["label"]
        diags[lam] = diag_hyper(zA["rep"], Z_by_pat, y, COND_NAMES)
        zstds[lam] = float(np.std(zA["rep"]))
        zby[lam] = Z_by_pat
        d = diags[lam]
        print(f"    [diag λ={lam}] probe_corr={d['probe_corr']:+.4f} R2={d['probe_r2']:+.4f} "
              f"eff_rank={d['eff_rank']:.3f} sil={d['sil_binary']:+.4f} z_std={zstds[lam]:.4f} "
              f"drift_T={d['per_pattern']['T missing']['drift_ratio']:.4f} "
              f"transfer_T={d['per_pattern']['T missing']['transfer']:+.4f}", flush=True)

    # ---- 表1/2: 固定条件 MAE / Corr (PB) ----
    def fixed_table(metric):
        rows = []
        for lam in LAMBDAS:
            row = {"λ_cons": lam}
            for cond in COND_NAMES:
                row[str(cond)] = round(fixed[(lam, cond)][metric], 4)
            rows.append(row)
        return pd.DataFrame(rows)
    print_table("[B3-sg MAE] Protocol B × 7 固定条件  (MAE ↓)", fixed_table("MAE"))
    print_table("[B3-sg Corr] Protocol B × 7 固定条件  (Corr ↑)", fixed_table("Corr"))

    # ---- 表3: 诊断汇总 (H0 vs sg-L2) ----
    drows = []
    for lam in LAMBDAS:
        d = diags[lam]
        pt, pa, pv = d["per_pattern"]["T missing"], d["per_pattern"]["A missing"], d["per_pattern"]["V missing"]
        drows.append({"λ_cons": lam, "probe_corr": round(d["probe_corr"], 4),
                      "probe_r2": round(d["probe_r2"], 4), "sil": round(d["sil_binary"], 4),
                      "eff_rank": round(d["eff_rank"], 3), "z_std": round(zstds[lam], 4),
                      "drift_T": round(pt["drift_ratio"], 4), "transfer_T": round(pt["transfer"], 4),
                      "drift_A": round(pa["drift_ratio"], 4), "transfer_A": round(pa["transfer"], 4),
                      "drift_V": round(pv["drift_ratio"], 4), "transfer_V": round(pv["transfer"], 4)})
    df_diag = pd.DataFrame(drows)
    print_table("[B3-sg 诊断] H0(λ=0) vs nsg-L2 λ∈{0.005,0.01,0.02} · 全部取自 Protocol B 模型", df_diag)

    # ---- 表4: 三范数 (靠近 vs 缩小) ----
    for lam in LAMBDAS:
        print_table(f"[三范数 λ={lam}] ‖z_full‖/‖z_miss‖/‖z_full−z_miss‖ (Protocol B, test)",
                    three_norm_table(zby[lam]))

    # ---- 五道硬门槛判定: 每个 consistency λ 对照 H0(λ=0) ----
    base = 0.0
    d0 = diags[base]
    missavg0 = _avg([fixed[(base, c)]["MAE"] for c in MISS])
    drift0 = d0["per_pattern"]["T missing"]["drift_ratio"]
    tr0 = d0["per_pattern"]["T missing"]["transfer"]
    CONS_LAMS = [l for l in LAMBDAS if l > 0.0]
    gate_rows, gates_by_lam = [], {}
    for tgt in CONS_LAMS:
        d1 = diags[tgt]
        missavg1 = _avg([fixed[(tgt, c)]["MAE"] for c in MISS])
        drift1 = d1["per_pattern"]["T missing"]["drift_ratio"]
        tr1 = d1["per_pattern"]["T missing"]["transfer"]
        g1 = d1["eff_rank"] >= RANK_FLOOR
        g2 = zstds[tgt] >= ZSTD_FLOOR_RATIO * zstds[base]
        g3 = drift1 < drift0
        g4 = tr1 > tr0
        g5 = missavg1 <= missavg0 + EPS_MAE
        passed = g1 and g2 and g3 and g4 and g5
        gates_by_lam[tgt] = {"g1_rank": bool(g1), "g2_zstd": bool(g2), "g3_drift": bool(g3),
                             "g4_transfer": bool(g4), "g5_mae": bool(g5), "passed": bool(passed),
                             "eff_rank": d1["eff_rank"], "z_std": zstds[tgt], "drift_T": drift1,
                             "transfer_T": tr1, "missavg_mae": missavg1}
        gate_rows.append({"λ_cons": tgt,
                          "①rank≥1.5": f"{d1['eff_rank']:.3f}{'✓' if g1 else '✗'}",
                          "②zstd≥.1H0": f"{zstds[tgt]:.3f}{'✓' if g2 else '✗'}",
                          "③driftT<H0": f"{drift1:.4f}{'✓' if g3 else '✗'}",
                          "④transT>H0": f"{tr1:+.4f}{'✓' if g4 else '✗'}",
                          "⑤missMAE": f"{missavg1:.4f}{'✓' if g5 else '✗'}",
                          "综合": "通过" if passed else "未通过"})

    print("\n" + "=" * 100)
    print(f"[五道硬门槛] 各 λ 对照 H0(λ=0) · Protocol B   "
          f"(H0: eff_rank={d0['eff_rank']:.3f} z_std={zstds[base]:.4f} drift_T={drift0:.4f} "
          f"transfer_T={tr0:+.4f} missMAE={missavg0:.4f})")
    print("=" * 100)
    print(pd.DataFrame(gate_rows).to_string(index=False))

    # ---- λ 邻域趋势 (核心问题: T_drift↓ / T_transfer↑ / transfer_A 是系统性 trade-off 还是单 seed 噪声) ----
    trend_rows = []
    for lam in LAMBDAS:
        d = diags[lam]
        pt, pa = d["per_pattern"]["T missing"], d["per_pattern"]["A missing"]
        trend_rows.append({"λ_cons": lam,
                           "T_drift": round(pt["drift_ratio"], 4),
                           "T_transfer": round(pt["transfer"], 4),
                           "transfer_A": round(pa["transfer"], 4),
                           "drift_A": round(pa["drift_ratio"], 4),
                           "eff_rank": round(d["eff_rank"], 3),
                           "z_std": round(zstds[lam], 4),
                           "probe_corr": round(d["probe_corr"], 4),
                           "missMAE": round(_avg([fixed[(lam, c)]["MAE"] for c in MISS]), 4)})
    df_trend = pd.DataFrame(trend_rows)
    print_table("[λ 邻域趋势] 焦点: T_drift(期望↓) · T_transfer(期望↑) · transfer_A(观察 H0 0.493→?)", df_trend)

    n_pass = sum(1 for l in CONS_LAMS if gates_by_lam[l]["passed"])
    pass_lams = [l for l in CONS_LAMS if gates_by_lam[l]["passed"]]
    print(f"\n  >>> λ 邻域: {n_pass}/{len(CONS_LAMS)} 个 consistency λ 通过全部五门槛 (通过点: {pass_lams}) <<<")

    # ---- 落盘 ----
    df_diag.to_csv(os.path.join(cfg.mosei_dir, "b3sg_diagnostics.csv"), index=False)
    fixed_table("MAE").to_csv(os.path.join(cfg.mosei_dir, "b3sg_fixed_mae.csv"), index=False)
    fixed_table("Corr").to_csv(os.path.join(cfg.mosei_dir, "b3sg_fixed_corr.csv"), index=False)
    for lam in LAMBDAS:
        three_norm_table(zby[lam]).to_csv(
            os.path.join(cfg.mosei_dir, f"b3sg_threenorm_lam{lam}.csv"), index=False)
    dump = {
        "stage": "3C", "variant": "normalized_stop_gradient_consistency_B3", "dataset": "MOSEI",
        "protocol": PROTOCOL, "cons_norm": CONS_NORM,
        "loss": "L_total = L_task + lam*mean||sg(zhat_full)-zhat_miss||_2^2, zhat=z/(||z||_2+eps)",
        "cons_view": "single_modality_missing (T/A/V); all-missing & double excluded",
        "lambdas": LAMBDAS, "gates": {"rank_floor": RANK_FLOOR, "zstd_floor_ratio": ZSTD_FLOOR_RATIO,
                                      "eps_mae": EPS_MAE},
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size,
                   "dims": [Dt, Da, Dv, L], "hyper_dim": cfg.hyper_dim,
                   "protocol_b_train_missing_prob": cfg.protocol_b_train_missing_prob},
        "fixed": {f"{lam}|{c}": fixed[(lam, c)] for (lam, c) in fixed},
        "missavg_mae": {str(lam): _avg([fixed[(lam, c)]["MAE"] for c in MISS]) for lam in LAMBDAS},
        "z_std": {str(lam): zstds[lam] for lam in LAMBDAS},
        "diag": {str(lam): diags[lam] for lam in LAMBDAS},
        "three_norms": {str(lam): three_norm_table(zby[lam]).to_dict(orient="records") for lam in LAMBDAS},
        "train_info": {str(lam): {"best_epoch": infos[lam]["best_epoch"],
                                  "best_valid_mae": infos[lam]["best_valid_mae"],
                                  "n_params": infos[lam]["n_params"],
                                  "history": infos[lam]["history"]} for lam in LAMBDAS},
        "gates_result": {"H0": {"eff_rank": d0["eff_rank"], "z_std": zstds[base], "drift_T": drift0,
                                "transfer_T": tr0, "missavg_mae": missavg0},
                         "by_lambda": {str(l): gates_by_lam[l] for l in CONS_LAMS}},
        "lambda_trend": df_trend.to_dict(orient="records"),
    }
    with open(os.path.join(cfg.mosei_dir, "b3sg_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 100)
    print(f"B3-sg 结果  : {os.path.join(cfg.mosei_dir, 'b3sg_results.json')}")
    print(f"三范数表    : {os.path.join(cfg.mosei_dir, 'b3sg_threenorm_lam*.csv')}")
    print(f"诊断表      : {os.path.join(cfg.mosei_dir, 'b3sg_diagnostics.csv')}")
    print(f"统一实验记录: {rec.csv} / {rec.jsonl}")
    print("=" * 100)


if __name__ == "__main__":
    main()
