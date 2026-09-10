"""
Stage 3C 编排 (MOSEI): Missing-Pattern Consistency Regularization (B3)
--------------------------------------------------
以 H0 为基础, 只增加缺失模式一致性正则 L_cons = ‖z_full − z_miss‖₂² (单模态缺失视图),
在 λ_cons ∈ {0(对照), 0.01, 0.05, 0.1} × Protocol A/B 上训练, 评估 7 固定条件,
并复用 run_mosei.diag_hyper 输出 H0 vs B3 的表征诊断, 按用户四项通过标准判定。

重点回答 (用户 Stage 3C 指令):
  - T-missing drift 是否从 MOSEI H0 的 0.90 明显下降?
  - None→T-missing transfer Corr 是否从 0.139 提高?
  - effective rank 是否未被一致性正则压塌 (不得退化成 ~1.0)?
  - 任务 MAE/Corr 是否满足 MAE_B3 ≤ MAE_H0 + ε (不明显伤害), 理想是同时改善?

边界: 不做 Dynamic Fusion / Attention / MCAC / CCL / KL / Missing Embedding;
      λ=0 即 w/o-consistency 对照 (与 H0 逐位一致); 双模态缺失一致性 (B3-2) 暂不加入。
运行:  python experiments/run_mosei_b3.py   (需在 run_mosei.py 之后, 以便与 H0 主实验对照)
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

PROTOCOLS = ("A", "B")
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
HARD = ["T missing", "T+A missing", "T+V missing"]
LAMBDAS = [0.0, 0.01, 0.05, 0.1]          # λ=0 为 w/o-consistency 对照 (= H0)
EPS_MAE = 0.005                            # 通过标准: MAE_B3 ≤ MAE_H0 + ε
RANK_FLOOR = 1.5                           # eff_rank 不得压塌到此值以下 (H0=2.21)
DATASET_TAG = "MOSEI-B3"


def lab(lam):
    return "H0 (w/o cons)" if lam == 0.0 else f"H0+cons λ={lam}"


def _avg(seq):
    return float(np.mean(seq))


def build_fixed_table(fixed, metric):
    """行 = (λ, Protocol), 列 = 7 固定条件。"""
    rows = []
    for lam in LAMBDAS:
        for pr in PROTOCOLS:
            row = {"λ_cons": lam, "Protocol": pr}
            for cond in COND_NAMES:
                row[str(cond)] = round(fixed[(lam, pr, cond)][metric], 4)
            rows.append(row)
    return pd.DataFrame(rows)


def print_table(title, df):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(df.to_string(index=False))


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.mosei_dir, exist_ok=True)

    datasets = build_mosei_datasets(cfg.mosei_data_path, cfg.mosei_conv_dir,
                                    inf_policy=cfg.mosei_inf_policy)
    Dt, Da, Dv, L = mosei_dims(datasets)
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = Dt, Da, Dv, L

    print("=" * 110)
    print("Stage 3C (MOSEI): Missing-Pattern Consistency Regularization (B3) —— 以 H0 为基础只加 L_cons")
    print(f"Device={device} | seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} | hyper D={cfg.hyper_dim}")
    print(f"dims: text={Dt} audio={Da} vision={Dv} L={L} | split train={len(datasets['train'])} "
          f"valid={len(datasets['valid'])} test={len(datasets['test'])}")
    print(f"λ_cons 网格={LAMBDAS} (λ=0 为 w/o-cons 对照=H0) | 一致性视图=单模态缺失(T/A/V) | "
          f"all-missing/双模态 不参与 | ε_MAE={EPS_MAE} rank_floor={RANK_FLOOR}")
    print("=" * 110)

    rec = ExperimentRecorder(cfg.mosei_dir)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])
    none_mask = generate_missing_mask(n_test, missing_pattern="T+A+V")

    models, infos, fixed, diags = {}, {}, {}, {}
    for lam in LAMBDAS:
        for pr in PROTOCOLS:
            print(f"\n>>> [B3] 训练 H0 + L_cons (λ={lam}) + Protocol {pr} ...", flush=True)
            model, info = train_stage3c_consistency(pr, cfg, datasets, device,
                                                    lam_cons=lam, hyper_dim=cfg.hyper_dim)
            models[(lam, pr)] = model
            infos[(lam, pr)] = info
            print(f"    best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f} "
                  f"| #params={info['n_params']:,} | 末轮 train_cons={info['history'][-1]['train_cons']:.4f}",
                  flush=True)
            torch.save({"model_state_dict": model.state_dict(), "model_type": "hyper_b3",
                        "protocol": pr, "lam_cons": lam, "variant": "b3",
                        "best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                        "n_params": info["n_params"]},
                       os.path.join(cfg.mosei_dir, f"b3_lam{lam}_protocol{pr}.pt"))

        # ---- 7 固定条件评估 + 记录 ----
        for pr in PROTOCOLS:
            info = infos[(lam, pr)]
            for cond_name, pattern in FIXED_TEST_CONDITIONS:
                mask = generate_missing_mask(n_test, missing_pattern=pattern)
                _, m = evaluate_with_mask_s3(models[(lam, pr)], "hyper", test_loader, mask, device)
                fixed[(lam, pr, cond_name)] = m
                rec.log(dataset=DATASET_TAG, protocol=pr, model=lab(lam), seed=cfg.seed,
                        missing_pattern=cond_name,
                        missing_probability=(0.0 if pr == "A" else cfg.protocol_b_train_missing_prob),
                        best_epoch=info["best_epoch"], n_params=info["n_params"],
                        MAE=m["MAE"], Corr=m["Corr"])

        # ---- 表征诊断 (与 run_mosei 口径一致: A 模型 None 求质量, B 模型 7 模式求 drift/transfer) ----
        zA = extract_representation_s3(models[(lam, "A")], "hyper", test_loader, none_mask, device)
        zs = []
        for cond_name, pattern in FIXED_TEST_CONDITIONS:
            mm = generate_missing_mask(n_test, missing_pattern=pattern)
            o = extract_representation_s3(models[(lam, "B")], "hyper", test_loader, mm, device)
            zs.append(o["rep"])
        Z_by_pat = np.stack(zs, axis=0)
        y = zA["label"]
        diags[lam] = diag_hyper(zA["rep"], Z_by_pat, y, COND_NAMES)
        print(f"    [diag λ={lam}] probe_corr={diags[lam]['probe_corr']:+.3f} "
              f"eff_rank={diags[lam]['eff_rank']:.2f} sil={diags[lam]['sil_binary']:+.3f} "
              f"drift_T={diags[lam]['per_pattern']['T missing']['drift_ratio']:.3f} "
              f"transfer_T={diags[lam]['per_pattern']['T missing']['transfer']:+.3f}", flush=True)

    # ---- 表1/表2: 固定条件 MAE / Corr ----
    df_mae = build_fixed_table(fixed, "MAE")
    df_corr = build_fixed_table(fixed, "Corr")
    print_table("[B3-MAE] λ_cons × Protocol A/B × 7 固定条件  (MAE ↓)", df_mae)
    print_table("[B3-Corr] λ_cons × Protocol A/B × 7 固定条件  (Corr ↑)", df_corr)

    # ---- 表3: H0 vs B3 表征诊断 ----
    drows = []
    for lam in LAMBDAS:
        d = diags[lam]
        pt, pa, pv = d["per_pattern"]["T missing"], d["per_pattern"]["A missing"], d["per_pattern"]["V missing"]
        drows.append({
            "λ_cons": lam,
            "probe_corr": round(d["probe_corr"], 4), "probe_r2": round(d["probe_r2"], 4),
            "sil": round(d["sil_binary"], 4), "eff_rank": round(d["eff_rank"], 3),
            "drift_T": round(pt["drift_ratio"], 4), "transfer_T": round(pt["transfer"], 4),
            "drift_A": round(pa["drift_ratio"], 4), "transfer_A": round(pa["transfer"], 4),
            "drift_V": round(pv["drift_ratio"], 4), "transfer_V": round(pv["transfer"], 4),
        })
    df_diag = pd.DataFrame(drows)
    print_table("[B3-诊断] H0(λ=0) vs B3(λ>0): 情感质量 + 单模态缺失漂移/迁移", df_diag)

    # ---- 表4: 通过标准判定 (以 Protocol B 为主, 用户 '优先在 PB 验证') ----
    base = 0.0
    missavg = {(lam, pr): _avg([fixed[(lam, pr, c)]["MAE"] for c in MISS]) for lam in LAMBDAS for pr in PROTOCOLS}
    vrows = []
    for lam in LAMBDAS:
        if lam == base:
            continue
        d0, d1 = diags[base], diags[lam]
        drift0 = d0["per_pattern"]["T missing"]["drift_ratio"]
        drift1 = d1["per_pattern"]["T missing"]["drift_ratio"]
        tr0 = d0["per_pattern"]["T missing"]["transfer"]
        tr1 = d1["per_pattern"]["T missing"]["transfer"]
        er1 = d1["eff_rank"]
        maeB0, maeB1 = missavg[(base, "B")], missavg[(lam, "B")]
        maeA0, maeA1 = missavg[(base, "A")], missavg[(lam, "A")]
        tc0 = fixed[(base, "B", "T missing")]["Corr"]
        tc1 = fixed[(lam, "B", "T missing")]["Corr"]
        c_drift = drift1 < drift0
        c_transfer = tr1 > tr0
        c_rank = er1 >= RANK_FLOOR
        c_mae = maeB1 <= maeB0 + EPS_MAE
        must = c_drift and c_transfer and c_mae
        ideal = (maeB1 < maeB0) and (tc1 > tc0)
        vrows.append({
            "λ_cons": lam,
            "drift_T": f"{drift0:.3f}→{drift1:.3f}", "drift↓?": c_drift,
            "transfer_T": f"{tr0:+.3f}→{tr1:+.3f}", "transfer↑?": c_transfer,
            "eff_rank": round(er1, 2), "rank未塌?": c_rank,
            "missMAE_PB": f"{maeB0:.4f}→{maeB1:.4f}", "MAE≤H0+ε?": c_mae,
            "missMAE_PA": f"{maeA0:.4f}→{maeA1:.4f}",
            "TmissCorr_PB": f"{tc0:+.4f}→{tc1:+.4f}",
            "必须项通过": must, "理想项(MAE↓&Corr↑)": ideal,
        })
    df_verdict = pd.DataFrame(vrows)
    print_table("[B3-通过标准] 对照 H0(λ=0) · 主看 Protocol B · 必须项=drift_T↓ & transfer_T↑ & MAE≤H0+ε", df_verdict)

    # ---- 控制台结论 ----
    print("\n" + "=" * 110)
    print("[结论摘要]")
    print("=" * 110)
    passed = [r["λ_cons"] for r in vrows if r["必须项通过"]]
    ideal = [r["λ_cons"] for r in vrows if r["理想项(MAE↓&Corr↑)"]]
    print(f"  必须项(drift_T↓ & transfer_T↑ & MAE≤H0+ε)通过的 λ: {passed if passed else '无'}")
    print(f"  理想项(missMAE_PB↓ 且 TmissCorr_PB↑)的 λ      : {ideal if ideal else '无'}")
    print(f"  参考 H0(主实验): drift_T=0.90 transfer_T=0.139 eff_rank=2.21 sil=0.106 probe_corr=0.579")

    # ---- 落盘 ----
    df_mae.to_csv(os.path.join(cfg.mosei_dir, "b3_fixed_mae.csv"), index=False)
    df_corr.to_csv(os.path.join(cfg.mosei_dir, "b3_fixed_corr.csv"), index=False)
    df_diag.to_csv(os.path.join(cfg.mosei_dir, "b3_diagnostics.csv"), index=False)
    df_verdict.to_csv(os.path.join(cfg.mosei_dir, "b3_pass_criteria.csv"), index=False)
    dump = {
        "stage": "3C", "variant": "missing_pattern_consistency_B3", "dataset": "MOSEI",
        "cons_view": "single_modality_missing (T/A/V), all-missing & double excluded (B3-1)",
        "loss": "L_total = L_task + lam_cons * mean||z_full - z_miss||_2^2",
        "lambdas": LAMBDAS, "eps_mae": EPS_MAE, "rank_floor": RANK_FLOOR,
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr,
                   "batch_size": cfg.batch_size, "dims": [Dt, Da, Dv, L],
                   "hyper_dim": cfg.hyper_dim,
                   "protocol_b_train_missing_prob": cfg.protocol_b_train_missing_prob},
        "fixed": {f"{lam}|{pr}|{c}": fixed[(lam, pr, c)] for (lam, pr, c) in fixed},
        "missavg_mae": {f"{lam}|{pr}": missavg[(lam, pr)] for (lam, pr) in missavg},
        "diag": {str(lam): diags[lam] for lam in LAMBDAS},
        "train_info": {f"{lam}|{pr}": {"best_epoch": infos[(lam, pr)]["best_epoch"],
                                       "best_valid_mae": infos[(lam, pr)]["best_valid_mae"],
                                       "n_params": infos[(lam, pr)]["n_params"],
                                       "final_train_cons": infos[(lam, pr)]["history"][-1]["train_cons"]}
                       for (lam, pr) in infos},
        "verdict": vrows, "passed_lambdas": passed, "ideal_lambdas": ideal,
    }
    with open(os.path.join(cfg.mosei_dir, "b3_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 110)
    print(f"B3 结果     : {os.path.join(cfg.mosei_dir, 'b3_results.json')}")
    print(f"通过标准表  : {os.path.join(cfg.mosei_dir, 'b3_pass_criteria.csv')}")
    print(f"诊断表      : {os.path.join(cfg.mosei_dir, 'b3_diagnostics.csv')}")
    print(f"统一实验记录: {rec.csv} / {rec.jsonl}")
    print("=" * 110)


if __name__ == "__main__":
    main()
