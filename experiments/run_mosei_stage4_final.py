"""
Stage 4 · Final (路线收缩) · 最终候选 = H0 + B3 + F2(GlobalQuery) 的封版消融 (MOSEI · Protocol B · 30ep)
--------------------------------------------------
用户 2026-09-06 裁决: 停止修 F3, 收缩为 [Missing-Aware Hyper Rep] + [Missing-Pattern Consistency(B3)]
+ [Global-Query Dynamic Fusion]。F3 Raw / F3 Norm 作为负结果保留, 不再加 temperature/LayerNorm/clipping/EMA 等。

【关键事实】现有 stage4 F2 已经 == H0 + B3(λ=0.005,sg,l2) + GlobalQuery 融合 (见 run_mosei_stage4.py:
F0/F1/F2/F3 全部经 train_stage3c_consistency(lam_cons=0.005) 训练)。因此 "最终候选 H0+B3+F2" 与现有 F2
是同一模型/同一权重 (stage4_F2_protocolB.pt), 无需重训; 若再跑一次 "H0+B3+F2" 并与 "现有 F2" 比较会得到
逐位相同的数字 -> 退化比较 -> 误判 "无增益"。

故本 runner 只新训【一个】消融: F2_noCons = H0 + GlobalQuery 融合, lam_cons=0 (use_cons=False),
即 "单独 F2 (不带 B3)"。与已有结果组成三级阶梯, 回答用户核心问题:
    H0+B3      (rep+cons, 无融合)   = 已有 F0
    H0+F2      (rep+fusion, 无cons) = 本 runner 新训 F2nb
    H0+B3+F2   (rep+fusion+cons)    = 已有 F2  == 最终候选 (FINAL)
关键增量:
    B3-on-fusion 增益  = F2(final) vs F2nb        -> 一致性在融合之上是否仍有帮助
    fusion+cons 增益   = F2(final) vs F0          -> 融合+一致性 相对 仅表征+一致性
不增加任何其他模块或正则; 缺失模态 attention 权重严格 0 (屏蔽逻辑不变); F2 用有界可学习 global query。
运行:  python experiments/run_mosei_stage4_final.py
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
from experiments.stage3c_lib import train_stage3c_consistency                # noqa: E402
from experiments.stage4_lib import (extract_alpha_s4, alpha_missing_audit,   # noqa: E402
                                    plot_alpha_heatmap)
from experiments.run_mosei import diag_hyper                                 # noqa: E402
from models.stage4_fusion import GlobalQueryFusion                           # noqa: E402

PROTOCOL = "B"
KEY = "F2nb"                       # 新训: H0 + GlobalQuery 融合, 无 consistency (λ=0)
LAM_NEW = 0.0                      # use_cons=False -> 纯 task, "单独 F2" 消融
PRIOR = ["F0", "F2"]               # 沿用已有: F0=H0+B3(无融合), F2=H0+B3+F2(最终候选)
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
MOD = ["T", "A", "V"]
ROWLABEL = {
    "F0": "H0+B3        (rep+cons, 无融合)",
    "F2nb": "H0+F2        (rep+fusion, 无cons)",
    "F2": "H0+B3+F2     (rep+fusion+cons) = FINAL",
}
DATASET_TAG = "MOSEI-Stage4-Final-F2noCons"


def _avg(seq):
    return float(np.mean(seq))


def print_table(title, df):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
    print(df.to_string(index=False))


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

    print("=" * 108)
    print("Stage 4 · Final: 最终候选=H0+B3+F2 的封版消融 · 只新训 F2_noCons(λ=0) · Protocol B · 30ep")
    print("  最终候选 H0+B3+F2 == 已有 F2 (stage4_F2_protocolB.pt), 不重训; 本跑仅补 '单独F2(无cons)' 消融")
    print(f"  seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} | hyper D={cfg.hyper_dim}")
    print("=" * 108)

    rec = ExperimentRecorder(cfg.mosei_dir)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])
    none_mask = generate_missing_mask(n_test, missing_pattern="T+A+V")
    cond_mask = {name: generate_missing_mask(n_test, missing_pattern=pat)
                 for name, pat in FIXED_TEST_CONDITIONS}

    common = dict(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim, vision_dim=cfg.vision_dim,
                  hyper_dim=cfg.hyper_dim, hidden_dim=cfg.hyper_hidden_dim, dropout=cfg.dropout)

    print(f"\n>>> [Final] 训练 {ROWLABEL[KEY]} (lam_cons={LAM_NEW} -> use_cons=False) + Protocol {PROTOCOL} ...",
          flush=True)
    model, info = train_stage3c_consistency(PROTOCOL, cfg, datasets, device,
                                            lam_cons=LAM_NEW, hyper_dim=cfg.hyper_dim,
                                            stop_grad_full=False, cons_norm=None,
                                            model_factory=lambda: GlobalQueryFusion(**common),
                                            verbose=True)
    print(f"    best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f} "
          f"| #params={info['n_params']:,}", flush=True)
    torch.save({"model_state_dict": model.state_dict(), "variant": KEY,
                "model_type": "hyper_stage4_fusion_nocons", "protocol": PROTOCOL,
                "lam_cons": LAM_NEW, "best_epoch": info["best_epoch"],
                "best_valid_mae": info["best_valid_mae"], "n_params": info["n_params"]},
               os.path.join(cfg.mosei_dir, f"stage4_{KEY}_protocolB.pt"))

    # ---- 7 条件评估 + 记录 ----
    fixed = {}
    for cond_name, _ in FIXED_TEST_CONDITIONS:
        _, m = evaluate_with_mask_s3(model, "hyper", test_loader, cond_mask[cond_name], device)
        fixed[cond_name] = m
        rec.log(dataset=DATASET_TAG, protocol=PROTOCOL, model=ROWLABEL[KEY], seed=cfg.seed,
                missing_pattern=cond_name, missing_probability=cfg.protocol_b_train_missing_prob,
                best_epoch=info["best_epoch"], n_params=info["n_params"],
                MAE=m["MAE"], Corr=m["Corr"])

    # ---- z 诊断 + per-sample α (F2nb) ----
    zA = extract_representation_s3(model, "hyper", test_loader, none_mask, device)
    Zs = [extract_representation_s3(model, "hyper", test_loader, cond_mask[c], device)["rep"]
          for c in COND_NAMES]
    diag_n = diag_hyper(zA["rep"], np.stack(Zs, axis=0), zA["label"], COND_NAMES)
    zstd_n = float(np.std(zA["rep"]))

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
    print(f"    [diag {KEY}] probe_corr={diag_n['probe_corr']:+.4f} eff_rank={diag_n['eff_rank']:.3f} "
          f"z_std={zstd_n:.4f} drift_T={diag_n['per_pattern']['T missing']['drift_ratio']:.4f}", flush=True)

    # ---- 载入 prior (F0=H0+B3, F2=FINAL) ----
    fixed_prior = {k: {c: R["fixed"][f"{k}|{c}"] for c in COND_NAMES} for k in PRIOR}
    missavg_prior = {k: R["missavg"][k] for k in PRIOR}
    info_prior = R["train_info"]
    diag_prior = R["diag"]; zstd_prior = R["z_std"]

    order = ["F0", "F2nb", "F2"]

    def get_fixed(k, c, metric):
        return fixed[c][metric] if k == KEY else fixed_prior[k][c][metric]

    def missavg(k, metric):
        return _avg([fixed[c][metric] for c in MISS]) if k == KEY else missavg_prior[k][metric]

    # ---- 表1: 三级消融阶梯 7 条件 MAE / Corr ----
    def ladder(metric):
        rows = []
        for k in order:
            rows.append({"model": ROWLABEL[k],
                         **{c: round(get_fixed(k, c, metric), 4) for c in COND_NAMES},
                         "missAvg": round(missavg(k, metric), 4)})
        return pd.DataFrame(rows)
    print_table("[Final MAE ↓] 消融阶梯: H0+B3 / H0+F2(无cons) / H0+B3+F2(FINAL)", ladder("MAE"))
    print_table("[Final Corr ↑] 消融阶梯", ladder("Corr"))

    # ---- 表2: 汇总 (选模/表征/缺失均值) ----
    srows = []
    for k in order:
        if k == KEY:
            d, zs, be, vm = diag_n, zstd_n, info["best_epoch"], info["best_valid_mae"]
        else:
            d, zs = diag_prior[k], zstd_prior[k]
            be, vm = info_prior[k]["best_epoch"], info_prior[k]["best_valid_mae"]
        pt = d["per_pattern"]["T missing"]
        srows.append({"model": ROWLABEL[k], "best_ep": be, "validMAE": round(vm, 4),
                      "probe_corr": round(d["probe_corr"], 4), "eff_rank": round(d["eff_rank"], 3),
                      "z_std": round(zs, 4), "drift_T": round(pt["drift_ratio"], 4),
                      "missMAE": round(missavg(k, "MAE"), 4), "missCorr": round(missavg(k, "Corr"), 4)})
    df_sum = pd.DataFrame(srows)
    print_table("[Final 汇总] 三级阶梯 · 选模/表征诊断/缺失均值", df_sum)

    # ---- 判定: B3-on-fusion 增益 & fusion+cons 增益 ----
    mae = {k: missavg(k, "MAE") for k in order}
    corr = {k: missavg(k, "Corr") for k in order}
    b3_gain = mae["F2"] < mae["F2nb"]            # 最终候选(带B3) vs 单独F2(无B3): MAE 更低=增益
    fuse_gain = mae["F2"] < mae["F0"]            # 最终候选 vs 仅表征+cons
    print("\n" + "=" * 108)
    print("[Final 判定] 最终候选 H0+B3+F2 (=已有F2) 的增量来源")
    print("=" * 108)
    print(f"  missMAE : H0+B3={mae['F0']:.4f} | H0+F2(无cons)={mae['F2nb']:.4f} | H0+B3+F2(FINAL)={mae['F2']:.4f}")
    print(f"  missCorr: H0+B3={corr['F0']:.4f} | H0+F2(无cons)={corr['F2nb']:.4f} | H0+B3+F2(FINAL)={corr['F2']:.4f}")
    print(f"  B3-on-fusion 增益 (FINAL vs 单独F2无cons): MAE {mae['F2']:.4f} vs {mae['F2nb']:.4f} "
          f"-> {'✓ 有增益' if b3_gain else '✗ 无增益/更差'} | Corr {corr['F2']:.4f} vs {corr['F2nb']:.4f}")
    print(f"  fusion+cons 增益 (FINAL vs H0+B3无融合):   MAE {mae['F2']:.4f} vs {mae['F0']:.4f} "
          f"-> {'✓ 有增益' if fuse_gain else '✗ 无增益/更差'} | Corr {corr['F2']:.4f} vs {corr['F0']:.4f}")

    # ---- α 热力图 (F2nb vs F2 final): B3 是否改变路由 ----
    png_hm = os.path.join(cfg.mosei_dir, "stage4_final_alpha_heatmap.png")
    ok = plot_alpha_heatmap({"F2_noCons": alpha_mean_n,
                             "F2_final": np.array(R["alpha_mean"]["F2"])},
                            {"F2_noCons": alpha_std_n,
                             "F2_final": np.array(R["alpha_std"]["F2"])}, COND_NAMES, png_hm)
    print(f"\n  α 热力图(F2_noCons vs F2_final): {'已保存 ' + png_hm if ok else '跳过'}")

    # ---- 落盘 ----
    ladder("MAE").to_csv(os.path.join(cfg.mosei_dir, "stage4_final_mae.csv"), index=False, encoding="utf-8-sig")
    ladder("Corr").to_csv(os.path.join(cfg.mosei_dir, "stage4_final_corr.csv"), index=False, encoding="utf-8-sig")
    df_sum.to_csv(os.path.join(cfg.mosei_dir, "stage4_final_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(os.path.join(cfg.mosei_dir, "stage4_final_alpha_audit.csv"),
                                    index=False, encoding="utf-8-sig")
    dump = {
        "stage": "4-final", "final_candidate": "H0+B3+F2 == existing F2 (stage4_F2_protocolB.pt)",
        "new_ablation": KEY, "new_lam_cons": LAM_NEW, "protocol": PROTOCOL,
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr,
                   "batch_size": cfg.batch_size, "hyper_dim": cfg.hyper_dim},
        "ladder": {"F0": "H0+B3 (rep+cons, no fusion)", "F2nb": "H0+F2 (rep+fusion, no cons)",
                   "F2": "H0+B3+F2 (rep+fusion+cons) = FINAL"},
        "fixed_F2nb": {c: fixed[c] for c in COND_NAMES},
        "missavg": {k: {"MAE": mae[k], "Corr": corr[k]} for k in order},
        "diag_F2nb": diag_n, "z_std_F2nb": zstd_n,
        "alpha_mean_F2nb": alpha_mean_n.tolist(), "alpha_audit_F2nb": audit_rows,
        "verdict": {"B3_gain_on_fusion": bool(b3_gain), "fusion_cons_gain_over_rep_cons": bool(fuse_gain)},
        "train_info_F2nb": {"best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                            "n_params": info["n_params"], "history": info["history"]},
    }
    with open(os.path.join(cfg.mosei_dir, "stage4_final_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)
    print("\n" + "=" * 108)
    print(f"Final 结果 : {os.path.join(cfg.mosei_dir, 'stage4_final_results.json')}")
    print(f"阶梯表     : stage4_final_mae.csv / stage4_final_corr.csv / stage4_final_summary.csv")
    print(f"统一记录   : {rec.csv} / {rec.jsonl}")
    print("=" * 108)


if __name__ == "__main__":
    main()
