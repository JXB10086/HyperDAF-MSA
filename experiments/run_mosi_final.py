"""
MOSI · Final Validation (Architecture Freeze 后) · 最终候选 H0+B3+F2 的跨数据集验证
--------------------------------------------------
用户 2026-09-06 裁决: 架构冻结 (不再新增模块/融合搜索/consistency 形式), 进入 Final Validation:
    MOSI -> MOSEI(已完成) -> IEMOCAP -> Multi-seed -> SOTA -> Paper。
本 runner 在 MOSI 上验证最终候选 = H0 + B3(λ=0.005,sg,l2) + F2(GlobalQuery) 及其两级消融:
    H0+B3        (rep+cons, 无融合)   model_factory=None,  lam=0.005
    H0+F2_nocons (rep+fusion, 无cons) GlobalQueryFusion,   lam=0.0
    H0+B3+F2     (rep+fusion+cons)    GlobalQueryFusion,   lam=0.005   <- FINAL
每个模型在 Protocol A 与 Protocol B 各训一次 (30ep, seed=42, bs=32, lr=1e-3), 评估:
    - 固定 7 缺失条件 (FIXED_TEST_CONDITIONS) MAE/Corr + missAvg
    - 随机缺失率 p∈{0.1,0.3,0.5,0.7,0.9} (cfg.random_test_probs) MAE/Corr + 均值
重点回答: 在 MOSI 上 B3+F2 是否仍有正向增益 (FINAL vs 两级消融), 即 MOSEI 结论是否跨数据集成立。
不新增任何模块/正则; 缺失模态 attention 权重严格 0; F2 用有界可学习 global query。
FINAL 在 Protocol B 下另存 per-sample α (7,N,3) + 缺失归零审计 + 热力图。
运行:  python experiments/run_mosi_final.py
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

from datasets import build_mosi_datasets                                      # noqa: E402
from configs.config import Config                                            # noqa: E402
from utils import generate_missing_mask, ExperimentRecorder                  # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS                     # noqa: E402
from experiments.stage3_lib import evaluate_with_mask_s3                     # noqa: E402
from experiments.stage3c_lib import train_stage3c_consistency                # noqa: E402
from experiments.stage4_lib import (extract_alpha_s4, alpha_missing_audit,   # noqa: E402
                                    plot_alpha_heatmap)
from models.stage4_fusion import GlobalQueryFusion                           # noqa: E402

LAM_B3 = 0.005
SG = True
CONS_NORM = "l2"
PROTOCOLS = ["A", "B"]
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
MOD = ["T", "A", "V"]
# 阶梯: (key, 展示名, model_factory或None, lam_cons, sg, cons_norm)
LADDER = [
    ("HB3", "H0+B3        (rep+cons, 无融合)", None, LAM_B3, SG, CONS_NORM),
    ("F2nb", "H0+F2        (rep+fusion, 无cons)", "F2", 0.0, False, None),
    ("FINAL", "H0+B3+F2     (rep+fusion+cons)=FINAL", "F2", LAM_B3, SG, CONS_NORM),
]
DATASET_TAG = "MOSI-Final"


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
    out_dir = os.path.join(cfg.exp_dir, "mosi_final")
    os.makedirs(out_dir, exist_ok=True)

    datasets = build_mosi_datasets(cfg.data_path)
    s0 = datasets["train"][0]
    Dt, Da, Dv = s0["text"].shape[1], s0["audio"].shape[1], s0["vision"].shape[1]
    L = s0["text"].shape[0]
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = Dt, Da, Dv, L

    print("=" * 108)
    print("MOSI · Final Validation · 最终候选 H0+B3+F2 + 两级消融 · Protocol A/B · 30ep")
    print(f"  B3 封版: λ={LAM_B3}, sg={SG}, norm={CONS_NORM} | F2=GlobalQuery(有界可学习 q, 不用 z_hyper)")
    print(f"  seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} | hyper D={cfg.hyper_dim}")
    print(f"  dims: text={Dt} audio={Da} vision={Dv} L={L} | train={len(datasets['train'])} "
          f"valid={len(datasets['valid'])} test={len(datasets['test'])}")
    print("=" * 108)

    rec = ExperimentRecorder(out_dir)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])
    cond_mask = {name: generate_missing_mask(n_test, missing_pattern=pat)
                 for name, pat in FIXED_TEST_CONDITIONS}
    rand_mask = {p: generate_missing_mask(n_test, missing_probability=p, allow_all_missing=False,
                                          seed=cfg.seed + 5000 + int(round(p * 100)))
                 for p in cfg.random_test_probs}
    common = dict(text_dim=Dt, audio_dim=Da, vision_dim=Dv, hyper_dim=cfg.hyper_dim,
                  hidden_dim=cfg.hyper_hidden_dim, dropout=cfg.dropout)

    # results[protocol][key] = {"fixed": {cond: {MAE,Corr}}, "rand": {p: {MAE,Corr}}, "info": ..., }
    results = {pr: {} for pr in PROTOCOLS}
    alpha_store = {}

    for pr in PROTOCOLS:
        for key, label, fac_kind, lam, sg, norm in LADDER:
            factory = None if fac_kind is None else (lambda: GlobalQueryFusion(**common))
            print(f"\n>>> [MOSI|P{pr}] 训练 {label} (lam_cons={lam}) ...", flush=True)
            model, info = train_stage3c_consistency(pr, cfg, datasets, device,
                                                    lam_cons=lam, hyper_dim=cfg.hyper_dim,
                                                    stop_grad_full=sg, cons_norm=norm,
                                                    model_factory=factory, verbose=False)
            print(f"    best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f}",
                  flush=True)
            torch.save({"model_state_dict": model.state_dict(), "variant": key,
                        "protocol": pr, "lam_cons": lam, "stop_grad_full": sg, "cons_norm": norm,
                        "best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                        "n_params": info["n_params"]},
                       os.path.join(out_dir, f"mosi_final_{key}_P{pr}.pt"))

            fixed = {}
            for cond_name, _ in FIXED_TEST_CONDITIONS:
                _, m = evaluate_with_mask_s3(model, "hyper", test_loader, cond_mask[cond_name], device)
                fixed[cond_name] = m
                rec.log(dataset=DATASET_TAG, protocol=pr, model=label, seed=cfg.seed,
                        missing_pattern=cond_name,
                        missing_probability=cfg.protocol_b_train_missing_prob,
                        best_epoch=info["best_epoch"], n_params=info["n_params"],
                        MAE=m["MAE"], Corr=m["Corr"])
            rand = {}
            for p in cfg.random_test_probs:
                _, m = evaluate_with_mask_s3(model, "hyper", test_loader, rand_mask[p], device)
                rand[p] = m
            results[pr][key] = {"fixed": fixed, "rand": rand,
                                "best_epoch": info["best_epoch"],
                                "best_valid_mae": info["best_valid_mae"],
                                "n_params": info["n_params"]}

            # FINAL 在 Protocol B 下存 per-sample α (供 Experiment 6)
            if key == "FINAL" and pr == "B":
                mean_mat, std_mat, audit_rows, alpha_by = [], [], [], {}
                for cond_name, _ in FIXED_TEST_CONDITIONS:
                    a = extract_alpha_s4(model, test_loader, cond_mask[cond_name], device)["alpha"]
                    alpha_by[cond_name] = a
                    mean_mat.append(a.mean(axis=0)); std_mat.append(a.std(axis=0))
                    aud = alpha_missing_audit(a, np.asarray(cond_mask[cond_name][0]))
                    audit_rows.append({"variant": key, "cond": cond_name,
                                       "max_missing_α": round(aud["max_missing_alpha"], 8),
                                       "mean_α_T": round(float(a[:, 0].mean()), 4),
                                       "mean_α_A": round(float(a[:, 1].mean()), 4),
                                       "mean_α_V": round(float(a[:, 2].mean()), 4)})
                alpha_store = {"mean": np.array(mean_mat), "std": np.array(std_mat),
                               "audit": audit_rows, "by": alpha_by}
                np.save(os.path.join(out_dir, "mosi_final_alpha_FINAL.npy"),
                        np.stack([alpha_by[c] for c in COND_NAMES], axis=0))
                worst = max(r["max_missing_α"] for r in audit_rows)
                print(f"    [α FINAL|PB] None: α_T/A/V={alpha_store['mean'][0].round(3)} | "
                      f"缺失归零 max_α_missing={worst:.2e}")

    # ---- 每 protocol 的阶梯表 ----
    def ladder_fixed(pr, metric):
        rows = []
        for key, label, *_ in LADDER:
            fx = results[pr][key]["fixed"]
            rows.append({"model": label, **{c: round(fx[c][metric], 4) for c in COND_NAMES},
                         "missAvg": round(_avg([fx[c][metric] for c in MISS]), 4)})
        return pd.DataFrame(rows)

    def ladder_rand(pr, metric):
        rows = []
        for key, label, *_ in LADDER:
            rd = results[pr][key]["rand"]
            rows.append({"model": label,
                         **{f"p={p}": round(rd[p][metric], 4) for p in cfg.random_test_probs},
                         "randMean": round(_avg([rd[p][metric] for p in cfg.random_test_probs]), 4)})
        return pd.DataFrame(rows)

    for pr in PROTOCOLS:
        print_table(f"[MOSI P{pr} · 固定7条件 MAE ↓]", ladder_fixed(pr, "MAE"))
        print_table(f"[MOSI P{pr} · 固定7条件 Corr ↑]", ladder_fixed(pr, "Corr"))
        print_table(f"[MOSI P{pr} · 随机缺失率 MAE ↓]", ladder_rand(pr, "MAE"))
        print_table(f"[MOSI P{pr} · 随机缺失率 Corr ↑]", ladder_rand(pr, "Corr"))

    # ---- 判定: B3+F2 增益是否在 MOSI 保持 ----
    print("\n" + "=" * 108)
    print("[MOSI 判定] 最终候选 H0+B3+F2 的增量 (vs 两级消融), 固定missAvg 与 随机randMean")
    print("=" * 108)
    verdict = {}
    for pr in PROTOCOLS:
        fx = {k: _avg([results[pr][k]["fixed"][c]["MAE"] for c in MISS]) for k, *_ in
              [(x[0],) for x in LADDER]}
        cx = {k: _avg([results[pr][k]["fixed"][c]["Corr"] for c in MISS]) for k, *_ in
              [(x[0],) for x in LADDER]}
        rm = {k: _avg([results[pr][k]["rand"][p]["MAE"] for p in cfg.random_test_probs])
              for k, *_ in [(x[0],) for x in LADDER]}
        b3 = fx["FINAL"] < fx["F2nb"] and rm["FINAL"] < rm["F2nb"]
        fu = fx["FINAL"] < fx["HB3"] and rm["FINAL"] < rm["HB3"]
        verdict[pr] = {"fixed_MAE": fx, "fixed_Corr": cx, "rand_MAE": rm,
                       "B3_gain": bool(b3), "fusion_gain": bool(fu)}
        print(f"  P{pr}: missMAE  HB3={fx['HB3']:.4f} F2nb={fx['F2nb']:.4f} FINAL={fx['FINAL']:.4f} | "
              f"randMAE HB3={rm['HB3']:.4f} F2nb={rm['F2nb']:.4f} FINAL={rm['FINAL']:.4f}")
        print(f"       missCorr HB3={cx['HB3']:.4f} F2nb={cx['F2nb']:.4f} FINAL={cx['FINAL']:.4f}")
        print(f"       B3-on-fusion 增益:{'✓' if b3 else '✗'} | fusion+cons 增益:{'✓' if fu else '✗'}")

    # ---- α 热力图 (FINAL vs F2nb, Protocol B) ----
    if alpha_store:
        png = os.path.join(out_dir, "mosi_final_alpha_heatmap.png")
        ok = plot_alpha_heatmap({"MOSI_FINAL": alpha_store["mean"]},
                                {"MOSI_FINAL": alpha_store["std"]}, COND_NAMES, png)
        print(f"\n  α 热力图(MOSI FINAL): {'已保存 ' + png if ok else '跳过'}")

    # ---- 落盘 ----
    for pr in PROTOCOLS:
        ladder_fixed(pr, "MAE").to_csv(os.path.join(out_dir, f"mosi_final_P{pr}_mae.csv"),
                                       index=False, encoding="utf-8-sig")
        ladder_fixed(pr, "Corr").to_csv(os.path.join(out_dir, f"mosi_final_P{pr}_corr.csv"),
                                        index=False, encoding="utf-8-sig")
        ladder_rand(pr, "MAE").to_csv(os.path.join(out_dir, f"mosi_final_P{pr}_randmae.csv"),
                                      index=False, encoding="utf-8-sig")
    if alpha_store:
        pd.DataFrame(alpha_store["audit"]).to_csv(os.path.join(out_dir, "mosi_final_alpha_audit.csv"),
                                                  index=False, encoding="utf-8-sig")
    dump = {
        "dataset": "MOSI", "stage": "final-validation", "protocol_list": PROTOCOLS,
        "lam_cons": LAM_B3, "stop_grad_full": SG, "cons_norm": CONS_NORM,
        "ladder": {k: label for k, label, *_ in LADDER},
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr,
                   "batch_size": cfg.batch_size, "hyper_dim": cfg.hyper_dim,
                   "dims": [Dt, Da, Dv, L], "splits": [len(datasets["train"]),
                                                       len(datasets["valid"]), len(datasets["test"])]},
        "random_test_probs": cfg.random_test_probs,
        "results": results, "verdict": verdict,
    }
    with open(os.path.join(out_dir, "mosi_final_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)
    print("\n" + "=" * 108)
    print(f"MOSI Final 结果 : {os.path.join(out_dir, 'mosi_final_results.json')}")
    print(f"统一记录        : {rec.csv} / {rec.jsonl}")
    print("=" * 108)


if __name__ == "__main__":
    main()
