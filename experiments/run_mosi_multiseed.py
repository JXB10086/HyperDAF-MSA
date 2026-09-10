"""
MOSI · Multi-seed 稳健性 (Architecture Freeze 后) · 解决单 seed 判定模糊
--------------------------------------------------
背景: run_mosi_final.py 单 seed(42) 显示 MOSEI 的干净阶梯 (HB3 < F2nb < FINAL) 在 MOSI 上未复现:
    Protocol A: FINAL 最差 (fusion 伤 MAE); Protocol B: FINAL≈HB3 (missMAE 差 0.002), Corr 反更差。
差距极小且 MOSI 极小 (train=1284) -> 单 seed 无法区分 "fusion 真无益" 与 "seed 噪声"。
本 runner 按用户既定原则 (冻结后对 最终模型 + 关键消融 做 seed=42/43/44, 报 Mean±Std) 补多 seed:
    阶梯: H0+B3(HB3) / H0+F2_nocons(F2nb) / H0+B3+F2(FINAL)
    Protocol A 与 B 各跑 3 seed, 聚合 固定missAvg(MAE/Corr) 与 随机randMean(MAE/Corr) 的 Mean±Std,
    并统计逐 seed 胜负 (FINAL vs HB3, FINAL vs F2nb) 以判断增益是否稳定。
不改架构; 不加模块; 缺失模态 α 严格 0。
运行:  python experiments/run_mosi_multiseed.py
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
from models.stage4_fusion import GlobalQueryFusion                           # noqa: E402

LAM_B3, SG, CONS_NORM = 0.005, True, "l2"
SEEDS = [42, 43, 44]
PROTOCOLS = ["A", "B"]
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
LADDER = [
    ("HB3", "H0+B3    (rep+cons)", None, LAM_B3, SG, CONS_NORM),
    ("F2nb", "H0+F2    (rep+fusion,nocons)", "F2", 0.0, False, None),
    ("FINAL", "H0+B3+F2 (FINAL)", "F2", LAM_B3, SG, CONS_NORM),
]
TAG = "MOSI-MultiSeed"


def print_table(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.exp_dir, "mosi_multiseed")
    os.makedirs(out_dir, exist_ok=True)

    datasets = build_mosi_datasets(cfg.data_path)
    s0 = datasets["train"][0]
    Dt, Da, Dv, L = (s0["text"].shape[1], s0["audio"].shape[1],
                     s0["vision"].shape[1], s0["text"].shape[0])
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = Dt, Da, Dv, L
    common = dict(text_dim=Dt, audio_dim=Da, vision_dim=Dv, hyper_dim=cfg.hyper_dim,
                  hidden_dim=cfg.hyper_hidden_dim, dropout=cfg.dropout)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])
    cond_mask = {name: generate_missing_mask(n_test, missing_pattern=pat)
                 for name, pat in FIXED_TEST_CONDITIONS}

    print("=" * 100)
    print("MOSI · Multi-seed 稳健性 · 阶梯 × Protocol A/B × seed", SEEDS, "· 30ep")
    print(f"  dims text={Dt} audio={Da} vision={Dv} L={L} | train={len(datasets['train'])} "
          f"valid={len(datasets['valid'])} test={n_test}")
    print("=" * 100)

    rec = ExperimentRecorder(out_dir)
    # store[protocol][key][metric] = list over seeds
    store = {pr: {k: {"missMAE": [], "missCorr": [], "randMAE": [], "randCorr": []}
                  for k, *_ in LADDER} for pr in PROTOCOLS}

    for seed in SEEDS:
        cfg.seed = seed
        for pr in PROTOCOLS:
            rand_mask = {p: generate_missing_mask(n_test, missing_probability=p,
                                                  allow_all_missing=False,
                                                  seed=seed + 5000 + int(round(p * 100)))
                         for p in cfg.random_test_probs}
            for key, label, fac_kind, lam, sg, norm in LADDER:
                factory = None if fac_kind is None else (lambda: GlobalQueryFusion(**common))
                model, info = train_stage3c_consistency(pr, cfg, datasets, device,
                                                        lam_cons=lam, hyper_dim=cfg.hyper_dim,
                                                        stop_grad_full=sg, cons_norm=norm,
                                                        model_factory=factory, verbose=False)
                fx = {}
                for cn, _ in FIXED_TEST_CONDITIONS:
                    _, m = evaluate_with_mask_s3(model, "hyper", test_loader, cond_mask[cn], device)
                    fx[cn] = m
                rd = {}
                for p in cfg.random_test_probs:
                    _, m = evaluate_with_mask_s3(model, "hyper", test_loader, rand_mask[p], device)
                    rd[p] = m
                store[pr][key]["missMAE"].append(float(np.mean([fx[c]["MAE"] for c in MISS])))
                store[pr][key]["missCorr"].append(float(np.mean([fx[c]["Corr"] for c in MISS])))
                store[pr][key]["randMAE"].append(float(np.mean([rd[p]["MAE"] for p in cfg.random_test_probs])))
                store[pr][key]["randCorr"].append(float(np.mean([rd[p]["Corr"] for p in cfg.random_test_probs])))
                rec.log(dataset=TAG, protocol=pr, model=label, seed=seed,
                        missing_pattern="missAvg", missing_probability=cfg.protocol_b_train_missing_prob,
                        best_epoch=info["best_epoch"], n_params=info["n_params"],
                        MAE=store[pr][key]["missMAE"][-1], Corr=store[pr][key]["missCorr"][-1])
                print(f"  [seed={seed}|P{pr}] {label:26s} missMAE={store[pr][key]['missMAE'][-1]:.4f} "
                      f"missCorr={store[pr][key]['missCorr'][-1]:.4f} "
                      f"randMAE={store[pr][key]['randMAE'][-1]:.4f}", flush=True)

    # ---- 聚合 Mean±Std ----
    def agg_table(pr, metric):
        rows = []
        for key, label, *_ in LADDER:
            v = np.array(store[pr][key][metric])
            rows.append({"model": label, "mean": round(float(v.mean()), 4),
                         "std": round(float(v.std(ddof=0)), 4),
                         "mean±std": f"{v.mean():.4f}±{v.std(ddof=0):.4f}",
                         **{f"s{s}": round(float(x), 4) for s, x in zip(SEEDS, v)}})
        return pd.DataFrame(rows)

    for pr in PROTOCOLS:
        for metric in ["missMAE", "missCorr", "randMAE", "randCorr"]:
            arrow = "↓" if "MAE" in metric else "↑"
            print_table(f"[MOSI P{pr} · {metric} {arrow} · Mean±Std over seeds {SEEDS}]",
                        agg_table(pr, metric))

    # ---- 逐 seed 胜负 (增益稳定性) ----
    print("\n" + "=" * 100)
    print("[逐 seed 胜负] FINAL vs 消融 (MAE 更低=胜, Corr 更高=胜)")
    print("=" * 100)
    verdict = {}
    for pr in PROTOCOLS:
        v = {}
        for base in ["HB3", "F2nb"]:
            mae_win = [int(store[pr]["FINAL"]["missMAE"][i] < store[pr][base]["missMAE"][i])
                       for i in range(len(SEEDS))]
            corr_win = [int(store[pr]["FINAL"]["missCorr"][i] > store[pr][base]["missCorr"][i])
                        for i in range(len(SEEDS))]
            v[f"FINAL_vs_{base}"] = {"missMAE_win": mae_win, "missCorr_win": corr_win,
                                     "mae_winrate": sum(mae_win) / len(SEEDS),
                                     "corr_winrate": sum(corr_win) / len(SEEDS)}
            print(f"  P{pr} FINAL vs {base}: missMAE 胜 {sum(mae_win)}/{len(SEEDS)} {mae_win} | "
                  f"missCorr 胜 {sum(corr_win)}/{len(SEEDS)} {corr_win}")
        verdict[pr] = v

    dump = {"dataset": "MOSI", "stage": "multiseed", "seeds": SEEDS, "protocols": PROTOCOLS,
            "ladder": {k: l for k, l, *_ in LADDER},
            "config": {"epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size,
                       "hyper_dim": cfg.hyper_dim, "lam_cons": LAM_B3, "dims": [Dt, Da, Dv, L],
                       "splits": [len(datasets["train"]), len(datasets["valid"]), n_test]},
            "random_test_probs": cfg.random_test_probs,
            "store": store, "verdict": verdict}
    with open(os.path.join(out_dir, "mosi_multiseed_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)
    for pr in PROTOCOLS:
        agg_table(pr, "missMAE").to_csv(os.path.join(out_dir, f"mosi_ms_P{pr}_missMAE.csv"),
                                        index=False, encoding="utf-8-sig")
        agg_table(pr, "missCorr").to_csv(os.path.join(out_dir, f"mosi_ms_P{pr}_missCorr.csv"),
                                         index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print(f"MOSI Multi-seed 结果 : {os.path.join(out_dir, 'mosi_multiseed_results.json')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
