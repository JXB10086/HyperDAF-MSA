"""
MOSEI · Multi-seed 稳健性 (Architecture Freeze 后) · 验证"干净阶梯"是否在 seed 方差下成立
--------------------------------------------------
动机: Stage-4-Final 的 MOSEI 阶梯 (HB3 0.7326 > F2nb 0.7314 > FINAL 0.7263 missMAE) 是【单 seed(42)】,
    差距仅 ~0.005; 而 MOSI 多 seed 显示同阶梯的 seed-std 达 0.02-0.03 (远超 0.005), 排序随 seed 翻转。
    => MOSEI 的"独立正边际贡献"结论本身可能是单 seed 噪声, 必须多 seed 复核后才能作为论文证据。
本 runner: Protocol B (原始 MOSEI 阶梯所用协议), 阶梯 3 行 × seed{42,43,44}, 聚合 Mean±Std + 逐 seed 胜负:
    H0+B3(HB3, 无融合) / H0+F2_nocons(F2nb) / H0+B3+F2(FINAL)
    指标: 固定 missAvg(MAE/Corr) + 随机 randMean(MAE/Corr)。
不改架构; 不加模块; 缺失模态 α 严格 0; B3 封版 λ=0.005/sg/l2。
运行:  python experiments/run_mosei_multiseed.py
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
from experiments.stage3_lib import evaluate_with_mask_s3                     # noqa: E402
from experiments.stage3c_lib import train_stage3c_consistency                # noqa: E402
from models.stage4_fusion import GlobalQueryFusion                           # noqa: E402

LAM_B3, SG, CONS_NORM = 0.005, True, "l2"
SEEDS = [42, 43, 44]
PROTOCOL = "B"
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
LADDER = [
    ("HB3", "H0+B3    (rep+cons)", None, LAM_B3, SG, CONS_NORM),
    ("F2nb", "H0+F2    (rep+fusion,nocons)", "F2", 0.0, False, None),
    ("FINAL", "H0+B3+F2 (FINAL)", "F2", LAM_B3, SG, CONS_NORM),
]
TAG = "MOSEI-MultiSeed"


def print_table(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.mosei_dir, "multiseed")
    os.makedirs(out_dir, exist_ok=True)

    datasets = build_mosei_datasets(cfg.mosei_data_path, cfg.mosei_conv_dir,
                                    inf_policy=cfg.mosei_inf_policy)
    Dt, Da, Dv, L = mosei_dims(datasets)
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = Dt, Da, Dv, L
    common = dict(text_dim=Dt, audio_dim=Da, vision_dim=Dv, hyper_dim=cfg.hyper_dim,
                  hidden_dim=cfg.hyper_hidden_dim, dropout=cfg.dropout)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])
    cond_mask = {name: generate_missing_mask(n_test, missing_pattern=pat)
                 for name, pat in FIXED_TEST_CONDITIONS}

    print("=" * 100)
    print("MOSEI · Multi-seed 稳健性 · 阶梯 × Protocol B × seed", SEEDS, "· 30ep")
    print(f"  dims text={Dt} audio={Da} vision={Dv} L={L} | train={len(datasets['train'])} "
          f"valid={len(datasets['valid'])} test={n_test}")
    print("=" * 100, flush=True)

    rec = ExperimentRecorder(out_dir)
    store = {k: {"missMAE": [], "missCorr": [], "randMAE": [], "randCorr": []} for k, *_ in LADDER}

    for seed in SEEDS:
        cfg.seed = seed
        rand_mask = {p: generate_missing_mask(n_test, missing_probability=p,
                                              allow_all_missing=False,
                                              seed=seed + 5000 + int(round(p * 100)))
                     for p in cfg.random_test_probs}
        for key, label, fac_kind, lam, sg, norm in LADDER:
            factory = None if fac_kind is None else (lambda: GlobalQueryFusion(**common))
            model, info = train_stage3c_consistency(PROTOCOL, cfg, datasets, device,
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
            store[key]["missMAE"].append(float(np.mean([fx[c]["MAE"] for c in MISS])))
            store[key]["missCorr"].append(float(np.mean([fx[c]["Corr"] for c in MISS])))
            store[key]["randMAE"].append(float(np.mean([rd[p]["MAE"] for p in cfg.random_test_probs])))
            store[key]["randCorr"].append(float(np.mean([rd[p]["Corr"] for p in cfg.random_test_probs])))
            rec.log(dataset=TAG, protocol=PROTOCOL, model=label, seed=seed,
                    missing_pattern="missAvg", missing_probability=cfg.protocol_b_train_missing_prob,
                    best_epoch=info["best_epoch"], n_params=info["n_params"],
                    MAE=store[key]["missMAE"][-1], Corr=store[key]["missCorr"][-1])
            print(f"  [seed={seed}|PB] {label:26s} missMAE={store[key]['missMAE'][-1]:.4f} "
                  f"missCorr={store[key]['missCorr'][-1]:.4f} "
                  f"randMAE={store[key]['randMAE'][-1]:.4f} best_ep={info['best_epoch']}", flush=True)

    def agg_table(metric):
        rows = []
        for key, label, *_ in LADDER:
            v = np.array(store[key][metric])
            rows.append({"model": label, "mean": round(float(v.mean()), 4),
                         "std": round(float(v.std(ddof=0)), 4),
                         "mean±std": f"{v.mean():.4f}±{v.std(ddof=0):.4f}",
                         **{f"s{s}": round(float(x), 4) for s, x in zip(SEEDS, v)}})
        return pd.DataFrame(rows)

    for metric in ["missMAE", "missCorr", "randMAE", "randCorr"]:
        arrow = "↓" if "MAE" in metric else "↑"
        print_table(f"[MOSEI PB · {metric} {arrow} · Mean±Std over seeds {SEEDS}]", agg_table(metric))

    print("\n" + "=" * 100)
    print("[逐 seed 胜负] FINAL vs 消融 (MAE 更低=胜, Corr 更高=胜)")
    print("=" * 100)
    verdict = {}
    for base in ["HB3", "F2nb"]:
        mae_win = [int(store["FINAL"]["missMAE"][i] < store[base]["missMAE"][i]) for i in range(len(SEEDS))]
        corr_win = [int(store["FINAL"]["missCorr"][i] > store[base]["missCorr"][i]) for i in range(len(SEEDS))]
        verdict[f"FINAL_vs_{base}"] = {"missMAE_win": mae_win, "missCorr_win": corr_win,
                                       "mae_winrate": sum(mae_win) / len(SEEDS),
                                       "corr_winrate": sum(corr_win) / len(SEEDS)}
        print(f"  FINAL vs {base}: missMAE 胜 {sum(mae_win)}/{len(SEEDS)} {mae_win} | "
              f"missCorr 胜 {sum(corr_win)}/{len(SEEDS)} {corr_win}")

    dump = {"dataset": "MOSEI", "stage": "multiseed", "seeds": SEEDS, "protocol": PROTOCOL,
            "ladder": {k: l for k, l, *_ in LADDER},
            "config": {"epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size,
                       "hyper_dim": cfg.hyper_dim, "lam_cons": LAM_B3, "dims": [Dt, Da, Dv, L],
                       "splits": [len(datasets["train"]), len(datasets["valid"]), n_test]},
            "random_test_probs": cfg.random_test_probs, "store": store, "verdict": verdict}
    with open(os.path.join(out_dir, "mosei_multiseed_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)
    for metric in ["missMAE", "missCorr", "randMAE", "randCorr"]:
        agg_table(metric).to_csv(os.path.join(out_dir, f"mosei_ms_{metric}.csv"),
                                 index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print(f"MOSEI Multi-seed 结果 : {os.path.join(out_dir, 'mosei_multiseed_results.json')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
