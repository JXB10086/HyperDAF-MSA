"""
阶段2 实验编排  (模态缺失基线 + Mask-aware Baseline)
--------------------------------------------------
一次性完成阶段2全部实验, 交付用户要求的 7 项结果:
  [1] 完整模态 Baseline MAE/Corr        (表1 的 None 列)
  [2] 6 种固定缺失模式 T/A/V/TA/TV/AV   (表1)
  [3] Protocol A: 完整训练 -> 缺失测试   (表1/表2 的 P-A 行)
  [4] Protocol B: 缺失训练 -> 缺失测试   (表1/表2 的 P-B 行)
  [5] Mask-aware(B) vs Baseline(A) 对比  (表1/表2 + 关键对比)
  [6] 随机缺失 p=0.1/0.3/0.5/0.7/0.9     (表2)
  [7] 两张曲线 p->MAE, p->Corr           (random_missing_curves.png)

约束: 相同 seed / 相同划分 / 相同训练超参数; 仅用 valid 选模型; 不碰 test 调参。
禁止: Hyper / Cross-Attention / MCAC / CCL / KL / Missing Embedding / 复杂门控。

运行:  python experiments/run_stage2.py
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

from datasets import build_mosi_datasets                       # noqa: E402
from configs.config import Config                              # noqa: E402
from utils import generate_missing_mask                        # noqa: E402
from experiments.stage2_lib import (                           # noqa: E402
    MODEL_TYPES, MODEL_LABEL, FIXED_TEST_CONDITIONS,
    train_stage2_model, evaluate_with_mask,
)

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

PROTOCOLS = ("A", "B")
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]


def train_all(cfg, datasets, device):
    """训练 4 个模型: {baseline, mask_aware} × {Protocol A, B}。"""
    trained, infos = {}, {}
    for model_type in MODEL_TYPES:
        for protocol in PROTOCOLS:
            print(f"\n>>> 训练 {MODEL_LABEL[model_type]} + Protocol {protocol} ...")
            model, info = train_stage2_model(model_type, protocol, cfg, datasets, device, verbose=False)
            trained[(model_type, protocol)] = model
            infos[(model_type, protocol)] = info
            h = info["history"]
            print(f"    train_loss {h[0]['train_loss']:.4f} -> {h[-1]['train_loss']:.4f} | "
                  f"best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f}")
            torch.save({"model_state_dict": model.state_dict(), "model_type": model_type,
                        "protocol": protocol, "best_epoch": info["best_epoch"],
                        "best_valid_mae": info["best_valid_mae"]},
                       os.path.join(cfg.stage2_dir, f"{model_type}_protocol{protocol}.pt"))
    return trained, infos


def eval_fixed(trained, test_loader, n_test, device):
    """表1: 7 种固定条件 (None + 6 缺失) 的 test 评估。"""
    res = {}
    for (mt, pr), model in trained.items():
        for cond_name, pattern in FIXED_TEST_CONDITIONS:
            mask = generate_missing_mask(n_test, missing_pattern=pattern)
            _, m = evaluate_with_mask(model, mt, test_loader, mask, device)
            res[(mt, pr, cond_name)] = m
    return res


def eval_random(trained, test_loader, n_test, cfg, device):
    """表2: 随机缺失 p 列表的 test 评估 (mask 固定 seed, A/B 用同一批 mask)。"""
    res = {}
    for (mt, pr), model in trained.items():
        for p in cfg.random_test_probs:
            mask = generate_missing_mask(n_test, missing_probability=p,
                                         allow_all_missing=False, seed=cfg.seed)
            _, m = evaluate_with_mask(model, mt, test_loader, mask, device)
            res[(mt, pr, p)] = m
    return res


def build_table(results, conditions, metric):
    rows = []
    for mt in MODEL_TYPES:
        for pr in PROTOCOLS:
            row = {"Model": MODEL_LABEL[mt], "Protocol": pr}
            for cond in conditions:
                row[str(cond)] = round(results[(mt, pr, cond)][metric], 4)
            rows.append(row)
    return pd.DataFrame(rows)


def print_table(title, df):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(df.to_string(index=False))


def print_key_findings(fixed, cfg):
    """自动汇总两条证据链: 缺失是否致退化 / Mask 是否带来恢复。"""
    print("\n" + "=" * 78)
    print("[关键对比] 基于 test 集实际结果 (6 种缺失模式平均)")
    print("=" * 78)
    miss = [c for c in COND_NAMES if c != "None"]
    for pr in PROTOCOLS:
        none_b = fixed[("baseline", pr, "None")]["MAE"]
        avg_b = float(np.mean([fixed[("baseline", pr, c)]["MAE"] for c in miss]))
        avg_m = float(np.mean([fixed[("mask_aware", pr, c)]["MAE"] for c in miss]))
        print(f"  Protocol {pr}:")
        print(f"    ① 缺失致退化? Baseline None MAE={none_b:.4f} -> 缺失均值={avg_b:.4f} "
              f"({'↑退化' if avg_b > none_b else '↓未退化'} {avg_b - none_b:+.4f})")
        print(f"    ② Mask 有用?  缺失均值 Baseline(A)={avg_b:.4f} vs MaskAware(B)={avg_m:.4f} "
              f"({'B更优' if avg_m < avg_b else 'A更优'} {avg_b - avg_m:+.4f})")


def plot_random_curves(random_res, cfg):
    probs = cfg.random_test_probs
    style = {("baseline", "A"): ("tab:blue", "-", "o", "Baseline(A)-PA"),
             ("mask_aware", "A"): ("tab:red", "-", "s", "MaskAware(B)-PA"),
             ("baseline", "B"): ("tab:blue", "--", "^", "Baseline(A)-PB"),
             ("mask_aware", "B"): ("tab:red", "--", "D", "MaskAware(B)-PB")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric, ax, title in [("MAE", axes[0], "Missing Rate -> MAE (lower is better)"),
                              ("Corr", axes[1], "Missing Rate -> Corr (higher is better)")]:
        for mt in MODEL_TYPES:
            for pr in PROTOCOLS:
                vals = [random_res[(mt, pr, p)][metric] for p in probs]
                c, ls, mk, lb = style[(mt, pr)]
                ax.plot(probs, vals, color=c, linestyle=ls, marker=mk, label=lb, linewidth=1.8)
        ax.set_xlabel("Missing Rate p")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(cfg.stage2_dir, "random_missing_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.stage2_dir, exist_ok=True)

    print("=" * 78)
    print("阶段2 实验: 模态缺失基线(A) + Mask-aware Baseline(B)")
    print(f"Device={device} | seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} "
          f"proj={cfg.proj_dim} hidden={cfg.hidden_dim} dropout={cfg.dropout}")
    print(f"Protocol B 训练缺失概率={cfg.protocol_b_train_missing_prob} | 随机测试 p={cfg.random_test_probs}")
    print("=" * 78)

    datasets = build_mosi_datasets(cfg.data_path)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])
    print(f"Test samples={n_test}")

    trained, infos = train_all(cfg, datasets, device)

    fixed = eval_fixed(trained, test_loader, n_test, device)
    rand = eval_random(trained, test_loader, n_test, cfg, device)

    df_fmae = build_table(fixed, COND_NAMES, "MAE")
    df_fcorr = build_table(fixed, COND_NAMES, "Corr")
    df_rmae = build_table(rand, cfg.random_test_probs, "MAE")
    df_rcorr = build_table(rand, cfg.random_test_probs, "Corr")

    print_table("[表1a] 固定缺失 -> MAE (↓)   列名=缺失的模态", df_fmae)
    print_table("[表1b] 固定缺失 -> Corr (↑)", df_fcorr)
    print_table("[表2a] 随机缺失 p -> MAE (↓)", df_rmae)
    print_table("[表2b] 随机缺失 p -> Corr (↑)", df_rcorr)
    print_key_findings(fixed, cfg)

    # ---- 落盘 ----
    df_fmae.to_csv(os.path.join(cfg.stage2_dir, "fixed_mae.csv"), index=False)
    df_fcorr.to_csv(os.path.join(cfg.stage2_dir, "fixed_corr.csv"), index=False)
    df_rmae.to_csv(os.path.join(cfg.stage2_dir, "random_mae.csv"), index=False)
    df_rcorr.to_csv(os.path.join(cfg.stage2_dir, "random_corr.csv"), index=False)
    dump = {
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size,
                   "proj_dim": cfg.proj_dim, "hidden_dim": cfg.hidden_dim, "dropout": cfg.dropout,
                   "protocol_b_train_missing_prob": cfg.protocol_b_train_missing_prob,
                   "random_test_probs": cfg.random_test_probs},
        "fixed": {f"{mt}|{pr}|{c}": fixed[(mt, pr, c)] for (mt, pr, c) in fixed},
        "random": {f"{mt}|{pr}|{p}": rand[(mt, pr, p)] for (mt, pr, p) in rand},
        "train_info": {f"{mt}|{pr}": {"best_epoch": infos[(mt, pr)]["best_epoch"],
                                      "best_valid_mae": infos[(mt, pr)]["best_valid_mae"]}
                       for (mt, pr) in infos},
    }
    with open(os.path.join(cfg.stage2_dir, "stage2_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)

    png = plot_random_curves(rand, cfg)
    print("\n" + "=" * 78)
    print(f"曲线已保存 : {png}")
    print(f"结果目录   : {cfg.stage2_dir}")
    print("产物: fixed_mae/corr.csv, random_mae/corr.csv, stage2_results.json, "
          "random_missing_curves.png, 4×checkpoint(.pt, 已gitignore)")
    print("=" * 78)


if __name__ == "__main__":
    main()
