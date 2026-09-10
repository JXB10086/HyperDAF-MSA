"""
阶段3 实验编排  (Missing-Aware Hyper Representation 验证)
--------------------------------------------------
目标: 单独验证 Hyper Representation 相对 Baseline / Mask-aware 是否有【稳定、可解释】的增益。
交付 (对应用户 3.4~3.14):
  [表3a/3b] Baseline vs Mask-aware vs Hyper(D=256)  ×  Protocol A/B  ×  7 固定条件
  [表3c/3d] 上述 3 模型的随机缺失 p 曲线 (继承阶段2 协议)
  [表4]     Hyper Dimension Ablation D=64/128/256/512 × A/B (None + 缺失均值 + 参数量)
  [表5]     Mask-only vs Mask-aware vs Hyper (3.5: 收益是否仅来自缺失模式编码)
  [判定]    3.14: Hyper>Mask 是否稳定 -> 能否进入阶段4 (数据驱动, 不预设)
  [z_hyper] 保存表征 (npz) + PCA 可视化 (情感上色 / 缺失模式上色), 为后续 t-SNE 准备

约束: 相同 seed / 相同划分 / 相同训练超参 / Protocol A&B / 仅 valid 选模型 / 只用 test 评估。
禁令: Dynamic Fusion / MCAC / CCL / KL / Missing Embedding / Cross-Attention / 复杂门控。
运行:  python experiments/run_stage3.py
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

from datasets import build_mosi_datasets                        # noqa: E402
from configs.config import Config                               # noqa: E402
from utils import generate_missing_mask, corr                   # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS        # noqa: E402
from experiments.stage3_lib import (                            # noqa: E402
    build_specs, train_stage3_model, evaluate_with_mask_s3,
    extract_representation_s3, pca_2d,
)

import matplotlib                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

PROTOCOLS = ("A", "B")
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
# 困难缺失: 文本缺失 + 双模态缺失 (用户 3.13 期望 Hyper 在此更明显)
HARD = ["T missing", "T+A missing", "T+V missing"]


def _avg(seq):
    return float(np.mean(seq))


# ----------------------------------------------------------------------------
# 训练 & 评估
# ----------------------------------------------------------------------------
def train_all_specs(cfg, specs, datasets, device):
    trained, infos = {}, {}
    for s in specs:
        for pr in PROTOCOLS:
            print(f"\n>>> 训练 {s['label']} + Protocol {pr} ...")
            model, info = train_stage3_model(s["type"], pr, cfg, datasets, device,
                                             hyper_dim=s["dim"], verbose=False)
            trained[(s["key"], pr)] = model
            infos[(s["key"], pr)] = info
            h = info["history"]
            print(f"    train_loss {h[0]['train_loss']:.4f} -> {h[-1]['train_loss']:.4f} | "
                  f"best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f} "
                  f"| #params={info['n_params']:,}")
            torch.save({"model_state_dict": model.state_dict(), "spec_key": s["key"],
                        "model_type": s["type"], "hyper_dim": s["dim"], "protocol": pr,
                        "best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                        "n_params": info["n_params"]},
                       os.path.join(cfg.stage3_dir, f"{s['key']}_protocol{pr}.pt"))
    return trained, infos


def eval_fixed_s3(trained, specs, test_loader, n_test, device):
    """7 固定条件 × 全部 spec × 2 协议 的 test 评估。"""
    res = {}
    for s in specs:
        for pr in PROTOCOLS:
            model = trained[(s["key"], pr)]
            for cond_name, pattern in FIXED_TEST_CONDITIONS:
                mask = generate_missing_mask(n_test, missing_pattern=pattern)
                _, m = evaluate_with_mask_s3(model, s["type"], test_loader, mask, device)
                res[(s["key"], pr, cond_name)] = m
    return res


def eval_random_s3(trained, type_of, keys, test_loader, n_test, cfg, device):
    """随机缺失 p × 指定 keys × 2 协议 (mask 固定 seed, 跨模型同 mask)。"""
    res = {}
    for k in keys:
        for pr in PROTOCOLS:
            model = trained[(k, pr)]
            for p in cfg.random_test_probs:
                mask = generate_missing_mask(n_test, missing_probability=p,
                                             allow_all_missing=False, seed=cfg.seed)
                _, m = evaluate_with_mask_s3(model, type_of[k], test_loader, mask, device)
                res[(k, pr, p)] = m
    return res


# ----------------------------------------------------------------------------
# 表格
# ----------------------------------------------------------------------------
def build_table(results, keys, LABEL, conditions, metric):
    rows = []
    for k in keys:
        for pr in PROTOCOLS:
            row = {"Model": LABEL[k], "Protocol": pr}
            for cond in conditions:
                row[str(cond)] = round(results[(k, pr, cond)][metric], 4)
            rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_table(fixed, infos, cfg):
    rows = []
    for d in cfg.hyper_dim_ablation:
        k = f"hyper_D{d}"
        for pr in PROTOCOLS:
            rows.append({
                "D": d, "Protocol": pr,
                "None_MAE": round(fixed[(k, pr, "None")]["MAE"], 4),
                "None_Corr": round(fixed[(k, pr, "None")]["Corr"], 4),
                "MissAvg_MAE": round(_avg([fixed[(k, pr, c)]["MAE"] for c in MISS]), 4),
                "MissAvg_Corr": round(_avg([fixed[(k, pr, c)]["Corr"] for c in MISS]), 4),
                "Params": infos[(k, pr)]["n_params"],
            })
    return pd.DataFrame(rows)


def print_table(title, df):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    print(df.to_string(index=False))


def consistency_check(fixed):
    """内置校验: 阶段3 重训的 Baseline-PA 的 None 应与阶段1/2 完全一致 (1.0256/0.6189)。"""
    m = fixed[("baseline", "A", "None")]
    ok = abs(m["MAE"] - 1.0256) < 1e-3 and abs(m["Corr"] - 0.6189) < 1e-3
    print("\n" + "=" * 92)
    print("[一致性校验] 阶段3 重训 Baseline(A)-PA @ None")
    print("=" * 92)
    print(f"  MAE={m['MAE']:.4f}  Corr={m['Corr']:.4f}  | 阶段1/2 基准=1.0256/0.6189  -> "
          f"{'✅ 一致 (训练协议可复现)' if ok else '⚠️ 不一致 (需排查)'}")
    return ok


def print_verdict(fixed, cfg):
    """3.14 数据驱动判定: Hyper 相对 Mask-aware 是否稳定占优 -> 能否进入阶段4。"""
    hyp = f"hyper_D{cfg.hyper_dim}"
    print("\n" + "=" * 92)
    print("[阶段3 判定] Baseline < Mask < Hyper ? (基于 test 集实际结果, MAE↓)")
    print("=" * 92)
    per_pr = {}
    for pr in PROTOCOLS:
        b = _avg([fixed[("baseline", pr, c)]["MAE"] for c in MISS])
        mk = _avg([fixed[("mask_aware", pr, c)]["MAE"] for c in MISS])
        hy = _avg([fixed[(hyp, pr, c)]["MAE"] for c in MISS])
        hy_hard = _avg([fixed[(hyp, pr, c)]["MAE"] for c in HARD])
        mk_hard = _avg([fixed[("mask_aware", pr, c)]["MAE"] for c in HARD])
        wins = sum(1 for c in MISS if fixed[(hyp, pr, c)]["MAE"] < fixed[("mask_aware", pr, c)]["MAE"])
        per_pr[pr] = dict(b=b, mk=mk, hy=hy, wins=wins, hy_hard=hy_hard, mk_hard=mk_hard)
        print(f"  Protocol {pr}: 缺失均值  Baseline={b:.4f}  MaskAware={mk:.4f}  Hyper={hy:.4f}")
        print(f"             Hyper vs Mask: {mk - hy:+.4f} ({'Hyper更优' if hy < mk else 'Mask更优'}) | "
              f"6条件中Hyper胜 {wins}/6 | 困难缺失(T/TA/TV) Hyper={hy_hard:.4f} vs Mask={mk_hard:.4f}")
    # 进入阶段4 的判据 (需同时满足)
    c_a = all(per_pr[pr]["hy"] < per_pr[pr]["mk"] for pr in PROTOCOLS)          # 两协议缺失均值 Hyper<Mask
    c_b = per_pr["B"]["wins"] >= 4                                               # PB 下 Hyper 胜 >=4/6
    c_c = all(per_pr[pr]["hy_hard"] < per_pr[pr]["mk_hard"] for pr in PROTOCOLS) # 困难缺失 Hyper<Mask
    go = c_a and c_b and c_c
    print("\n  判据: (a)两协议缺失均值 Hyper<Mask=%s  (b)PB下Hyper胜>=4/6=%s  (c)困难缺失 Hyper<Mask=%s"
          % (c_a, c_b, c_c))
    if go:
        print("  ✅ 结论: Hyper 相对 Mask 有稳定且集中在困难缺失的增益 -> 支持进入阶段4 (Dynamic Fusion)。")
    else:
        print("  ⚠️ 结论: Hyper 相对 Mask 收益不稳定/有限 -> 建议先调整 Hyper Encoder (结构/深度/训练),")
        print("            暂不叠加 Dynamic Fusion; 若贸然进入阶段4 将无法归因增益来源。")
    return {"per_protocol": per_pr, "criteria": {"a": c_a, "b": c_b, "c": c_c}, "go_stage4": go}


# ----------------------------------------------------------------------------
# 表征保存 + PCA 可视化 (3.11 / 3.12)
# ----------------------------------------------------------------------------
def save_rep_and_plots(trained, cfg, test_loader, n_test, device):
    hyp = f"hyper_D{cfg.hyper_dim}"
    out_dir = cfg.stage3_dir

    # ---- (1) Protocol A, 完整模态(None): Hyper z_hyper vs MaskAware 融合表征 h, 按情感上色 ----
    none_mask = generate_missing_mask(n_test, missing_pattern="T+A+V")
    hyperA = extract_representation_s3(trained[(hyp, "A")], "hyper", test_loader, none_mask, device)
    maskA = extract_representation_s3(trained[("mask_aware", "A")], "mask_aware", test_loader, none_mask, device)
    labels = hyperA["label"]
    cH, vH = pca_2d(hyperA["rep"])
    cM, vM = pca_2d(maskA["rep"])
    pc1_label_H = corr(cH[:, 0], labels)
    pc1_label_M = corr(cM[:, 0], labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, c, v, pc1, title in [
        (axes[0], cM, vM, pc1_label_M, f"MaskAware(B) fused rep  (PCA, dim={maskA['rep'].shape[1]})"),
        (axes[1], cH, vH, pc1_label_H, f"Hyper z_hyper (D={cfg.hyper_dim})  (PCA)")]:
        sc = ax.scatter(c[:, 0], c[:, 1], c=labels, cmap="coolwarm", s=7, alpha=0.7)
        ax.set_title(f"{title}\nEVR={v[0]*100:.1f}%/{v[1]*100:.1f}%  corr(PC1,label)={pc1:+.3f}", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax, label="sentiment")
    fig.suptitle("Representation vs Sentiment (Protocol A, full-modal test) — PCA (t-SNE later needs sklearn)",
                 fontsize=11)
    plt.tight_layout()
    fig1 = os.path.join(out_dir, "repr_pca_sentiment.png")
    plt.savefig(fig1, dpi=150); plt.close()

    # ---- (2) Protocol B: z_hyper 在 7 种缺失模式下的分布, 按模式上色 (3.12, 仅分析不过度宣称) ----
    zs, pidx = [], []
    for i, (cond_name, pattern) in enumerate(FIXED_TEST_CONDITIONS):
        m = generate_missing_mask(n_test, missing_pattern=pattern)
        o = extract_representation_s3(trained[(hyp, "B")], "hyper", test_loader, m, device)
        zs.append(o["rep"]); pidx.append(np.full(n_test, i, dtype=np.int64))
    Zall = np.concatenate(zs); Pall = np.concatenate(pidx)
    cAll, vAll = pca_2d(Zall)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, (cond_name, _) in enumerate(FIXED_TEST_CONDITIONS):
        sel = Pall == i
        ax.scatter(cAll[sel, 0], cAll[sel, 1], color=cmap(i), s=6, alpha=0.5, label=cond_name)
    ax.set_title(f"Hyper z_hyper across missing patterns (Protocol B)\n"
                 f"PCA EVR={vAll[0]*100:.1f}%/{vAll[1]*100:.1f}%  — analysis only, NOT an invariance claim",
                 fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True, alpha=0.25); ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    fig2 = os.path.join(out_dir, "z_hyper_pca_missing_patterns.png")
    plt.savefig(fig2, dpi=150); plt.close()

    # ---- (3) 保存原始表征, 为后续 t-SNE 准备 (3.11) ----
    npz = os.path.join(out_dir, "z_hyper_stage3.npz")
    np.savez_compressed(
        npz,
        labels=labels,
        hyperA_none_z=hyperA["rep"], hyperA_none_pred=hyperA["pred"],
        maskA_none_h=maskA["rep"], maskA_none_pred=maskA["pred"],
        hyperB_all_patterns_z=Zall, hyperB_all_patterns_idx=Pall,
        pattern_names=np.array([c[0] for c in FIXED_TEST_CONDITIONS]),
        meta=np.array([f"hyper_dim={cfg.hyper_dim}", f"seed={cfg.seed}", "proj=PCA-ready, run t-SNE later"]))
    return fig1, fig2, npz, {"pc1_label_corr_hyperA": pc1_label_H, "pc1_label_corr_maskA": pc1_label_M}


def plot_random_curves(rand, keys, LABEL, cfg):
    hyp = f"hyper_D{cfg.hyper_dim}"
    color = {"baseline": "tab:blue", "mask_aware": "tab:orange", hyp: "tab:red"}
    marker = {"baseline": "o", "mask_aware": "s", hyp: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric, ax, title in [("MAE", axes[0], "Missing Rate -> MAE (lower better)"),
                              ("Corr", axes[1], "Missing Rate -> Corr (higher better)")]:
        for k in keys:
            for pr in PROTOCOLS:
                vals = [rand[(k, pr, p)][metric] for p in cfg.random_test_probs]
                ax.plot(cfg.random_test_probs, vals, color=color[k], marker=marker[k],
                        linestyle="-" if pr == "A" else "--", linewidth=1.8,
                        label=f"{LABEL[k]}-P{pr}")
        ax.set_xlabel("Missing Rate p"); ax.set_ylabel(metric); ax.set_title(title)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(cfg.stage3_dir, "random_missing_curves_stage3.png")
    plt.savefig(out, dpi=150); plt.close()
    return out


# ----------------------------------------------------------------------------
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.stage3_dir, exist_ok=True)

    specs = build_specs(cfg)
    LABEL = {s["key"]: s["label"] for s in specs}
    TYPE = {s["key"]: s["type"] for s in specs}
    hyp = f"hyper_D{cfg.hyper_dim}"
    MAIN3 = ["baseline", "mask_aware", hyp]
    MASKCMP = ["mask_only", "mask_aware", hyp]

    print("=" * 92)
    print("阶段3 实验: Missing-Aware Hyper Representation 验证 (Mask -> Hyper, 暂不含 Dynamic Fusion)")
    print(f"Device={device} | seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} dropout={cfg.dropout}")
    print(f"Baseline/MaskAware: proj={cfg.proj_dim} hidden={cfg.hidden_dim} | "
          f"Hyper: D={cfg.hyper_dim} enc/pred_hidden={cfg.hyper_hidden_dim} | 维度消融={cfg.hyper_dim_ablation}")
    print(f"Protocol B 训练缺失概率={cfg.protocol_b_train_missing_prob} | 随机测试 p={cfg.random_test_probs}")
    print("=" * 92)

    datasets = build_mosi_datasets(cfg.data_path)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])
    print(f"Test samples={n_test} | 待训练模型数={len(specs)*2}")

    trained, infos = train_all_specs(cfg, specs, datasets, device)

    fixed = eval_fixed_s3(trained, specs, test_loader, n_test, device)
    rand = eval_random_s3(trained, TYPE, MAIN3, test_loader, n_test, cfg, device)

    # ---- 表格 ----
    df3a = build_table(fixed, MAIN3, LABEL, COND_NAMES, "MAE")
    df3b = build_table(fixed, MAIN3, LABEL, COND_NAMES, "Corr")
    df3c = build_table(rand, MAIN3, LABEL, cfg.random_test_probs, "MAE")
    df3d = build_table(rand, MAIN3, LABEL, cfg.random_test_probs, "Corr")
    df4 = build_ablation_table(fixed, infos, cfg)
    df5 = build_table(fixed, MASKCMP, LABEL, COND_NAMES, "MAE")

    consistency_check(fixed)
    print_table("[表3a] 三模型 · 固定缺失 -> MAE (↓)   列名=缺失的模态", df3a)
    print_table("[表3b] 三模型 · 固定缺失 -> Corr (↑)", df3b)
    print_table("[表3c] 三模型 · 随机缺失 p -> MAE (↓)", df3c)
    print_table("[表3d] 三模型 · 随机缺失 p -> Corr (↑)", df3d)
    print_table("[表4]  Hyper Dimension Ablation (D=64/128/256/512)", df4)
    print_table("[表5]  Mask-only vs Mask-aware vs Hyper -> MAE (↓)  [3.5]", df5)
    verdict = print_verdict(fixed, cfg)

    # ---- 表征 & 可视化 ----
    fig1, fig2, npz, viz = save_rep_and_plots(trained, cfg, test_loader, n_test, device)
    png = plot_random_curves(rand, MAIN3, LABEL, cfg)

    # ---- 落盘 ----
    df3a.to_csv(os.path.join(cfg.stage3_dir, "fixed_mae_3model.csv"), index=False)
    df3b.to_csv(os.path.join(cfg.stage3_dir, "fixed_corr_3model.csv"), index=False)
    df3c.to_csv(os.path.join(cfg.stage3_dir, "random_mae_3model.csv"), index=False)
    df3d.to_csv(os.path.join(cfg.stage3_dir, "random_corr_3model.csv"), index=False)
    df4.to_csv(os.path.join(cfg.stage3_dir, "hyper_dim_ablation.csv"), index=False)
    df5.to_csv(os.path.join(cfg.stage3_dir, "maskonly_vs_hyper_mae.csv"), index=False)
    dump = {
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size,
                   "proj_dim": cfg.proj_dim, "hidden_dim": cfg.hidden_dim,
                   "hyper_dim": cfg.hyper_dim, "hyper_hidden_dim": cfg.hyper_hidden_dim,
                   "hyper_dim_ablation": cfg.hyper_dim_ablation, "dropout": cfg.dropout,
                   "protocol_b_train_missing_prob": cfg.protocol_b_train_missing_prob,
                   "random_test_probs": cfg.random_test_probs},
        "fixed": {f"{k}|{pr}|{c}": fixed[(k, pr, c)] for (k, pr, c) in fixed},
        "random": {f"{k}|{pr}|{p}": rand[(k, pr, p)] for (k, pr, p) in rand},
        "train_info": {f"{k}|{pr}": {"best_epoch": infos[(k, pr)]["best_epoch"],
                                     "best_valid_mae": infos[(k, pr)]["best_valid_mae"],
                                     "n_params": infos[(k, pr)]["n_params"]} for (k, pr) in infos},
        "verdict": verdict,
        "viz": viz,
    }
    with open(os.path.join(cfg.stage3_dir, "stage3_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 92)
    print(f"曲线      : {png}")
    print(f"表征图1   : {fig1}")
    print(f"表征图2   : {fig2}")
    print(f"z_hyper   : {npz}")
    print(f"结果目录  : {cfg.stage3_dir}")
    print("产物: fixed/random_*_3model.csv, hyper_dim_ablation.csv, maskonly_vs_hyper_mae.csv,")
    print("      stage3_results.json, 3×png, z_hyper_stage3.npz, 14×checkpoint(.pt, 已gitignore)")
    print("=" * 92)


if __name__ == "__main__":
    main()
