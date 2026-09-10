"""
阶段3B 实验编排  (Hyper Representation 修正实验: D -> B1 -> B2)
--------------------------------------------------
目标: 修正阶段3-D 诊断出的两个问题, 并【单独归因】每个改动的贡献:
  ① 文本依赖 (T missing 时 drift≈0.79, 读出 corr 0.68->0.245)  -> B1 可用模态池化
  ② 维度塌缩 (有效秩 1.5)                                      -> B2 在 B1 上加 LN+Residual

模型 (4): MaskAware / H0 原始Hyper / H1 可用模态池化 / H2 池化+LN+Res
协议: 与阶段2/3 完全一致 (seed=42, 同划分, Protocol A/B, 仅 valid 选模型, 只用 test 评估)。
交付:
  [表]  4 模型 × Protocol A/B × 7 固定条件 的 MAE / Corr
  [诊断] 重导 z_hyper, 重算 silhouette / 探针 corr,R2 / kNN / 缺失漂移+迁移 / 有效秩
  [图]  t-SNE: H0/H1/H2 情感着色; H0 vs H2 缺失模式着色
  [判定] 4 项成功标准 (相对 H0 与 MaskAware), 全满足才进入阶段4
禁令: Attention / Dynamic Fusion / MCAC / CCL / KL / Missing Embedding / 门控。
运行:  python experiments/run_stage3b.py
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

from datasets import build_mosi_datasets                                # noqa: E402
from configs.config import Config                                       # noqa: E402
from utils import generate_missing_mask                                # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS               # noqa: E402
from experiments.stage3_lib import (evaluate_with_mask_s3,             # noqa: E402
                                    extract_representation_s3)
from experiments.stage3b_lib import train_stage3b_model                 # noqa: E402
from experiments.diagnose_zhyper import (ridge_cv, ridge_fit, ridge_apply,  # noqa: E402
                                         corr as dcorr, knn_label_consistency,
                                         silhouette_binary, eff_rank, tsne)

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

PROTOCOLS = ("A", "B")
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
HARD = ["T missing", "T+A missing", "T+V missing"]
SPECS = [
    {"key": "mask_aware", "type": "mask_aware", "label": "MaskAware(B)"},
    {"key": "hyper_h0", "type": "hyper_h0", "label": "H0 Original"},
    {"key": "hyper_h1", "type": "hyper_h1", "label": "H1 AvailPool"},
    {"key": "hyper_h2", "type": "hyper_h2", "label": "H2 Pool+LN/Res"},
]
LABEL = {s["key"]: s["label"] for s in SPECS}
TYPE = {s["key"]: s["type"] for s in SPECS}
HYPER_KEYS = ["hyper_h0", "hyper_h1", "hyper_h2"]
# 阶段3-D 对 H0 的参考值 (用于连续性对照, 判定以本轮 in-run H0 为准)
REF_H0 = {"transfer_Tmiss": 0.245, "drift_ratio_Tmiss": 0.79, "eff_rank": 1.5}


def _avg(seq):
    return float(np.mean(seq))


# ----------------------------------------------------------------------------
# 训练 & 评估
# ----------------------------------------------------------------------------
def train_all(cfg, datasets, device):
    trained, infos = {}, {}
    for s in SPECS:
        for pr in PROTOCOLS:
            print(f"\n>>> 训练 {s['label']} + Protocol {pr} ...")
            model, info = train_stage3b_model(s["type"], pr, cfg, datasets, device)
            trained[(s["key"], pr)] = model
            infos[(s["key"], pr)] = info
            h = info["history"]
            print(f"    train_loss {h[0]['train_loss']:.4f} -> {h[-1]['train_loss']:.4f} | "
                  f"best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f} "
                  f"| #params={info['n_params']:,}")
            torch.save({"model_state_dict": model.state_dict(), "spec_key": s["key"],
                        "model_type": s["type"], "protocol": pr,
                        "best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                        "n_params": info["n_params"]},
                       os.path.join(cfg.stage3b_dir, f"{s['key']}_protocol{pr}.pt"))
    return trained, infos


def eval_fixed(trained, test_loader, n_test, device):
    res = {}
    for s in SPECS:
        for pr in PROTOCOLS:
            model = trained[(s["key"], pr)]
            for cond_name, pattern in FIXED_TEST_CONDITIONS:
                mask = generate_missing_mask(n_test, missing_pattern=pattern)
                _, m = evaluate_with_mask_s3(model, s["type"], test_loader, mask, device)
                res[(s["key"], pr, cond_name)] = m
    return res


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


# ----------------------------------------------------------------------------
# 表征诊断 (复用阶段3-D 口径)
# ----------------------------------------------------------------------------
def diag_variant(zA, Z_by_pat, y, pnames):
    """对单个 Hyper 变体计算四类诊断: 完整模态质量 + 缺失漂移/迁移。"""
    pc, r2, _ = ridge_cv(zA, y)
    ss, dl = knn_label_consistency(zA, y, k=10)
    er = eff_rank(zA)
    sil = silhouette_binary(zA, (y > 0))
    Znone = Z_by_pat[0]
    from scipy.spatial.distance import cdist
    med = float(np.median(cdist(Znone, Znone)))
    p0 = ridge_fit(Znone, y)
    base = dcorr(ridge_apply(p0, Znone), y)
    per_pat = {}
    for p, name in enumerate(pnames):
        Zp = Z_by_pat[p]
        drift = float(np.linalg.norm(Zp - Znone, axis=1).mean())
        per_pat[name] = dict(drift=drift, drift_ratio=drift / (med + 1e-9),
                             transfer=dcorr(ridge_apply(p0, Zp), y))
    return dict(probe_corr=pc, probe_r2=r2, knn_same_sign=ss, knn_mean_dlabel=dl,
                eff_rank=er, sil_binary=sil, none_probe_corr=base,
                median_pair_dist=med, per_pattern=per_pat)


def extract_all_reps(trained, cfg, test_loader, n_test, device):
    """重导 z_hyper: 每变体 Protocol A(None) + Protocol B(7 模式); 另存 maskA 参考。"""
    none_mask = generate_missing_mask(n_test, missing_pattern="T+A+V")
    out = {}
    maskA = extract_representation_s3(trained[("mask_aware", "A")], "mask_aware",
                                      test_loader, none_mask, device)
    out["maskA"] = maskA
    for k in HYPER_KEYS:
        zA = extract_representation_s3(trained[(k, "A")], TYPE[k], test_loader, none_mask, device)
        zs, pidx = [], []
        for i, (cond_name, pattern) in enumerate(FIXED_TEST_CONDITIONS):
            m = generate_missing_mask(n_test, missing_pattern=pattern)
            o = extract_representation_s3(trained[(k, "B")], TYPE[k], test_loader, m, device)
            zs.append(o["rep"]); pidx.append(np.full(n_test, i, dtype=np.int64))
        out[k] = {"zA": zA, "Z_by_pat": np.stack(zs, axis=0), "pidx": np.concatenate(pidx)}
    return out


def save_npz_and_plots(reps, diag, cfg):
    y = reps["maskA"]["label"]
    pnames = COND_NAMES
    save_kw = {"labels": y,
               "maskA_none_h": reps["maskA"]["rep"], "maskA_none_pred": reps["maskA"]["pred"],
               "pattern_names": np.array(pnames),
               "meta": np.array([f"hyper_dim={cfg.hyper_dim}", f"seed={cfg.seed}", "stage3B"])}
    for k in HYPER_KEYS:
        save_kw[f"{k}_A_none_z"] = reps[k]["zA"]["rep"]
        save_kw[f"{k}_A_none_pred"] = reps[k]["zA"]["pred"]
        Z = reps[k]["Z_by_pat"]
        save_kw[f"{k}_B_all_z"] = Z.reshape(-1, Z.shape[-1])
        save_kw[f"{k}_B_all_idx"] = reps[k]["pidx"]
    npz = os.path.join(cfg.stage3b_dir, "z_hyper_stage3b.npz")
    np.savez_compressed(npz, **save_kw)

    # 图1: t-SNE 情感着色 H0/H1/H2 (Protocol A, None)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, k in zip(axes, HYPER_KEYS):
        emb = tsne(reps[k]["zA"]["rep"])
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="coolwarm", s=12, alpha=0.8)
        ax.set_title(f"{LABEL[k]}  (eff_rank={diag[k]['eff_rank']:.1f}, sil={diag[k]['sil_binary']:+.3f})")
        ax.set_xlabel("t-SNE1"); ax.set_ylabel("t-SNE2")
        plt.colorbar(sc, ax=ax, label="label")
    fig.suptitle("Stage3B: z_hyper t-SNE colored by sentiment (Protocol A, full-modal)", fontsize=11)
    plt.tight_layout()
    f1 = os.path.join(cfg.stage3b_dir, "stage3b_tsne_sentiment.png")
    fig.savefig(f1, dpi=130); plt.close(fig)

    # 图2: t-SNE 缺失模式着色 H0 vs H2 (Protocol B, 每模式抽样 200)
    per, rng = 200, np.random.default_rng(cfg.seed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, k in zip(axes, ["hyper_h0", "hyper_h2"]):
        Z = reps[k]["Z_by_pat"]
        subs, plab = [], []
        for p in range(len(pnames)):
            take = rng.choice(len(y), size=min(per, len(y)), replace=False)
            subs.append(Z[p][take]); plab.append(np.full(len(take), p))
        Xsub = np.concatenate(subs); plab = np.concatenate(plab)
        emb = tsne(Xsub)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=plab, cmap="tab10", s=12, alpha=0.8)
        tr = diag[k]["per_pattern"]["T missing"]["transfer"]
        dr = diag[k]["per_pattern"]["T missing"]["drift_ratio"]
        ax.set_title(f"{LABEL[k]} across patterns\nT-miss transfer={tr:+.3f} drift={dr:.2f}")
        ax.set_xlabel("t-SNE1"); ax.set_ylabel("t-SNE2")
        plt.colorbar(sc, ax=ax, label="pattern")
    fig.suptitle("Stage3B: z_hyper t-SNE colored by missing pattern (Protocol B)", fontsize=11)
    plt.tight_layout()
    f2 = os.path.join(cfg.stage3b_dir, "stage3b_tsne_patterns.png")
    fig.savefig(f2, dpi=130); plt.close(fig)
    return npz, f1, f2


# ----------------------------------------------------------------------------
# 判定 (阶段3B 成功标准, 数据驱动)
# ----------------------------------------------------------------------------
def print_verdict(fixed, diag):
    h0, h1, h2 = diag["hyper_h0"], diag["hyper_h1"], diag["hyper_h2"]
    print("\n" + "=" * 100)
    print("[阶段3B 诊断对比] H0 vs H1 vs H2  (参考: 阶段3-D 的 H0 = transfer0.245 / drift0.79 / rank1.5)")
    print("=" * 100)
    hdr = f"  {'variant':14s} | {'probe_corr':>10s} {'R2':>7s} {'sil':>7s} {'kNNsign':>8s} {'eff_rank':>8s} | " \
          f"{'Tmiss_transfer':>14s} {'Tmiss_drift':>11s}"
    print(hdr)
    for k in HYPER_KEYS:
        d = diag[k]
        tp = d["per_pattern"]["T missing"]
        print(f"  {LABEL[k]:14s} | {d['probe_corr']:+10.3f} {d['probe_r2']:+7.3f} {d['sil_binary']:+7.3f} "
              f"{d['knn_same_sign']:8.3f} {d['eff_rank']:8.1f} | {tp['transfer']:+14.3f} {tp['drift_ratio']:11.2f}")
    print("\n  全模式漂移/迁移明细 (drift_ratio = 漂移/样本间中位距):")
    for k in HYPER_KEYS:
        dd = diag[k]   # 注意: 必须用本变体的 dd, 不能用上一循环泄漏的 d
        row = "  " + LABEL[k] + ": "
        row += " | ".join(f"{nm.split()[0]}:dr{dd['per_pattern'][nm]['drift_ratio']:.2f}/tr{dd['per_pattern'][nm]['transfer']:+.2f}"
                          for nm in MISS)
        print(row)

    print("\n" + "=" * 100)
    print("[阶段3B 成功标准] (相对 in-run H0 与 MaskAware; 不预设 MAE 阈值)")
    print("=" * 100)
    c1 = h2["per_pattern"]["T missing"]["transfer"] > h0["per_pattern"]["T missing"]["transfer"]
    c2 = h2["per_pattern"]["T missing"]["drift_ratio"] < h0["per_pattern"]["T missing"]["drift_ratio"]
    c3 = (h2["eff_rank"] > h0["eff_rank"]) and (h2["probe_corr"] >= h0["probe_corr"] - 0.05)
    # 预测: 两协议下 H2 困难缺失均值 & 缺失均值 均优于 MaskAware 与 H0
    c4 = True
    for pr in PROTOCOLS:
        for other in ["mask_aware", "hyper_h0"]:
            h2_hard = _avg([fixed[("hyper_h2", pr, c)]["MAE"] for c in HARD])
            ot_hard = _avg([fixed[(other, pr, c)]["MAE"] for c in HARD])
            h2_miss = _avg([fixed[("hyper_h2", pr, c)]["MAE"] for c in MISS])
            ot_miss = _avg([fixed[(other, pr, c)]["MAE"] for c in MISS])
            ok = (h2_hard < ot_hard) and (h2_miss < ot_miss)
            c4 = c4 and ok
            print(f"  P{pr}: H2 vs {LABEL[other]:14s} 困难缺失 {h2_hard:.4f} vs {ot_hard:.4f} | "
                  f"缺失均值 {h2_miss:.4f} vs {ot_miss:.4f} -> {'✅' if ok else '❌'}")
    print(f"\n  判据: (1)T-miss迁移corr H2>H0: {c1}  (2)T-miss漂移 H2<H0: {c2}  "
          f"(3)有效秩升且情感保留: {c3}  (4)预测困难/缺失均值 H2 最优: {c4}")
    go = c1 and c2 and c3 and c4
    if go:
        print("  ✅ 结论: B2 同时改善表征鲁棒性与预测 -> 支持进入阶段4 (Hyper-Guided Dynamic Fusion)。")
    else:
        print("  ⚠️ 结论: 尚未同时满足四项 -> 暂不进入阶段4; 视缺失项决定继续修正 B 或调整归因。")
    return {"criteria": {"c1_transfer": bool(c1), "c2_drift": bool(c2),
                         "c3_rank": bool(c3), "c4_pred": bool(c4)},
            "go_stage4": bool(go)}


# ----------------------------------------------------------------------------
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.stage3b_dir, exist_ok=True)

    print("=" * 100)
    print("阶段3B 实验: Hyper Representation 修正 (D -> B1 -> B2), 暂不含 Dynamic Fusion")
    print(f"Device={device} | seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} "
          f"dropout={cfg.dropout} | hyper D={cfg.hyper_dim} hidden={cfg.hyper_hidden_dim}")
    print("=" * 100)

    datasets = build_mosi_datasets(cfg.data_path)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])

    trained, infos = train_all(cfg, datasets, device)
    fixed = eval_fixed(trained, test_loader, n_test, device)

    keys = [s["key"] for s in SPECS]
    df_mae_a = build_table(fixed, keys, COND_NAMES, "MAE")
    df_corr_a = build_table(fixed, keys, COND_NAMES, "Corr")
    print_table("[表3B-MAE] 4 模型 × 7 固定条件 × Protocol A/B  (MAE ↓)", df_mae_a)
    print_table("[表3B-Corr] 4 模型 × 7 固定条件 × Protocol A/B  (Corr ↑)", df_corr_a)

    # 一致性: 重训 mask_aware / h0 的 PA-None 应与阶段3 一致 (1.0308 / 1.0245)
    m_ma = fixed[("mask_aware", "A", "None")]; m_h0 = fixed[("hyper_h0", "A", "None")]
    ok_ma = abs(m_ma["MAE"] - 1.0308) < 1e-3 and abs(m_ma["Corr"] - 0.6160) < 5e-2
    ok_h0 = abs(m_h0["MAE"] - 1.0245) < 1e-3
    print("\n[一致性校验] 重训 MaskAware-PA@None MAE=%.4f Corr=%.4f (阶段3≈1.0308) -> %s"
          % (m_ma["MAE"], m_ma["Corr"], "✅" if ok_ma else "⚠️"))
    print("[一致性校验] 重训 H0-PA@None     MAE=%.4f Corr=%.4f (阶段3≈1.0245) -> %s"
          % (m_h0["MAE"], m_h0["Corr"], "✅" if ok_h0 else "⚠️"))

    reps = extract_all_reps(trained, cfg, test_loader, n_test, device)
    y = reps["maskA"]["label"]
    diag = {}
    for k in HYPER_KEYS:
        diag[k] = diag_variant(reps[k]["zA"]["rep"], reps[k]["Z_by_pat"], y, COND_NAMES)
    # mask_aware 参考 (完整模态质量)
    pc, r2, _ = ridge_cv(reps["maskA"]["rep"], y)
    diag["mask_aware"] = dict(probe_corr=pc, probe_r2=r2,
                              eff_rank=eff_rank(reps["maskA"]["rep"]),
                              sil_binary=silhouette_binary(reps["maskA"]["rep"], (y > 0)))

    npz, f1, f2 = save_npz_and_plots(reps, diag, cfg)
    verdict = print_verdict(fixed, diag)

    df_mae_a.to_csv(os.path.join(cfg.stage3b_dir, "fixed_mae_4model.csv"), index=False)
    df_corr_a.to_csv(os.path.join(cfg.stage3b_dir, "fixed_corr_4model.csv"), index=False)
    dump = {
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr,
                   "batch_size": cfg.batch_size, "hyper_dim": cfg.hyper_dim,
                   "hyper_hidden_dim": cfg.hyper_hidden_dim, "dropout": cfg.dropout,
                   "protocol_b_train_missing_prob": cfg.protocol_b_train_missing_prob},
        "fixed": {f"{k}|{pr}|{c}": fixed[(k, pr, c)] for (k, pr, c) in fixed},
        "train_info": {f"{k}|{pr}": {"best_epoch": infos[(k, pr)]["best_epoch"],
                                     "best_valid_mae": infos[(k, pr)]["best_valid_mae"],
                                     "n_params": infos[(k, pr)]["n_params"]} for (k, pr) in infos},
        "diag": diag,
        "ref_H0_stage3D": REF_H0,
        "verdict": verdict,
    }
    with open(os.path.join(cfg.stage3b_dir, "stage3b_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 100)
    print(f"t-SNE 情感图 : {f1}")
    print(f"t-SNE 模式图 : {f2}")
    print(f"z_hyper      : {npz}")
    print(f"结果目录     : {cfg.stage3b_dir}")
    print("产物: fixed_mae/corr_4model.csv, stage3b_results.json, 2×png, z_hyper_stage3b.npz, 8×checkpoint")
    print("=" * 100)


if __name__ == "__main__":
    main()
