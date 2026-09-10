"""
MOSEI 跨数据集验证编排  (MOSEI-1)
--------------------------------------------------
目的: 判断 MOSI 上观察到的两个现象是否可跨数据集复现:
  ① 文本缺失是否仍然特别严重 (数据属性 vs 模型缺陷)
  ② Hyper 是否仍出现 "文本依赖" (drift_T >> drift_A/V, transfer_T 低)
  ③ Hyper 是否在 A/V/A+V 缺失下稳定优于 Mask
  ④ Protocol A 与 B 趋势是否一致 (缺失分布迁移 vs 结构问题)

模型 (仅 4 个, 不做 H1/H2/B3/DynamicFusion/MCAC/CCL/KL):
  M0/M1 = Full-modal Baseline (完整/缺失测试)  M2 = MaskAware  M3 = H0 Original Hyper
协议: 与 MOSI 完全一致 (seed=42, 同划分规则, Protocol A/B, 仅 valid 选模型, 只用 test 评估)。
固定缺失: None + T/A/V/T+A/T+V/A+V。
自本阶段起启用 ExperimentRecorder: 每次评估完整记录
  Dataset/Protocol/Model/Seed/MissingPattern/MissingProbability/BestEpoch/MAE/Corr。
运行:  python experiments/run_mosei.py
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

from datasets import build_mosei_datasets, mosei_dims                       # noqa: E402
from configs.config import Config                                           # noqa: E402
from utils import generate_missing_mask, ExperimentRecorder                 # noqa: E402
from experiments.stage2_lib import FIXED_TEST_CONDITIONS                    # noqa: E402
from experiments.stage3_lib import (train_stage3_model, evaluate_with_mask_s3,  # noqa: E402
                                    extract_representation_s3)
from experiments.diagnose_zhyper import (ridge_cv, ridge_fit, ridge_apply,  # noqa: E402
                                         corr as dcorr, knn_label_consistency,
                                         silhouette_binary, eff_rank, tsne)
from scipy.spatial.distance import cdist                                    # noqa: E402

import matplotlib                                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402

PROTOCOLS = ("A", "B")
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
HARD = ["T missing", "T+A missing", "T+V missing"]
NONTEXT = ["A missing", "V missing", "A+V missing"]
SPECS = [
    {"key": "baseline", "type": "baseline", "label": "M0/M1 Baseline"},
    {"key": "mask_aware", "type": "mask_aware", "label": "M2 MaskAware"},
    {"key": "hyper", "type": "hyper", "label": "M3 H0 Hyper"},
]
LABEL = {s["key"]: s["label"] for s in SPECS}
# MOSI 阶段3 参考值 (仅用于跨数据集对照, 明确标注来源)
MOSI_REF = {
    "mask_Tmiss_Corr_PA": 0.1917, "hyper_Tmiss_Corr_PA": 0.1452,
    "mask_missavg_MAE_PA": 1.2709, "hyper_missavg_MAE_PA": 1.2993,
    "mask_missavg_MAE_PB": 1.2708, "hyper_missavg_MAE_PB": 1.3158,
    "hyper_drift_T": 0.79, "hyper_transfer_T": 0.245, "hyper_eff_rank": 1.5,
}


def _avg(seq):
    return float(np.mean(seq))


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


def diag_hyper(zA, Z_by_pat, y, pnames):
    pc, r2, _ = ridge_cv(zA, y)
    ss, dl = knn_label_consistency(zA, y, k=10)
    er = eff_rank(zA)
    sil = silhouette_binary(zA, (y > 0))
    Znone = Z_by_pat[0]
    med = float(np.median(cdist(Znone, Znone)))
    p0 = ridge_fit(Znone, y)
    per_pat = {}
    for p, name in enumerate(pnames):
        Zp = Z_by_pat[p]
        drift = float(np.linalg.norm(Zp - Znone, axis=1).mean())
        per_pat[name] = dict(drift=drift, drift_ratio=drift / (med + 1e-9),
                             transfer=dcorr(ridge_apply(p0, Zp), y))
    return dict(probe_corr=pc, probe_r2=r2, knn_same_sign=ss, knn_mean_dlabel=dl,
                eff_rank=er, sil_binary=sil, median_pair_dist=med, per_pattern=per_pat)


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.mosei_dir, exist_ok=True)

    datasets = build_mosei_datasets(cfg.mosei_data_path, cfg.mosei_conv_dir,
                                    inf_policy=cfg.mosei_inf_policy)
    Dt, Da, Dv, L = mosei_dims(datasets)
    # 运行时覆盖维度, 保证模型/协议与数据一致 (不在 config 写死)
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = Dt, Da, Dv, L

    print("=" * 100)
    print("MOSEI 跨数据集验证 (MOSEI-1): M0/M1 Baseline + M2 MaskAware + M3 H0 Hyper")
    print(f"Device={device} | seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size}")
    print(f"探测维度: text={Dt} audio={Da} vision={Dv} seq_len={L} | hyper D={cfg.hyper_dim}")
    print(f"split 大小: train={len(datasets['train'])} valid={len(datasets['valid'])} test={len(datasets['test'])}")
    print("=" * 100)

    rec = ExperimentRecorder(cfg.mosei_dir)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])

    # ---- 训练 ----
    trained, infos = {}, {}
    for s in SPECS:
        for pr in PROTOCOLS:
            print(f"\n>>> 训练 {s['label']} + Protocol {pr} ...")
            hd = cfg.hyper_dim if s["type"] == "hyper" else None
            model, info = train_stage3_model(s["type"], pr, cfg, datasets, device, hyper_dim=hd)
            trained[(s["key"], pr)] = model
            infos[(s["key"], pr)] = info
            print(f"    best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f} "
                  f"| #params={info['n_params']:,}")
            torch.save({"model_state_dict": model.state_dict(), "spec_key": s["key"],
                        "model_type": s["type"], "protocol": pr,
                        "best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                        "n_params": info["n_params"]},
                       os.path.join(cfg.mosei_dir, f"{s['key']}_protocol{pr}.pt"))

    # ---- 固定 7 条件评估 + 记录 ----
    fixed = {}
    for s in SPECS:
        for pr in PROTOCOLS:
            info = infos[(s["key"], pr)]
            for cond_name, pattern in FIXED_TEST_CONDITIONS:
                mask = generate_missing_mask(n_test, missing_pattern=pattern)
                _, m = evaluate_with_mask_s3(trained[(s["key"], pr)], s["type"],
                                             test_loader, mask, device)
                fixed[(s["key"], pr, cond_name)] = m
                rec.log(dataset="MOSEI", protocol=pr, model=LABEL[s["key"]], seed=cfg.seed,
                        missing_pattern=cond_name,
                        missing_probability=(0.0 if pr == "A" else cfg.protocol_b_train_missing_prob),
                        best_epoch=info["best_epoch"], n_params=info["n_params"],
                        MAE=m["MAE"], Corr=m["Corr"])

    keys = [s["key"] for s in SPECS]
    df_mae = build_table(fixed, keys, COND_NAMES, "MAE")
    df_corr = build_table(fixed, keys, COND_NAMES, "Corr")
    print_table("[MOSEI-MAE] 3 模型 × 7 固定条件 × Protocol A/B  (MAE ↓)", df_mae)
    print_table("[MOSEI-Corr] 3 模型 × 7 固定条件 × Protocol A/B  (Corr ↑)", df_corr)

    # ---- H0 z_hyper 诊断 ----
    none_mask = generate_missing_mask(n_test, missing_pattern="T+A+V")
    zA = extract_representation_s3(trained[("hyper", "A")], "hyper", test_loader, none_mask, device)
    zs, pidx = [], []
    for i, (cond_name, pattern) in enumerate(FIXED_TEST_CONDITIONS):
        m = generate_missing_mask(n_test, missing_pattern=pattern)
        o = extract_representation_s3(trained[("hyper", "B")], "hyper", test_loader, m, device)
        zs.append(o["rep"]); pidx.append(np.full(n_test, i, dtype=np.int64))
    Z_by_pat = np.stack(zs, axis=0)
    y = zA["label"]
    diag = diag_hyper(zA["rep"], Z_by_pat, y, COND_NAMES)
    maskA = extract_representation_s3(trained[("mask_aware", "A")], "mask_aware",
                                      test_loader, none_mask, device)
    m_pc, m_r2, _ = ridge_cv(maskA["rep"], y)
    diag_mask = dict(probe_corr=m_pc, probe_r2=m_r2, eff_rank=eff_rank(maskA["rep"]),
                     sil_binary=silhouette_binary(maskA["rep"], (y > 0)))

    print("\n" + "=" * 100)
    print("[MOSEI 表征诊断] H0 z_hyper (完整模态质量 + 缺失漂移/迁移)")
    print("=" * 100)
    print(f"  Hyper : probe corr={diag['probe_corr']:+.3f} R2={diag['probe_r2']:+.3f} "
          f"sil={diag['sil_binary']:+.3f} kNNsign={diag['knn_same_sign']:.3f} eff_rank={diag['eff_rank']:.1f}")
    print(f"  Mask  : probe corr={diag_mask['probe_corr']:+.3f} R2={diag_mask['probe_r2']:+.3f} "
          f"sil={diag_mask['sil_binary']:+.3f} eff_rank={diag_mask['eff_rank']:.1f}")
    for nm in MISS:
        pp = diag["per_pattern"][nm]
        print(f"    {nm:12s} drift={pp['drift']:6.2f} (x中位距 {pp['drift_ratio']:.2f}) transfer={pp['transfer']:+.3f}")

    # ---- 四个问题 ----
    print("\n" + "=" * 100)
    print("[MOSEI 四问]")
    print("=" * 100)
    q1 = {}
    for s in SPECS:
        for pr in PROTOCOLS:
            worst = max(MISS, key=lambda c: fixed[(s["key"], pr, c)]["MAE"])
            q1[(s["key"], pr)] = worst
            print(f"  ① {LABEL[s['key']]:16s} P{pr}: 最差缺失条件={worst} "
                  f"(MAE={fixed[(s['key'], pr, worst)]['MAE']:.4f})")
    dt = diag["per_pattern"]["T missing"]; da = diag["per_pattern"]["A missing"]; dv = diag["per_pattern"]["V missing"]
    q2 = (dt["drift_ratio"] > max(da["drift_ratio"], dv["drift_ratio"])) and (dt["transfer"] < min(da["transfer"], dv["transfer"]))
    print(f"  ② 文本依赖: drift_T={dt['drift_ratio']:.2f} vs A={da['drift_ratio']:.2f}/V={dv['drift_ratio']:.2f}; "
          f"transfer_T={dt['transfer']:+.3f} vs A={da['transfer']:+.3f}/V={dv['transfer']:+.3f} -> {'是' if q2 else '否'}")
    q3 = {}
    for pr in PROTOCOLS:
        wins = sum(1 for c in NONTEXT if fixed[("hyper", pr, c)]["MAE"] < fixed[("mask_aware", pr, c)]["MAE"])
        delta = _avg([fixed[("mask_aware", pr, c)]["MAE"] - fixed[("hyper", pr, c)]["MAE"] for c in NONTEXT])
        q3[pr] = dict(wins=wins, delta=delta)
        print(f"  ③ P{pr}: Hyper 在 A/V/A+V 胜 {wins}/3, 平均 MAE 差(mask-hyper)={delta:+.4f}")
    pa_delta = _avg([fixed[("hyper", "A", c)]["MAE"] - fixed[("mask_aware", "A", c)]["MAE"] for c in MISS])
    pb_delta = _avg([fixed[("hyper", "B", c)]["MAE"] - fixed[("mask_aware", "B", c)]["MAE"] for c in MISS])
    q4 = dict(pa_delta=pa_delta, pb_delta=pb_delta,
              t_worst_PA=q1[("hyper", "A")] in HARD, t_worst_PB=q1[("hyper", "B")] in HARD)
    print(f"  ④ PA 缺失均值差(hyper-mask)={pa_delta:+.4f} | PB={pb_delta:+.4f} | "
          f"T-系最差: PA={q4['t_worst_PA']} PB={q4['t_worst_PB']}")

    # ---- 跨数据集对照 ----
    print("\n" + "=" * 100)
    print("[跨数据集对照] MOSI(阶段3) vs MOSEI(本轮)")
    print("=" * 100)
    mo_mask_Tcorr = fixed[("mask_aware", "A", "T missing")]["Corr"]
    mo_hyper_Tcorr = fixed[("hyper", "A", "T missing")]["Corr"]
    mo_mask_missA = _avg([fixed[("mask_aware", "A", c)]["MAE"] for c in MISS])
    mo_hyper_missA = _avg([fixed[("hyper", "A", c)]["MAE"] for c in MISS])
    mo_mask_missB = _avg([fixed[("mask_aware", "B", c)]["MAE"] for c in MISS])
    mo_hyper_missB = _avg([fixed[("hyper", "B", c)]["MAE"] for c in MISS])
    rows = [
        ("Mask T-miss Corr (PA)", MOSI_REF["mask_Tmiss_Corr_PA"], mo_mask_Tcorr),
        ("Hyper T-miss Corr (PA)", MOSI_REF["hyper_Tmiss_Corr_PA"], mo_hyper_Tcorr),
        ("Mask 缺失均值 MAE (PA)", MOSI_REF["mask_missavg_MAE_PA"], mo_mask_missA),
        ("Hyper 缺失均值 MAE (PA)", MOSI_REF["hyper_missavg_MAE_PA"], mo_hyper_missA),
        ("Mask 缺失均值 MAE (PB)", MOSI_REF["mask_missavg_MAE_PB"], mo_mask_missB),
        ("Hyper 缺失均值 MAE (PB)", MOSI_REF["hyper_missavg_MAE_PB"], mo_hyper_missB),
        ("Hyper drift_T (x中位距)", MOSI_REF["hyper_drift_T"], dt["drift_ratio"]),
        ("Hyper transfer_T", MOSI_REF["hyper_transfer_T"], dt["transfer"]),
        ("Hyper eff_rank", MOSI_REF["hyper_eff_rank"], diag["eff_rank"]),
    ]
    dfx = pd.DataFrame([{"指标": n, "MOSI": round(a, 4), "MOSEI": round(b, 4)} for n, a, b in rows])
    print(dfx.to_string(index=False))

    # ---- 图 & npz ----
    rng = np.random.default_rng(cfg.seed)
    sub = rng.choice(len(y), size=min(1500, len(y)), replace=False)
    emb = tsne(zA["rep"][sub])
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y[sub], cmap="coolwarm", s=12, alpha=0.8)
    ax.set_title(f"MOSEI H0 z_hyper t-SNE (sentiment)\neff_rank={diag['eff_rank']:.1f} sil={diag['sil_binary']:+.3f}")
    plt.colorbar(sc, ax=ax, label="label")
    f1 = os.path.join(cfg.mosei_dir, "mosei_tsne_sentiment.png")
    fig.savefig(f1, dpi=130); plt.close(fig)

    per = 200
    subs, plab = [], []
    for p in range(len(COND_NAMES)):
        take = rng.choice(n_test, size=min(per, n_test), replace=False)
        subs.append(Z_by_pat[p][take]); plab.append(np.full(len(take), p))
    Xsub = np.concatenate(subs); plab = np.concatenate(plab)
    emb2 = tsne(Xsub)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb2[:, 0], emb2[:, 1], c=plab, cmap="tab10", s=12, alpha=0.8)
    ax.set_title(f"MOSEI H0 z_hyper across patterns\nT-miss transfer={dt['transfer']:+.3f} drift={dt['drift_ratio']:.2f}")
    plt.colorbar(sc, ax=ax, label="pattern")
    f2 = os.path.join(cfg.mosei_dir, "mosei_tsne_patterns.png")
    fig.savefig(f2, dpi=130); plt.close(fig)

    npz = os.path.join(cfg.mosei_dir, "z_hyper_mosei.npz")
    np.savez_compressed(npz, labels=y,
                        hyperA_none_z=zA["rep"], hyperA_none_pred=zA["pred"],
                        maskA_none_h=maskA["rep"], maskA_none_pred=maskA["pred"],
                        hyperB_all_patterns_z=Z_by_pat.reshape(-1, Z_by_pat.shape[-1]),
                        hyperB_all_patterns_idx=np.concatenate(pidx),
                        pattern_names=np.array(COND_NAMES),
                        meta=np.array([f"dims={Dt},{Da},{Dv},{L}", f"seed={cfg.seed}", "MOSEI-1"]))

    # ---- 落盘 ----
    df_mae.to_csv(os.path.join(cfg.mosei_dir, "fixed_mae_mosei.csv"), index=False)
    df_corr.to_csv(os.path.join(cfg.mosei_dir, "fixed_corr_mosei.csv"), index=False)
    dump = {
        "config": {"dataset": "MOSEI", "seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr,
                   "batch_size": cfg.batch_size, "dims": [Dt, Da, Dv, L],
                   "hyper_dim": cfg.hyper_dim, "protocol_b_train_missing_prob": cfg.protocol_b_train_missing_prob},
        "fixed": {f"{k}|{pr}|{c}": fixed[(k, pr, c)] for (k, pr, c) in fixed},
        "train_info": {f"{k}|{pr}": {"best_epoch": infos[(k, pr)]["best_epoch"],
                                     "best_valid_mae": infos[(k, pr)]["best_valid_mae"],
                                     "n_params": infos[(k, pr)]["n_params"]} for (k, pr) in infos},
        "diag_hyper": diag, "diag_mask": diag_mask,
        "questions": {"q1_worst_condition": {f"{k}|{pr}": v for (k, pr), v in q1.items()},
                      "q2_text_dependency": bool(q2), "q3_nontext": q3, "q4_protocol_consistency": q4},
        "cross_dataset": [{"metric": n, "MOSI": a, "MOSEI": b} for n, a, b in rows],
    }
    with open(os.path.join(cfg.mosei_dir, "mosei_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 100)
    print(f"t-SNE 情感图 : {f1}")
    print(f"t-SNE 模式图 : {f2}")
    print(f"z_hyper      : {npz}")
    print(f"实验记录     : {rec.csv} / {rec.jsonl}")
    print(f"结果目录     : {cfg.mosei_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
