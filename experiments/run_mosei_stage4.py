"""
Stage 4 · Hyper-Guided Dynamic Fusion (MOSEI · Protocol B · 30ep · F0/F1/F2/F3)
--------------------------------------------------
用户 2026-09-05 裁决: B3 封版 (λ_cons=0.005, normalized stop-gradient), 进入 Stage 4。
本轮【只】验证一个假设: z_hyper 能否有效指导动态模态融合? 严格最小化, 不加任何新组件
(无 MCAC / CCL / KL / MoE / 专家网络 / Prompt / Reliability Network)。

固定 (全部继承封版 B3, 不改):
  z_hyper 仍由【未修改】的 HyperEncoder 产生, 继续受 normalized stop-gradient consistency
  (λ=0.005, ẑ=z/‖z‖, L_cons=‖sg(ẑ_full)−ẑ_miss‖², 单模态缺失视图) 约束; seed=42, epochs=30,
  bs=32, lr=1e-3, Protocol B, 仅 valid MAE 选模, test 仅评估。

四对照 (唯一变量 = 融合头 α 的来源; backbone / V_i / h_fuse / consistency 完全共享):
  F0 = H0            z_hyper -> Predictor                          (= 封版 B3, 无融合; 应逐位复现 λ=0.005)
  F1 = MaskGate      α = softmax(mask · MLP([x_T,x_A,x_V,m]))      (简单 mask-aware gate, 不用 z_hyper)
  F2 = GlobalQuery   α = softmax(mask · q_global·K_i/√D)           (静态可学习 query, 样本无关)
  F3 = HyperGuided   α = softmax(mask · (W_Q z_hyper)·K_i/√D)      (样本级 hyper-guided, 本项创新)
  融合: h_fuse = z_hyper + Σ_i α_i V_i,  V_i = W_V x_i,  缺失位 logit=-inf -> α_missing 严格 0。

核心比较 = F3 vs F1 (Hyper Representation 是否比 Mask Gate 更适合做融合条件), 及 F3 vs F0。
理想: F3 > F1 > F0, 尤其在 T-missing / T+A / T+V / A+V 困难场景。
必做: 保存 per-sample α_T/α_A/α_V; 审计缺失模态权重严格归零; 分析 None/单缺/双缺 的 α 分布 + 热力图。
运行:  python experiments/run_mosei_stage4.py
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
from experiments.stage4_lib import (extract_alpha_s4, alpha_missing_audit,   # noqa: E402
                                    plot_alpha_heatmap)
from experiments.run_mosei import diag_hyper                                # noqa: E402
from models.stage4_fusion import (MaskGateFusion, GlobalQueryFusion,         # noqa: E402
                                  HyperGuidedFusion)

PROTOCOL = "B"                       # 第一轮仅 Protocol B (成功再议 PA)
LAM_CONS = 0.005                     # 封版工作点 (B3 邻域实验选定)
SG = True                            # normalized stop-gradient
CONS_NORM = "l2"
COND_NAMES = [c[0] for c in FIXED_TEST_CONDITIONS]
MISS = [c for c in COND_NAMES if c != "None"]
KEY_CONDS = ["T missing", "T+A missing", "T+V missing", "A+V missing"]   # 用户指定 F3 vs F1 焦点
VARIANTS = ["F0", "F1", "F2", "F3"]
VLABEL = {"F0": "F0 H0(z→Pred)", "F1": "F1 MaskGate", "F2": "F2 GlobalQuery", "F3": "F3 HyperGuided"}
HAS_ALPHA = {"F0": False, "F1": True, "F2": True, "F3": True}
MOD = ["T", "A", "V"]
DATASET_TAG = "MOSEI-Stage4-Fusion"


def make_factory(key, cfg):
    """返回 model_factory (无参 callable -> 模型实例) 供 train_stage3c_consistency 在种子重置后调用。
    F0 -> None (默认构建原生 HyperRepresentationModel = 封版 B3); F1/F2/F3 -> 对应融合类。
    backbone (proj_T/A/V + hyper_encoder + predictor) 在 super().__init__ 中先建 -> 与 F0 init 逐字节一致。"""
    if key == "F0":
        return None
    common = dict(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim, vision_dim=cfg.vision_dim,
                  hyper_dim=cfg.hyper_dim, hidden_dim=cfg.hyper_hidden_dim, dropout=cfg.dropout)
    if key == "F1":
        return lambda: MaskGateFusion(**common)
    if key == "F2":
        return lambda: GlobalQueryFusion(**common)
    if key == "F3":
        return lambda: HyperGuidedFusion(**common)
    raise ValueError(f"未知 variant: {key}")


def _avg(seq):
    return float(np.mean(seq))


def print_table(title, df):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print(df.to_string(index=False))


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.mosei_dir, exist_ok=True)

    datasets = build_mosei_datasets(cfg.mosei_data_path, cfg.mosei_conv_dir,
                                    inf_policy=cfg.mosei_inf_policy)
    Dt, Da, Dv, L = mosei_dims(datasets)
    cfg.text_dim, cfg.audio_dim, cfg.vision_dim, cfg.seq_len = Dt, Da, Dv, L

    print("=" * 104)
    print("Stage 4 (MOSEI): Hyper-Guided Dynamic Fusion · 最小实现 · F0/F1/F2/F3 · Protocol B · 30ep")
    print(f"  融合: h_fuse = z_hyper + Σ α_i V_i,  V_i=W_V x_i,  s_i=Q·K_i/√D,  缺失位=-inf,  α=softmax(s)")
    print(f"  F3(本项): Q=W_Q z_hyper | F2: Q=全局可学习 q | F1: α=MLP([x_T,x_A,x_V,m]) | F0: 无融合(=封版B3)")
    print(f"  封版 consistency 保留: λ={LAM_CONS}, sg={SG}, norm={CONS_NORM} (不改 HyperEncoder / loss)")
    print(f"Device={device} | seed={cfg.seed} epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} | hyper D={cfg.hyper_dim}")
    print(f"dims: text={Dt} audio={Da} vision={Dv} L={L} | split train={len(datasets['train'])} "
          f"valid={len(datasets['valid'])} test={len(datasets['test'])}")
    print("=" * 104)

    rec = ExperimentRecorder(cfg.mosei_dir)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    n_test = len(datasets["test"])
    none_mask = generate_missing_mask(n_test, missing_pattern="T+A+V")
    # 预生成 7 条件的整split mask (α 提取与 MAE/Corr 评估共用同一 mask)
    cond_mask = {name: generate_missing_mask(n_test, missing_pattern=pat)
                 for name, pat in FIXED_TEST_CONDITIONS}

    models, infos, fixed, diags, zstds = {}, {}, {}, {}, {}
    alpha_by = {}        # key -> {cond: (N,3)}
    alpha_mean = {}      # key -> (7,3)
    alpha_std = {}       # key -> (7,3)
    audit_rows = []

    for key in VARIANTS:
        factory = make_factory(key, cfg)
        print(f"\n>>> [Stage4] 训练 {VLABEL[key]} + consistency(λ={LAM_CONS},sg,norm={CONS_NORM}) + Protocol {PROTOCOL} ...", flush=True)
        model, info = train_stage3c_consistency(PROTOCOL, cfg, datasets, device,
                                                lam_cons=LAM_CONS, hyper_dim=cfg.hyper_dim,
                                                stop_grad_full=SG, cons_norm=CONS_NORM,
                                                model_factory=factory, verbose=True)
        models[key] = model
        infos[key] = info
        print(f"    best_epoch={info['best_epoch']} best_valid_MAE={info['best_valid_mae']:.4f} "
              f"| #params={info['n_params']:,}", flush=True)
        torch.save({"model_state_dict": model.state_dict(), "variant": key,
                    "model_type": "hyper_stage4_fusion", "protocol": PROTOCOL,
                    "lam_cons": LAM_CONS, "stop_grad_full": SG, "cons_norm": CONS_NORM,
                    "best_epoch": info["best_epoch"], "best_valid_mae": info["best_valid_mae"],
                    "n_params": info["n_params"]},
                   os.path.join(cfg.mosei_dir, f"stage4_{key}_protocolB.pt"))

        # ---- 7 固定条件评估 (MAE/Corr) + 记录 ----
        for cond_name, _ in FIXED_TEST_CONDITIONS:
            _, m = evaluate_with_mask_s3(model, "hyper", test_loader, cond_mask[cond_name], device)
            fixed[(key, cond_name)] = m
            rec.log(dataset=DATASET_TAG, protocol=PROTOCOL, model=VLABEL[key], seed=cfg.seed,
                    missing_pattern=cond_name, missing_probability=cfg.protocol_b_train_missing_prob,
                    best_epoch=info["best_epoch"], n_params=info["n_params"],
                    MAE=m["MAE"], Corr=m["Corr"])

        # ---- z_hyper 诊断 (连续性: 融合头是否改变 z_hyper 的 rank/drift/transfer) ----
        zA = extract_representation_s3(model, "hyper", test_loader, none_mask, device)
        Zs = [extract_representation_s3(model, "hyper", test_loader, cond_mask[c], device)["rep"]
              for c in COND_NAMES]
        Z_by_pat = np.stack(Zs, axis=0)
        diags[key] = diag_hyper(zA["rep"], Z_by_pat, zA["label"], COND_NAMES)
        zstds[key] = float(np.std(zA["rep"]))
        d = diags[key]
        print(f"    [diag {key}] probe_corr={d['probe_corr']:+.4f} eff_rank={d['eff_rank']:.3f} "
              f"z_std={zstds[key]:.4f} drift_T={d['per_pattern']['T missing']['drift_ratio']:.4f} "
              f"transfer_T={d['per_pattern']['T missing']['transfer']:+.4f}", flush=True)

        # ---- per-sample α 提取 + 缺失归零审计 (F1/F2/F3) ----
        if HAS_ALPHA[key]:
            alpha_by[key] = {}
            mean_mat, std_mat = [], []
            for cond_name, _ in FIXED_TEST_CONDITIONS:
                out = extract_alpha_s4(model, test_loader, cond_mask[cond_name], device)
                a = out["alpha"]                                   # (N,3)
                alpha_by[key][cond_name] = a
                mean_mat.append(a.mean(axis=0)); std_mat.append(a.std(axis=0))
                mask_row = np.asarray(cond_mask[cond_name][0])      # 该条件可用性 (1/0)
                aud = alpha_missing_audit(a, mask_row)
                audit_rows.append({"variant": key, "cond": cond_name,
                                   "missing_dims": "".join(MOD[i] for i in aud["missing_dims"]) or "-",
                                   "max_missing_α": round(aud["max_missing_alpha"], 8),
                                   "avail_α_sum": round(aud.get("avail_alpha_sum_mean", float("nan")), 4),
                                   "mean_α_T": round(float(a[:, 0].mean()), 4),
                                   "mean_α_A": round(float(a[:, 1].mean()), 4),
                                   "mean_α_V": round(float(a[:, 2].mean()), 4)})
            alpha_mean[key] = np.array(mean_mat)                    # (7,3)
            alpha_std[key] = np.array(std_mat)                      # (7,3)
            print(f"    [α {key}] None: α_T/A/V={alpha_mean[key][0].round(3)} | "
                  f"缺失归零 max_α_missing="
                  f"{max(r['max_missing_α'] for r in audit_rows if r['variant']==key):.2e}", flush=True)

    # ---- 表1/2: 7 条件 MAE / Corr (F0-F3) ----
    def fixed_table(metric):
        rows = []
        for key in VARIANTS:
            row = {"variant": VLABEL[key]}
            for cond in COND_NAMES:
                row[str(cond)] = round(fixed[(key, cond)][metric], 4)
            row["missAvg"] = round(_avg([fixed[(key, c)][metric] for c in MISS]), 4)
            rows.append(row)
        return pd.DataFrame(rows)
    print_table("[Stage4 MAE] Protocol B × 7 固定条件  (MAE ↓, 含缺失均值 missAvg)", fixed_table("MAE"))
    print_table("[Stage4 Corr] Protocol B × 7 固定条件  (Corr ↑, 含缺失均值 missAvg)", fixed_table("Corr"))

    # ---- 表3: 参数 / 训练 / z_hyper 诊断汇总 ----
    srows = []
    for key in VARIANTS:
        d = diags[key]
        pt = d["per_pattern"]["T missing"]
        srows.append({"variant": VLABEL[key], "#params": f"{infos[key]['n_params']:,}",
                      "best_ep": infos[key]["best_epoch"],
                      "validMAE": round(infos[key]["best_valid_mae"], 4),
                      "probe_corr": round(d["probe_corr"], 4), "eff_rank": round(d["eff_rank"], 3),
                      "z_std": round(zstds[key], 4), "drift_T": round(pt["drift_ratio"], 4),
                      "transfer_T": round(pt["transfer"], 4),
                      "missMAE": round(_avg([fixed[(key, c)]["MAE"] for c in MISS]), 4),
                      "missCorr": round(_avg([fixed[(key, c)]["Corr"] for c in MISS]), 4)})
    df_sum = pd.DataFrame(srows)
    print_table("[Stage4 汇总] 参数 / 选模 / z_hyper 诊断 / 缺失均值 (全部 Protocol B)", df_sum)

    # ---- 表4: F3 vs F1 vs F0 焦点困难条件 (用户核心问题) ----
    focus_rows = []
    for cond in KEY_CONDS:
        r = {"cond": cond}
        for metric, better in [("MAE", min), ("Corr", max)]:
            vals = {k: fixed[(k, cond)][metric] for k in VARIANTS}
            bestk = better(vals, key=lambda k: vals[k])
            for k in VARIANTS:
                r[f"{k}_{metric}"] = round(vals[k], 4)
            r[f"best_{metric}"] = bestk
        focus_rows.append(r)
    df_focus = pd.DataFrame(focus_rows)
    print_table("[Stage4 焦点] F3 vs F1 vs F2 vs F0 · 困难条件 (T-missing / T+A / T+V / A+V)", df_focus)

    # ---- 表5: α 缺失归零审计 + 分布 ----
    if audit_rows:
        df_audit = pd.DataFrame(audit_rows)
        print_table("[Stage4 α 审计] 缺失模态权重严格归零? (max_missing_α 应 ~0) + 各条件平均 α", df_audit)
        worst = max(r["max_missing_α"] for r in audit_rows)
        print(f"\n  >>> α 归零硬检查: 所有 F1/F2/F3 × 所有缺失位 max_missing_α = {worst:.3e} "
              f"({'✓ 严格归零' if worst < 1e-4 else '✗ 未归零'}) <<<")

    # ---- 判定: F3 > F1 > F0 ? ----
    def missavg(key, metric):
        return _avg([fixed[(key, c)][metric] for c in MISS])
    print("\n" + "=" * 104)
    print("[Stage4 判定] 理想 F3 > F1 > F0 (MAE↓ / Corr↑, 缺失均值)")
    print("=" * 104)
    for metric, better, sign in [("MAE", "lower", "↓"), ("Corr", "higher", "↑")]:
        v = {k: missavg(k, metric) for k in VARIANTS}
        f3f1 = (v["F3"] < v["F1"]) if better == "lower" else (v["F3"] > v["F1"])
        f1f0 = (v["F1"] < v["F0"]) if better == "lower" else (v["F1"] > v["F0"])
        f3f0 = (v["F3"] < v["F0"]) if better == "lower" else (v["F3"] > v["F0"])
        print(f"  {metric}{sign}: F0={v['F0']:.4f} F1={v['F1']:.4f} F2={v['F2']:.4f} F3={v['F3']:.4f} | "
              f"F3>F1:{'✓' if f3f1 else '✗'} F1>F0:{'✓' if f1f0 else '✗'} F3>F0:{'✓' if f3f0 else '✗'}")

    # ---- α 热力图可视化 ----
    viz_keys = [k for k in VARIANTS if HAS_ALPHA[k]]
    if viz_keys:
        png = os.path.join(cfg.mosei_dir, "stage4_alpha_heatmap.png")
        ok = plot_alpha_heatmap({k: alpha_mean[k] for k in viz_keys},
                                {k: alpha_std[k] for k in viz_keys}, COND_NAMES, png)
        print(f"\n  α 热力图: {'已保存 ' + png if ok else '跳过 (matplotlib 不可用)'}")

    # ---- 落盘 ----
    fixed_table("MAE").to_csv(os.path.join(cfg.mosei_dir, "stage4_fixed_mae.csv"), index=False)
    fixed_table("Corr").to_csv(os.path.join(cfg.mosei_dir, "stage4_fixed_corr.csv"), index=False)
    df_sum.to_csv(os.path.join(cfg.mosei_dir, "stage4_summary.csv"), index=False)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(os.path.join(cfg.mosei_dir, "stage4_alpha_audit.csv"), index=False)
    for key in viz_keys:                                   # per-sample α (7,N,3) 全量保存
        np.save(os.path.join(cfg.mosei_dir, f"stage4_alpha_{key}.npy"),
                np.stack([alpha_by[key][c] for c in COND_NAMES], axis=0))
    dump = {
        "stage": "4", "variant": "hyper_guided_dynamic_fusion", "dataset": "MOSEI",
        "protocol": PROTOCOL, "lam_cons": LAM_CONS, "stop_grad_full": SG, "cons_norm": CONS_NORM,
        "fusion": "h_fuse = z_hyper + sum_i alpha_i V_i; V_i=W_V x_i; s_i=Q.K_i/sqrt(D); missing->-inf; alpha=softmax(s)",
        "variants": {k: {"label": VLABEL[k], "has_alpha": HAS_ALPHA[k]} for k in VARIANTS},
        "config": {"seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size,
                   "dims": [Dt, Da, Dv, L], "hyper_dim": cfg.hyper_dim,
                   "protocol_b_train_missing_prob": cfg.protocol_b_train_missing_prob},
        "fixed": {f"{k}|{c}": fixed[(k, c)] for k in VARIANTS for c in COND_NAMES},
        "missavg": {k: {"MAE": missavg(k, "MAE"), "Corr": missavg(k, "Corr")} for k in VARIANTS},
        "summary": df_sum.to_dict(orient="records"),
        "focus": df_focus.to_dict(orient="records"),
        "diag": {k: diags[k] for k in VARIANTS},
        "z_std": {k: zstds[k] for k in VARIANTS},
        "alpha_mean": {k: alpha_mean[k].tolist() for k in viz_keys},
        "alpha_std": {k: alpha_std[k].tolist() for k in viz_keys},
        "alpha_audit": audit_rows,
        "train_info": {k: {"best_epoch": infos[k]["best_epoch"], "best_valid_mae": infos[k]["best_valid_mae"],
                           "n_params": infos[k]["n_params"], "history": infos[k]["history"]} for k in VARIANTS},
    }
    with open(os.path.join(cfg.mosei_dir, "stage4_results.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 104)
    print(f"Stage4 结果 : {os.path.join(cfg.mosei_dir, 'stage4_results.json')}")
    print(f"MAE/Corr 表 : {os.path.join(cfg.mosei_dir, 'stage4_fixed_mae.csv')} / stage4_fixed_corr.csv")
    print(f"α 审计/全量 : {os.path.join(cfg.mosei_dir, 'stage4_alpha_audit.csv')} / stage4_alpha_F*.npy")
    print(f"统一实验记录: {rec.csv} / {rec.jsonl}")
    print("=" * 104)


if __name__ == "__main__":
    main()
