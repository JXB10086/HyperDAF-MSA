"""Build CMRP evidence tables from existing artifacts. No training."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CONDS = [
    "None", "T missing", "A missing", "V missing",
    "T+A missing", "T+V missing", "A+V missing",
]
MISS = [c for c in CONDS if c != "None"]


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    with open(ROOT / "mosei" / "b3sg_results.json", encoding="utf-8") as f:
        b3 = json.load(f)
    with open(ROOT / "stage3" / "stage3_results.json", encoding="utf-8") as f:
        s3 = json.load(f)
    with open(ROOT / "mosi_final" / "mosi_final_results.json", encoding="utf-8") as f:
        mf = json.load(f)
    with open(ROOT / "stage3" / "diag_zhyper_results.json", encoding="utf-8") as f:
        dg = json.load(f)
    with open(ROOT / "mosei" / "multiseed" / "mosei_multiseed_results.json", encoding="utf-8") as f:
        msj = json.load(f)

    rows_fixed = []
    # MOSEI H0 vs H0+B3 (paired, same b3sg run)
    for lam, model in [("0.0", "H0 (lam=0)"), ("0.005", "H0+B3 (lam=0.005)")]:
        full_m = b3["fixed"][f"{lam}|None"]["MAE"]
        full_c = b3["fixed"][f"{lam}|None"]["Corr"]
        for c in CONDS:
            m = b3["fixed"][f"{lam}|{c}"]["MAE"]
            co = b3["fixed"][f"{lam}|{c}"]["Corr"]
            rows_fixed.append({
                "dataset": "MOSEI", "model": model, "condition": c,
                "MAE": round(m, 4), "Corr": round(co, 4),
                "dMAE": "" if c == "None" else round(m - full_m, 4),
                "dCorr": "" if c == "None" else round(full_c - co, 4),
                "seed": 42, "protocol": "B", "status": "unified",
                "source": "experiments/mosei/b3sg_results.json",
            })
        ms = [b3["fixed"][f"{lam}|{c}"]["MAE"] for c in MISS]
        cs = [b3["fixed"][f"{lam}|{c}"]["Corr"] for c in MISS]
        mm, mc = sum(ms) / 6, sum(cs) / 6
        rows_fixed.append({
            "dataset": "MOSEI", "model": model, "condition": "missAvg(6)",
            "MAE": round(mm, 4), "Corr": round(mc, 4),
            "dMAE": round(mm - full_m, 4), "dCorr": round(full_c - mc, 4),
            "seed": 42, "protocol": "B", "status": "unified",
            "source": "experiments/mosei/b3sg_results.json",
        })

    # MOSI H0 (stage3) and H0+B3 (mosi_final) — same seed/protocol, different runners
    full_m = s3["fixed"]["hyper_D256|B|None"]["MAE"]
    full_c = s3["fixed"]["hyper_D256|B|None"]["Corr"]
    for c in CONDS:
        m = s3["fixed"][f"hyper_D256|B|{c}"]["MAE"]
        co = s3["fixed"][f"hyper_D256|B|{c}"]["Corr"]
        rows_fixed.append({
            "dataset": "MOSI", "model": "H0 (no B3)", "condition": c,
            "MAE": round(m, 4), "Corr": round(co, 4),
            "dMAE": "" if c == "None" else round(m - full_m, 4),
            "dCorr": "" if c == "None" else round(full_c - co, 4),
            "seed": 42, "protocol": "B", "status": "unified",
            "source": "experiments/stage3/stage3_results.json",
            "note": "H0 from stage3 hyper_D256; not same run as mosi_final H0+B3",
        })
    ms = [s3["fixed"][f"hyper_D256|B|{c}"]["MAE"] for c in MISS]
    cs = [s3["fixed"][f"hyper_D256|B|{c}"]["Corr"] for c in MISS]
    mm, mc = sum(ms) / 6, sum(cs) / 6
    rows_fixed.append({
        "dataset": "MOSI", "model": "H0 (no B3)", "condition": "missAvg(6)",
        "MAE": round(mm, 4), "Corr": round(mc, 4),
        "dMAE": round(mm - full_m, 4), "dCorr": round(full_c - mc, 4),
        "seed": 42, "protocol": "B", "status": "unified",
        "source": "experiments/stage3/stage3_results.json",
        "note": "H0 from stage3 hyper_D256; not same run as mosi_final H0+B3",
    })

    hb = mf["results"]["B"]["HB3"]
    full_m = hb["fixed"]["None"]["MAE"]
    full_c = hb["fixed"]["None"]["Corr"]
    for c in CONDS:
        m = hb["fixed"][c]["MAE"]
        co = hb["fixed"][c]["Corr"]
        rows_fixed.append({
            "dataset": "MOSI", "model": "H0+B3 (lam=0.005)", "condition": c,
            "MAE": round(m, 4), "Corr": round(co, 4),
            "dMAE": "" if c == "None" else round(m - full_m, 4),
            "dCorr": "" if c == "None" else round(full_c - co, 4),
            "seed": 42, "protocol": "B", "status": "unified",
            "source": "experiments/mosi_final/mosi_final_results.json",
        })
    ms = [hb["fixed"][c]["MAE"] for c in MISS]
    cs = [hb["fixed"][c]["Corr"] for c in MISS]
    mm, mc = sum(ms) / 6, sum(cs) / 6
    rows_fixed.append({
        "dataset": "MOSI", "model": "H0+B3 (lam=0.005)", "condition": "missAvg(6)",
        "MAE": round(mm, 4), "Corr": round(mc, 4),
        "dMAE": round(mm - full_m, 4), "dCorr": round(full_c - mc, 4),
        "seed": 42, "protocol": "B", "status": "unified",
        "source": "experiments/mosi_final/mosi_final_results.json",
    })

    # Random
    rows_rand = []
    full_m = s3["fixed"]["hyper_D256|B|None"]["MAE"]
    full_c = s3["fixed"]["hyper_D256|B|None"]["Corr"]
    for p in ["0.1", "0.3", "0.5", "0.7", "0.9"]:
        m = s3["random"][f"hyper_D256|B|{p}"]["MAE"]
        co = s3["random"][f"hyper_D256|B|{p}"]["Corr"]
        rows_rand.append({
            "dataset": "MOSI", "model": "H0 (no B3)", "p": float(p),
            "MAE": round(m, 4), "Corr": round(co, 4),
            "dMAE": round(m - full_m, 4), "dCorr": round(full_c - co, 4),
            "seed": 42, "protocol": "B", "status": "unified",
            "source": "experiments/stage3/stage3_results.json",
        })
    full_m = hb["fixed"]["None"]["MAE"]
    full_c = hb["fixed"]["None"]["Corr"]
    for p in ["0.1", "0.3", "0.5", "0.7", "0.9"]:
        m = hb["rand"][p]["MAE"]
        co = hb["rand"][p]["Corr"]
        rows_rand.append({
            "dataset": "MOSI", "model": "H0+B3 (lam=0.005)", "p": float(p),
            "MAE": round(m, 4), "Corr": round(co, 4),
            "dMAE": round(m - full_m, 4), "dCorr": round(full_c - co, 4),
            "seed": 42, "protocol": "B", "status": "unified",
            "source": "experiments/mosi_final/mosi_final_results.json",
        })
    full_m = b3["fixed"]["0.005|None"]["MAE"]
    full_c = b3["fixed"]["0.005|None"]["Corr"]
    rm = sum(msj["store"]["HB3"]["randMAE"]) / 3
    rc = sum(msj["store"]["HB3"]["randCorr"]) / 3
    rows_rand.append({
        "dataset": "MOSEI", "model": "H0+B3 (lam=0.005)", "p": "randMean(0.1-0.9)",
        "MAE": round(rm, 4), "Corr": round(rc, 4),
        "dMAE": round(rm - full_m, 4), "dCorr": round(full_c - rc, 4),
        "seed": "42/43/44", "protocol": "B", "status": "unified",
        "source": "experiments/mosei/multiseed/mosei_multiseed_results.json",
        "note": "per-p curve not stored for MOSEI H0+B3; aggregate only",
    })

    # D_cross
    rows_d = []
    for lam, model in [("0.0", "H0 (lam=0)"), ("0.005", "H0+B3 (lam=0.005)")]:
        pp = b3["diag"][lam]["per_pattern"]
        med = b3["diag"][lam]["median_pair_dist"]
        for c in MISS:
            rows_d.append({
                "dataset": "MOSEI", "model": model, "condition": c,
                "D_cross": round(pp[c]["drift_ratio"], 4),
                "raw_drift_L2": round(pp[c]["drift"], 4),
                "median_pair_dist_full": round(med, 4),
                "transfer": round(pp[c]["transfer"], 4),
                "seed": 42, "protocol": "B", "status": "unified",
                "source": "experiments/mosei/b3sg_results.json",
                "definition": "D_cross=||z_full-z_miss||_2 / median_pairwise(z_full)",
            })
        vals = [pp[c]["drift_ratio"] for c in MISS]
        rows_d.append({
            "dataset": "MOSEI", "model": model, "condition": "mean(6)",
            "D_cross": round(sum(vals) / 6, 4),
            "raw_drift_L2": "", "median_pair_dist_full": round(med, 4), "transfer": "",
            "seed": 42, "protocol": "B", "status": "unified",
            "source": "experiments/mosei/b3sg_results.json",
            "definition": "D_cross=||z_full-z_miss||_2 / median_pairwise(z_full)",
        })

    for item in dg["missing_B"]["drift"]:
        if item["pattern"] == "None":
            continue
        rows_d.append({
            "dataset": "MOSI", "model": "H0 (no B3)", "condition": item["pattern"],
            "D_cross": round(item["drift_over_pair"], 4),
            "raw_drift_L2": round(item["drift"], 4),
            "median_pair_dist_full": round(dg["missing_B"]["median_pair_dist"], 4),
            "transfer": "",
            "seed": 42, "protocol": "B", "status": "unified",
            "source": "experiments/stage3/diag_zhyper_results.json",
            "definition": "D_cross=drift/median_pairwise(z_full)",
            "note": "H0+B3 D_cross on MOSI absent in existing artifacts",
        })

    # Joint MOSEI summary
    summary = []
    for c in MISS + ["missAvg(6)"]:
        h0 = next(r for r in rows_fixed if r["dataset"] == "MOSEI"
                  and r["model"] == "H0 (lam=0)" and r["condition"] == c)
        b3r = next(r for r in rows_fixed if r["dataset"] == "MOSEI"
                   and r["model"] == "H0+B3 (lam=0.005)" and r["condition"] == c)
        dc = "mean(6)" if c == "missAvg(6)" else c
        d0 = next(r for r in rows_d if r["dataset"] == "MOSEI"
                  and r["model"] == "H0 (lam=0)" and r["condition"] == dc)
        d3 = next(r for r in rows_d if r["dataset"] == "MOSEI"
                  and r["model"] == "H0+B3 (lam=0.005)" and r["condition"] == dc)
        summary.append({
            "dataset": "MOSEI", "condition": c,
            "H0_dMAE": h0["dMAE"], "B3_dMAE": b3r["dMAE"],
            "dMAE_B3_minus_H0": round(float(b3r["dMAE"]) - float(h0["dMAE"]), 4),
            "H0_dCorr": h0["dCorr"], "B3_dCorr": b3r["dCorr"],
            "dCorr_B3_minus_H0": round(float(b3r["dCorr"]) - float(h0["dCorr"]), 4),
            "H0_Dcross": d0["D_cross"], "B3_Dcross": d3["D_cross"],
            "Dcross_B3_minus_H0": round(float(d3["D_cross"]) - float(d0["D_cross"]), 4),
            "seed": 42, "protocol": "B", "status": "unified",
            "note": "negative dMAE/Dcross change => B3 better (less degradation / less drift)",
        })

    write_csv(OUT / "table1_fixed_degradation.csv", rows_fixed)
    write_csv(OUT / "table2_random_degradation.csv", rows_rand)
    write_csv(OUT / "table3_dcross_h0_vs_b3.csv", rows_d)
    write_csv(OUT / "table4_mosei_h0_vs_b3_joint.csv", summary)

    conclusions = {
        "status": "evidence_tables_only",
        "core_model": "CMRP-MSA = H0+B3 (lam_cons=0.005, nsg-L2)",
        "no_new_training": True,
        "mosei_paired": True,
        "mosi_dcross_paired": False,
        "key_findings": [
            "MOSEI missAvg dMAE: H0 0.0758 -> H0+B3 0.0746 (slightly smaller degradation).",
            "MOSEI missAvg dCorr: H0 0.1838 -> H0+B3 0.1837 (essentially unchanged).",
            "MOSEI D_cross mean(6): H0 0.6978 -> H0+B3 0.6785 (slightly down).",
            "Largest D_cross drop on MOSEI: T-missing 0.8996->0.7879; T+V 0.9948->0.8937.",
            "Not uniform: A-missing / T+A / A+V D_cross slightly increase under B3.",
            "T+V MAE improves with B3 (0.8296->0.8144) alongside D_cross drop — partial support for claim.",
            "MOSI missAvg dMAE: H0 0.1578 -> H0+B3 0.1312 (lower degradation), but runs are not paired for D_cross.",
            "Do not claim outperforms HME/CMAD/EASE; this is internal stability evidence only.",
        ],
    }
    with open(OUT / "conclusions.json", "w", encoding="utf-8") as f:
        json.dump(conclusions, f, indent=2, ensure_ascii=False)
    print("Wrote tables to", OUT)


if __name__ == "__main__":
    main()
