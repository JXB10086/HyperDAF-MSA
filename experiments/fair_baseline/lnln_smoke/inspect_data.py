"""Inspect MMSA MOSI unaligned_50.pkl for LNLN Step3 smoke."""
import hashlib
import os
import pickle
import json

p = r"F:/winoptimizeDir/Desktop/HyperDAF-MSA/data/mmsa/MOSI/Processed/unaligned_50.pkl"
out = {"path": p, "exists": os.path.exists(p)}
if out["exists"]:
    out["size_mb"] = round(os.path.getsize(p) / 1e6, 2)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    out["sha256"] = h.hexdigest()
    with open(p, "rb") as f:
        d = pickle.load(f)
    out["splits"] = list(d.keys())
    out["per_split"] = {}
    for sp in ["train", "valid", "test"]:
        s = d[sp]
        info = {"n": len(s["id"])}
        for k in ["text_bert", "vision", "audio", "regression_labels", "audio_lengths", "vision_lengths", "raw_text"]:
            if k not in s:
                info[k] = "MISSING"
                continue
            v = s[k]
            if hasattr(v, "shape"):
                info[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                info[k] = {"type": type(v).__name__, "len": len(v) if hasattr(v, "__len__") else None}
        out["per_split"][sp] = info

dest = r"F:/winoptimizeDir/Desktop/HyperDAF-MSA/experiments/fair_baseline/lnln_smoke/data_inspect.json"
os.makedirs(os.path.dirname(dest), exist_ok=True)
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
