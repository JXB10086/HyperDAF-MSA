"""
Fair Baseline Step3: LNLN native MOSI smoke (1-3 epochs).
Does NOT touch CMRP-MSA. Does NOT convert MultiBench features.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"F:/winoptimizeDir/Desktop/HyperDAF-MSA")
LNLN = ROOT / "third_party" / "LNLN"
SMOKE_DIR = ROOT / "experiments" / "fair_baseline" / "lnln_smoke"
CFG_SRC = ROOT / "experiments" / "fair_baseline" / "lnln_smoke_mosi.yaml"
CFG_DST = LNLN / "configs" / "train_mosi_smoke.yaml"
LOG = SMOKE_DIR / "train_smoke.log"
REPORT = SMOKE_DIR / "STEP3_SMOKE_REPORT.json"

SMOKE_DIR.mkdir(parents=True, exist_ok=True)

# Copy smoke yaml into LNLN configs (no edit of LNLN train.py)
shutil.copy2(CFG_SRC, CFG_DST)

env = os.environ.copy()
# Official MMSA is local; BERT is cached — do not hit the network during smoke.
env["HF_HUB_OFFLINE"] = "1"
env["TRANSFORMERS_OFFLINE"] = "1"
env["HF_HUB_DISABLE_TELEMETRY"] = "1"

cmd = [
    sys.executable,
    "-u",
    "train.py",
    "--config_file",
    "configs/train_mosi_smoke.yaml",
    "--seed",
    "1111",
]

t0 = time.time()
print("CWD:", LNLN)
print("CMD:", " ".join(cmd))
print("LOG:", LOG)

with open(LOG, "w", encoding="utf-8", errors="replace") as lf:
    lf.write(f"cmd={' '.join(cmd)}\n")
    lf.write(f"cwd={LNLN}\n\n")
    lf.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(LNLN),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lf.write(line)
        lf.flush()
    rc = proc.wait()

elapsed = time.time() - t0

# Collect checkpoints
ckpt_dir = LNLN / "ckpt" / "mosi"
ckpts = []
if ckpt_dir.exists():
    ckpts = sorted([p.name for p in ckpt_dir.glob("*.pth")])

# Parse key lines from log
loss_lines = []
result_lines = []
best_lines = []
errors = []
with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        s = line.strip()
        if "Train Loss Epoch" in s:
            loss_lines.append(s)
        if "Train Results Epoch" in s:
            result_lines.append(s)
        if "Current Best" in s:
            best_lines.append(s)
        if any(x in s.lower() for x in ["error", "traceback", "exception", "cuda"]):
            if "device" not in s.lower() or "error" in s.lower() or "traceback" in s.lower():
                errors.append(s)

report = {
    "step": "Fair Baseline Step3 · LNLN native smoke",
    "status": "PASS" if rc == 0 else "FAIL",
    "returncode": rc,
    "elapsed_sec": round(elapsed, 1),
    "environment": {
        "python": sys.version,
        "cwd": str(LNLN),
        "config": str(CFG_DST),
        "data_note": "official MMSA unaligned_50.pkl (SHA 78e0f8b5... matched); BERT offline cache",
        "hf_offline": True,
    },
    "train_loss_lines": loss_lines,
    "train_result_lines": result_lines,
    "best_lines": best_lines[-6:],
    "checkpoints": ckpts,
    "error_like_lines": errors[-30:],
    "log_path": str(LOG),
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n=== SMOKE REPORT ===")
print(json.dumps(report, indent=2, ensure_ascii=False))
sys.exit(rc)
