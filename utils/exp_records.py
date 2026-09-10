"""
实验记录器  (自 MOSEI 阶段起启用)
--------------------------------------------------
每一次评估都完整落盘一行, 字段固定, 供论文 Mean±Std 与可复现性:
  dataset / protocol / model / seed / missing_pattern / missing_probability
  / best_epoch / n_params / MAE / Corr / timestamp
同时写 .jsonl (追加, 机器读) 与 .csv (追加, 人读)。
"""
import os
import csv
import json
import time

FIELDS = ["dataset", "protocol", "model", "seed", "missing_pattern",
          "missing_probability", "best_epoch", "n_params", "MAE", "Corr", "timestamp"]


class ExperimentRecorder:
    def __init__(self, out_dir, name="experiment_records"):
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl = os.path.join(out_dir, name + ".jsonl")
        self.csv = os.path.join(out_dir, name + ".csv")
        if not os.path.exists(self.csv):
            with open(self.csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log(self, **kw):
        rec = {k: kw.get(k) for k in FIELDS}
        rec["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        with open(self.csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(rec)
        return rec
