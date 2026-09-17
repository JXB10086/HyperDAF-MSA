import hashlib
import json
import os

p = r"F:\winoptimizeDir\Desktop\HyperDAF-MSA\data\mmsa\MOSI\Processed\unaligned_50.pkl"
out = r"F:\winoptimizeDir\Desktop\HyperDAF-MSA\experiments\fair_baseline\lnln_smoke\sha256.json"
h = hashlib.sha256()
with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
digest = h.hexdigest()
official = "78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524"
report = {
    "path": p,
    "bytes": os.path.getsize(p),
    "sha256": digest,
    "official_mmsa_sha256": official,
    "matches_official": digest == official,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
