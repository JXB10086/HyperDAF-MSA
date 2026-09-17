"""Document LNLN missing protocol from code (no model change)."""
import inspect
import json
from pathlib import Path

root = Path(r"F:/winoptimizeDir/Desktop/HyperDAF-MSA/third_party/LNLN")
import sys
sys.path.insert(0, str(root))
from core import dataset as ds

src = inspect.getsource(ds.MMDataset.generate_m)
out = {
    "missing_implementation": "token/frame-level erase via generate_m",
    "NOT_CMRP_whole_modality_zeroing": True,
    "train_behavior": "per-modality missing_rate ~ Uniform(0,1); half samples forced rate=0; then erase tokens/frames",
    "eval_behavior": "missing_rate_eval_test applied to L/A/V simultaneously at token/frame level",
    "text_erase": "replaced with UNK token id 100; CLS/SEP preserved",
    "av_erase": "frame mask * modality (zero erased frames)",
    "robust_evaluation_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "generate_m_source_excerpt": src[:1200],
}
dest = Path(r"F:/winoptimizeDir/Desktop/HyperDAF-MSA/experiments/fair_baseline/lnln_smoke/missing_protocol_note.json")
dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote", dest)
