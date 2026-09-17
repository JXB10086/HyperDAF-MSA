"""
Diagnostic-only: build a tiny VALID MOSI-like pkl so LNLN code path can smoke
without using MultiBench / without modifying LNLN model.
NOT official MMSA; NOT for paper numbers.
"""
import pickle
from pathlib import Path
import numpy as np
from transformers import BertTokenizer

out = Path(r"F:/winoptimizeDir/Desktop/HyperDAF-MSA/experiments/fair_baseline/lnln_smoke/synthetic_valid_mosi_tiny.pkl")
tok = BertTokenizer.from_pretrained("bert-base-uncased")

texts = [
    "i am happy today",
    "this movie is terrible",
    "the product is okay",
    "absolutely wonderful experience",
    "worst service ever",
    "neutral comment about weather",
    "love the soundtrack",
    "hate the ending",
] * 4  # 32
labels = np.array([1.2, -1.5, 0.1, 2.0, -2.2, 0.0, 1.5, -1.0] * 4, dtype=np.float32)

def encode(batch_texts, max_len=50):
    n = len(batch_texts)
    ids = np.zeros((n, 50), dtype=np.float32)
    mask = np.zeros((n, 50), dtype=np.float32)
    seg = np.zeros((n, 50), dtype=np.float32)
    for i, t in enumerate(batch_texts):
        enc = tok(t, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
        ids[i] = enc["input_ids"][0]
        mask[i] = enc["attention_mask"][0]
    return np.stack([ids, mask, seg], axis=1)

def make_split(n, seed, texts_slice, labs):
    rng = np.random.RandomState(seed)
    vision = rng.randn(n, 375, 20).astype(np.float32) * 0.1
    audio = rng.randn(n, 500, 5).astype(np.float32) * 0.1
    vlen = np.full(n, 40, dtype=np.int64)
    alen = np.full(n, 60, dtype=np.int64)
    return {
        "text_bert": encode(texts_slice),
        "vision": vision,
        "audio": audio,
        "raw_text": list(texts_slice),
        "id": [f"syn_{seed}_{i}" for i in range(n)],
        "regression_labels": labs.reshape(n, 1, 1),
        "audio_lengths": alen,
        "vision_lengths": vlen,
    }

# lengths match smoke yaml adapted to stub shapes: V=375 A=500
data = {
    "train": make_split(32, 1, texts[:32], labels[:32]),
    "valid": make_split(8, 2, texts[:8], labels[:8]),
    "test": make_split(8, 3, texts[:8], labels[:8]),
}
with open(out, "wb") as f:
    pickle.dump(data, f)
print("wrote", out, "bytes", out.stat().st_size)
print("ids max", data["train"]["text_bert"][:,0,:].max(), "mask unique", np.unique(data["train"]["text_bert"][:,1,:]))
