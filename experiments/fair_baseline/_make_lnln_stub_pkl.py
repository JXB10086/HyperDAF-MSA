"""Create minimal MMSA-schema MOSI stub for LNLN dataloader smoke (NOT real data)."""
import os
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "mmsa", "MOSI", "Processed", "unaligned_50.pkl")


def _split(n_train=128, n_valid=32, n_test=32, seed=0):
    rng = np.random.RandomState(seed)
    split = {}
    for name, n in [("train", n_train), ("valid", n_valid), ("test", n_test)]:
        split[name] = {
            "text_bert": rng.randint(0, 30000, size=(n, 3, 50)).astype(np.float32),
            "audio": rng.randn(n, 500, 5).astype(np.float32),
            "vision": rng.randn(n, 375, 20).astype(np.float32),
            "audio_lengths": rng.randint(100, 500, size=n).astype(np.int64),
            "vision_lengths": rng.randint(100, 375, size=n).astype(np.int64),
            "regression_labels": rng.uniform(-3, 3, size=(n, 1, 1)).astype(np.float32),
            "raw_text": [f"sample_{name}_{i}" for i in range(n)],
            "id": [f"vid_{name}_{i}$clip" for i in range(n)],
        }
    return split


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(_split(), f)
    print("Wrote stub:", OUT, "bytes", os.path.getsize(OUT))
