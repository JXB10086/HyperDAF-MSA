"""End-to-end wiring test: train_variant must save a checkpoint and stay bit-identical.

Runs the real Config, real model and real train_variant on a small synthetic
MOSEIDataset so that the torch.save path is exercised through the actual
training loop rather than through save_best_checkpoint alone.
"""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


V2_ROOT = Path(__file__).resolve().parents[1]
R1_ROOT = V2_ROOT / "r1_relational"
sys.path.insert(0, str(R1_ROOT))

import run_r1  # noqa: E402

from configs.config import Config  # noqa: E402
from datasets.mosei_dataset import MOSEIDataset  # noqa: E402


SEQ_LEN = 8
N_TRAIN = 16
N_VALID = 8
VARIANT = "REL"


def synthetic_split(n, cfg, seed):
    rng = np.random.default_rng(seed)
    shape = (n, SEQ_LEN)
    return MOSEIDataset(
        text=rng.standard_normal(shape + (cfg.text_dim,)).astype(np.float32),
        audio=rng.standard_normal(shape + (cfg.audio_dim,)).astype(np.float32),
        vision=rng.standard_normal(shape + (cfg.vision_dim,)).astype(np.float32),
        label=rng.uniform(-3.0, 3.0, size=n).astype(np.float32),
        ids=[f"seed{seed}-{i}" for i in range(n)],
    )


def tiny_config():
    cfg = Config()
    cfg.epochs = 1
    cfg.batch_size = 8
    cfg.num_workers = 0
    cfg.hyper_dim = 32
    cfg.hyper_hidden_dim = 32
    return cfg


def run_once(cfg, checkpoint_dir):
    datasets = {
        "train": synthetic_split(N_TRAIN, cfg, 0),
        "valid": synthetic_split(N_VALID, cfg, 1),
    }
    return run_r1.train_variant(
        VARIANT, 42, cfg, datasets, torch.device("cpu"), checkpoint_dir
    )


class TrainVariantCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg_plain = tiny_config()
        cls.model_plain, cls.info_plain = run_once(cls.cfg_plain, None)

        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg_ckpt = tiny_config()
        cls.model_ckpt, cls.info_ckpt = run_once(
            cls.cfg_ckpt, Path(cls.tmp.name) / "checkpoints"
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_checkpoint_disabled_by_default(self):
        self.assertIsNone(self.info_plain["checkpoint"])

    def test_checkpoint_written_with_provenance(self):
        record = self.info_ckpt["checkpoint"]
        self.assertIsNotNone(record)
        path = Path(record["path"])
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent.name, "checkpoints")
        self.assertEqual(record["bytes"], path.stat().st_size)
        self.assertEqual(
            record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def test_checkpoint_matches_reported_selection(self):
        payload = torch.load(
            self.info_ckpt["checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        self.assertEqual(payload["variant"], VARIANT)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["best_epoch"], self.info_ckpt["best_epoch"])
        self.assertAlmostEqual(
            payload["best_valid_mae"], self.info_ckpt["best_valid_mae"], places=12
        )
        self.assertEqual(payload["protocol"], run_r1.PROTOCOL)
        self.assertEqual(set(payload["state_dict"]), set(self.model_ckpt.state_dict()))

    def test_saved_weights_equal_loaded_model(self):
        payload = torch.load(
            self.info_ckpt["checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        live = self.model_ckpt.state_dict()
        for key, value in payload["state_dict"].items():
            self.assertTrue(torch.equal(value, live[key]), key)

    def test_history_and_hashes_are_checkpoint_independent(self):
        volatile = ("checkpoint", "runtime_sec")
        stripped_ckpt = {
            k: v for k, v in self.info_ckpt.items() if k not in volatile
        }
        stripped_plain = {
            k: v for k, v in self.info_plain.items() if k not in volatile
        }
        self.assertEqual(stripped_ckpt, stripped_plain)
        self.assertIsNotNone(self.info_ckpt["checkpoint"])

    def test_model_parameters_stay_bit_identical(self):
        a = self.model_plain.state_dict()
        b = self.model_ckpt.state_dict()
        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)


if __name__ == "__main__":
    unittest.main()