"""Round-trip tests for the validation-selected checkpoint instrumentation in run_r1."""

import hashlib
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


V2_ROOT = Path(__file__).resolve().parents[1]
R1_ROOT = V2_ROOT / "r1_relational"
sys.path.insert(0, str(R1_ROOT))

import run_r1  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.norm = nn.LayerNorm(2)


class CheckpointTests(unittest.TestCase):
    def test_save_round_trip_records_provenance(self):
        torch.manual_seed(0)
        model = TinyModel()
        state = model.state_dict()

        with tempfile.TemporaryDirectory() as tmp:
            info = run_r1.save_best_checkpoint(
                Path(tmp) / "nested" / "checkpoints", "REL", 43, 7, 0.6123, state
            )

            path = Path(info["path"])
            self.assertEqual(path.name, "REL_seed43_best_val_mae.pth")
            self.assertTrue(path.is_file())
            self.assertEqual(info["bytes"], path.stat().st_size)

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(info["sha256"], digest)
            self.assertEqual(run_r1.file_sha256(path), digest)

            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["variant"], "REL")
            self.assertEqual(payload["seed"], 43)
            self.assertEqual(payload["best_epoch"], 7)
            self.assertAlmostEqual(payload["best_valid_mae"], 0.6123, places=12)
            self.assertEqual(payload["protocol"], run_r1.PROTOCOL)
            self.assertEqual(payload["selection"], "validation MAE only")
            self.assertAlmostEqual(payload["lambda_point"], run_r1.LAMBDA_POINT)
            self.assertAlmostEqual(payload["lambda_rel"], run_r1.LAMBDA_REL)

            reloaded = payload["state_dict"]
            self.assertEqual(set(reloaded), set(state))
            for key, value in state.items():
                self.assertTrue(torch.equal(reloaded[key], value.cpu()), key)
                self.assertEqual(reloaded[key].device.type, "cpu")

            rebuilt = TinyModel()
            rebuilt.load_state_dict(reloaded)
            for original, restored in zip(model.parameters(), rebuilt.parameters()):
                self.assertTrue(torch.equal(original, restored))

    def test_saved_state_does_not_alias_live_model(self):
        torch.manual_seed(1)
        model = TinyModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(
                run_r1.save_best_checkpoint(
                    tmp, "H0", 42, 1, 1.0, model.state_dict()
                )["path"]
            )
            with torch.no_grad():
                model.linear.weight.add_(100.0)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertFalse(
                torch.equal(payload["state_dict"]["linear.weight"], model.linear.weight)
            )

    def test_variant_seed_is_encoded_in_filename(self):
        torch.manual_seed(2)
        state = TinyModel().state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            names = {
                Path(
                    run_r1.save_best_checkpoint(tmp, v, s, 1, 1.0, state)["path"]
                ).name
                for v in ("H0", "B3_POINT", "REL")
                for s in (42, 43, 44)
            }
        self.assertEqual(len(names), 9)

    def test_checkpointing_is_opt_in(self):
        default = inspect.signature(run_r1.train_variant).parameters["checkpoint_dir"]
        self.assertIsNone(default.default)


if __name__ == "__main__":
    unittest.main()