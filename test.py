"""
Baseline 测试脚本  (第一阶段 · Step 8)
--------------------------------------------------
加载 train.py 保存的最佳 checkpoint, 在 test 集上评估完整模态 T+A+V。
输出第一组 baseline 结果 (MAE / Corr), 由实际实验产生, 不预设数值。
同时在 valid 上评估作对照, 并将结果落盘到 experiments/。

运行:  python test.py   (需先运行 python train.py)
"""
import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from datasets import build_mosi_datasets      # noqa: E402
from models import FullModalBaseline          # noqa: E402
from utils import eval_regression             # noqa: E402
from configs.config import Config             # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, loader, device):
    """返回 (preds, labels, ids)。"""
    model.eval()
    preds_all, labels_all, ids_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            pred = model(text, audio, vision)
            preds_all.append(pred.cpu().numpy())
            labels_all.append(batch["label"].numpy())
            ids_all.extend(batch["id"])
    return np.concatenate(preds_all), np.concatenate(labels_all), ids_all


def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(cfg.ckpt_path):
        raise FileNotFoundError(f"未找到 checkpoint: {cfg.ckpt_path}\n请先运行: python train.py")

    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt["config"]
    model = FullModalBaseline(
        text_dim=mcfg["text_dim"], audio_dim=mcfg["audio_dim"], vision_dim=mcfg["vision_dim"],
        proj_dim=mcfg["proj_dim"], hidden_dim=mcfg["hidden_dim"], dropout=mcfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"已加载 checkpoint: epoch={ckpt['epoch']}  best_valid_mae={ckpt['best_valid_mae']:.4f}")

    datasets = build_mosi_datasets(cfg.data_path)
    valid_loader = DataLoader(datasets["valid"], batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)

    vpreds, vlabels, _ = predict(model, valid_loader, device)
    tpreds, tlabels, tids = predict(model, test_loader, device)
    vmetrics = eval_regression(vpreds, vlabels)
    tmetrics = eval_regression(tpreds, tlabels)

    print("\n" + "=" * 56)
    print("MOSI Baseline  (完整模态 T+A+V)")
    print("=" * 56)
    print(f"Valid  (N={len(vlabels)}) : MAE={vmetrics['MAE']:.4f}   Corr={vmetrics['Corr']:.4f}")
    print(f"Test   (N={len(tlabels)}) : MAE={tmetrics['MAE']:.4f}   Corr={tmetrics['Corr']:.4f}")
    print("=" * 56)

    print("test 抽样 (前8条): id | label | pred")
    for k in range(8):
        print(f"  {tids[k]:<18s} label={tlabels[k]:+.3f}  pred={tpreds[k]:+.3f}")

    # 结果落盘
    result = {
        "model": "FullModalBaseline",
        "modality": "T+A+V",
        "checkpoint_epoch": ckpt["epoch"],
        "best_valid_mae_from_training": ckpt["best_valid_mae"],
        "valid": vmetrics,
        "test": tmetrics,
        "n_valid": int(len(vlabels)),
        "n_test": int(len(tlabels)),
    }
    out_path = os.path.join(cfg.exp_dir, "baseline_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
