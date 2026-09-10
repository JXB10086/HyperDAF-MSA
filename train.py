"""
Baseline 训练脚本  (第一阶段 · Step 7)
--------------------------------------------------
仅训练完整模态 T+A+V, 不涉及缺失/mask。
目标: 确认 loss 能下降, 记录 train_loss / valid_loss / MAE / Corr。
损失: L1Loss(MAE), 对应 MOSI 连续情感回归 (项目说明第九节)。
选择: 按 valid MAE 保存最佳 checkpoint (不使用 test 调参)。

运行:  python train.py
"""
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
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
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, criterion):
    """在给定 loader 上评估, 返回 (平均loss, {'MAE','Corr'})。"""
    model.eval()
    preds_all, labels_all = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["label"].to(device)
            pred = model(text, audio, vision)
            loss = criterion(pred, label)
            total_loss += loss.item() * label.size(0)
            n += label.size(0)
            preds_all.append(pred.cpu().numpy())
            labels_all.append(label.cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return total_loss / n, eval_regression(preds, labels)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.exp_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print(f"Device: {device} | Seed: {cfg.seed}")
    print(f"模型: proj_dim={cfg.proj_dim} hidden_dim={cfg.hidden_dim} dropout={cfg.dropout}")
    print(f"训练: batch_size={cfg.batch_size} lr={cfg.lr} epochs={cfg.epochs} loss=L1(MAE)")
    print("=" * 64)

    print("加载数据中...")
    datasets = build_mosi_datasets(cfg.data_path)
    train_loader = DataLoader(datasets["train"], batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    valid_loader = DataLoader(datasets["valid"], batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)
    print(f"Train={len(datasets['train'])}  Valid={len(datasets['valid'])}  Test={len(datasets['test'])}")

    model = FullModalBaseline(
        text_dim=cfg.text_dim, audio_dim=cfg.audio_dim, vision_dim=cfg.vision_dim,
        proj_dim=cfg.proj_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}\n")

    criterion = nn.L1Loss()  # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_mae = float("inf")
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, n = 0.0, 0
        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            pred = model(text, audio, vision)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * label.size(0)
            n += label.size(0)
        train_loss = running_loss / n

        valid_loss, vm = evaluate(model, valid_loader, device, criterion)
        history.append((epoch, train_loss, valid_loss, vm["MAE"], vm["Corr"]))
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train_loss={train_loss:.4f} | "
              f"valid_loss={valid_loss:.4f} | valid MAE={vm['MAE']:.4f} Corr={vm['Corr']:.4f}")

        if vm["MAE"] < best_mae:
            best_mae = vm["MAE"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_valid_mae": best_mae,
                "valid_metrics": vm,
                "config": {
                    "text_dim": cfg.text_dim, "audio_dim": cfg.audio_dim,
                    "vision_dim": cfg.vision_dim, "proj_dim": cfg.proj_dim,
                    "hidden_dim": cfg.hidden_dim, "dropout": cfg.dropout,
                    "seed": cfg.seed,
                },
            }, cfg.ckpt_path)

    elapsed = time.time() - t0
    first_train, last_train = history[0][1], history[-1][1]
    print("\n" + "=" * 64)
    print(f"训练耗时: {elapsed:.1f}s")
    print(f"train_loss: 首 epoch={first_train:.4f} -> 末 epoch={last_train:.4f}  "
          f"({'✅ 下降' if last_train < first_train else '❌ 未下降'})")
    print(f"最佳 valid MAE={best_mae:.4f}")
    print(f"Checkpoint 已保存: {cfg.ckpt_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
