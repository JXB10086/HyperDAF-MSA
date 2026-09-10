"""
阶段3B 训练 / 评估库  (Hyper Representation 修正实验: D -> B1 -> B2)
--------------------------------------------------
模型集合 (4): mask_aware / hyper_h0(原始) / hyper_h1(可用模态池化) / hyper_h2(池化+LN+Res)

训练协议与 stage3_lib.train_stage3_model 【逐行一致】(相同 seed/划分/超参/Protocol A&B/
仅 valid 选模型), 因此 mask_aware 与 hyper_h0 应精确复现阶段3 结果 (内置一致性校验)。
评估 / 表征抽取 / 前向 直接复用 stage3_lib (它们与模型类型无关, 仅 baseline 特殊),
保证 3B 与 3 的口径完全可比。

禁令 (阶段3B): 无 Attention / Dynamic Fusion / MCAC / CCL / KL / Missing Embedding / 门控。
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import (MaskAwareBaseline, HyperRepresentationModel,
                    HyperAvailablePoolingModel, HyperAvailablePoolingResModel)
from utils import generate_missing_mask
from experiments.stage2_lib import valid_mask_for
# 复用阶段3 的与类型无关件 (forward 对非 baseline 一律传 mask, 适配 h0/h1/h2)
from experiments.stage3_lib import (forward_model_s3, evaluate_with_mask_s3,
                                    extract_representation_s3, count_params,
                                    _mask_to_tensor)

MODEL_TYPES_S3B = ("mask_aware", "hyper_h0", "hyper_h1", "hyper_h2")


def build_model_s3b(model_type, cfg, device):
    """按 3B 类型构建模型; hyper 系列统一 D=cfg.hyper_dim, 隐层=cfg.hyper_hidden_dim。"""
    common = dict(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim, vision_dim=cfg.vision_dim,
                  hyper_dim=cfg.hyper_dim, hidden_dim=cfg.hyper_hidden_dim, dropout=cfg.dropout)
    if model_type == "mask_aware":
        return MaskAwareBaseline(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim,
                                 vision_dim=cfg.vision_dim, proj_dim=cfg.proj_dim,
                                 hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    if model_type == "hyper_h0":
        return HyperRepresentationModel(**common).to(device)
    if model_type == "hyper_h1":
        return HyperAvailablePoolingModel(**common).to(device)
    if model_type == "hyper_h2":
        return HyperAvailablePoolingResModel(**common).to(device)
    raise ValueError(f"未知 model_type: {model_type}")


def train_stage3b_model(model_type, protocol, cfg, datasets, device, verbose=False):
    """
    训练一个阶段3B模型; 循环体与 stage3_lib.train_stage3_model 逐行一致。
      protocol: 'A' (完整模态训练) | 'B' (训练即随机缺失)
    返回 (装载最佳 valid 权重的 model, info dict)。
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = build_model_s3b(model_type, cfg, device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(datasets["train"], batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    valid_loader = DataLoader(datasets["valid"], batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)
    v_mask = valid_mask_for(len(datasets["valid"]), protocol, cfg)
    p_train = cfg.protocol_b_train_missing_prob

    best_mae, best_state, best_epoch = float("inf"), None, 0
    history = []
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["label"].to(device)
            bs = label.size(0)
            if protocol == "A":
                mask_np = np.ones((bs, 3), dtype=np.int64)
            else:
                mask_np = generate_missing_mask(bs, missing_probability=p_train,
                                                allow_all_missing=False,
                                                seed=cfg.seed + global_step)
            global_step += 1
            mask = _mask_to_tensor(mask_np, device)

            optimizer.zero_grad()
            pred = forward_model_s3(model, model_type, text, audio, vision, mask)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            run += loss.item() * bs
            n += bs
        train_loss = run / n

        valid_loss, vm = evaluate_with_mask_s3(model, model_type, valid_loader, v_mask, device, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss,
                        "valid_MAE": vm["MAE"], "valid_Corr": vm["Corr"]})
        if vm["MAE"] < best_mae:
            best_mae, best_epoch = vm["MAE"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    [{model_type}|P{protocol}] Epoch {epoch:02d}/{cfg.epochs} "
                  f"train_loss={train_loss:.4f} valid_MAE={vm['MAE']:.4f} valid_Corr={vm['Corr']:.4f}")

    model.load_state_dict(best_state)
    info = {"model_type": model_type, "protocol": protocol,
            "best_epoch": best_epoch, "best_valid_mae": best_mae,
            "n_params": count_params(model), "history": history}
    return model, info
