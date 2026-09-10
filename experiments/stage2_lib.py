"""
阶段2 训练 / 评估库
--------------------------------------------------
支撑阶段2 (模态缺失基线 Model A + Mask-aware Baseline Model B) 的核心逻辑:
  - build_model         构建 Model A(baseline) / Model B(mask_aware)
  - forward_model       统一前向: 先 x_i'=m_i·x_i 置零, 再按模型类型 forward
  - train_stage2_model  在指定 protocol 下训练, 按 valid MAE 选最佳 (绝不用 test)
  - evaluate_with_mask  用【预生成整split的mask】在任意 loader 上评估 MAE/Corr

公平性 & 可复现 (阶段2约束):
  - 所有模型: 相同 seed / 相同超参数 / 相同数据划分
  - 每次训练前重置 torch & numpy 种子 -> 相同初始化 + 相同 shuffle 顺序
  - Protocol B 训练 mask 用 seed=cfg.seed+global_step 逐 batch 生成,
    使 baseline 与 mask_aware 看到【完全相同】的缺失序列
  - valid / test 的 mask 预生成整个 split 后按 batch 切片 (loader shuffle=False)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import FullModalBaseline, MaskAwareBaseline
from utils import eval_regression, generate_missing_mask, apply_missing_mask

MODEL_TYPES = ("baseline", "mask_aware")
MODEL_LABEL = {"baseline": "Baseline(A)", "mask_aware": "MaskAware(B)"}

# 固定缺失测试条件: (显示名[缺失语义], simulator pattern[可用语义])
FIXED_TEST_CONDITIONS = [
    ("None",        "T+A+V"),   # 无缺失   -> [1,1,1]
    ("T missing",   "A+V"),     # 缺文本   -> [0,1,1]
    ("A missing",   "T+V"),     # 缺音频   -> [1,0,1]
    ("V missing",   "T+A"),     # 缺视觉   -> [1,1,0]
    ("T+A missing", "V"),       # 仅视觉   -> [0,0,1]
    ("T+V missing", "A"),       # 仅音频   -> [0,1,0]
    ("A+V missing", "T"),       # 仅文本   -> [1,0,0]
]


def build_model(model_type, cfg, device):
    """按类型构建模型; Model A/B 共享相同的 proj 结构, 唯一差异是 B 多拼 3 维 mask。"""
    common = dict(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim, vision_dim=cfg.vision_dim,
                  proj_dim=cfg.proj_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    if model_type == "baseline":
        return FullModalBaseline(**common).to(device)
    if model_type == "mask_aware":
        return MaskAwareBaseline(**common).to(device)
    raise ValueError(f"未知 model_type: {model_type}")


def forward_model(model, model_type, text, audio, vision, mask):
    """统一前向: 先置零缺失模态, 再按模型类型调用 (B 额外传入 mask)。"""
    tz, az, vz = apply_missing_mask(text, audio, vision, mask)
    if model_type == "mask_aware":
        return model(tz, az, vz, mask)
    return model(tz, az, vz)


def _mask_to_tensor(mask_np, device):
    return torch.tensor(np.asarray(mask_np), dtype=torch.float32, device=device)


def valid_mask_for(n_valid, protocol, cfg):
    """valid 集用于模型选择的 mask (固定, 每 epoch 一致)。"""
    if protocol == "A":
        return np.ones((n_valid, 3), dtype=np.int64)          # 完整模态
    return generate_missing_mask(n_valid,                                 # Protocol B: 随机缺失
                                 missing_probability=cfg.protocol_b_train_missing_prob,
                                 allow_all_missing=False, seed=cfg.seed)


def evaluate_with_mask(model, model_type, loader, mask_all, device, criterion=None):
    """
    在 shuffle=False 的 loader 上评估; mask_all 为整个 split 预生成的 (N,3), 按 batch 切片。
    返回 (avg_loss 或 None, {'MAE','Corr'})。
    """
    model.eval()
    preds, labels = [], []
    total, n, offset = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["label"].to(device)
            bs = label.size(0)
            mask = _mask_to_tensor(mask_all[offset:offset + bs], device)
            offset += bs
            pred = forward_model(model, model_type, text, audio, vision, mask)
            if criterion is not None:
                total += criterion(pred, label).item() * bs
            n += bs
            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
    metrics = eval_regression(np.concatenate(preds), np.concatenate(labels))
    avg_loss = (total / n) if criterion is not None else None
    return avg_loss, metrics


def train_stage2_model(model_type, protocol, cfg, datasets, device, verbose=False):
    """
    训练一个阶段2模型。
      model_type: 'baseline' | 'mask_aware'
      protocol  : 'A' (完整模态训练) | 'B' (训练即随机模拟缺失)
    模型选择: 仅用 valid MAE (Protocol A 用完整 valid, Protocol B 用随机缺失 valid)。
    返回: (装载了最佳权重的 model, info dict)
    """
    # 重置种子 -> 相同初始化 + 相同 shuffle + 相同 mask 序列 (跨模型可比)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = build_model(model_type, cfg, device)
    criterion = nn.L1Loss()   # MAE
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
                mask_np = np.ones((bs, 3), dtype=np.int64)     # 完整模态
            else:
                # Protocol B: 逐 batch 随机缺失, seed 随 step 递增 -> 可复现且跨模型一致
                mask_np = generate_missing_mask(bs, missing_probability=p_train,
                                                allow_all_missing=False,
                                                seed=cfg.seed + global_step)
            global_step += 1
            mask = _mask_to_tensor(mask_np, device)

            optimizer.zero_grad()
            pred = forward_model(model, model_type, text, audio, vision, mask)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            run += loss.item() * bs
            n += bs
        train_loss = run / n

        valid_loss, vm = evaluate_with_mask(model, model_type, valid_loader, v_mask, device, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss,
                        "valid_MAE": vm["MAE"], "valid_Corr": vm["Corr"]})
        if vm["MAE"] < best_mae:
            best_mae, best_epoch = vm["MAE"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    [{MODEL_LABEL[model_type]}|P{protocol}] Epoch {epoch:02d}/{cfg.epochs} "
                  f"train_loss={train_loss:.4f} valid_MAE={vm['MAE']:.4f} valid_Corr={vm['Corr']:.4f}")

    model.load_state_dict(best_state)
    info = {"model_type": model_type, "protocol": protocol, "best_epoch": best_epoch,
            "best_valid_mae": best_mae, "history": history}
    return model, info
