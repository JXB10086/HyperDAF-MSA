"""
阶段3 训练 / 评估库  (Missing-Aware Hyper Representation)
--------------------------------------------------
在【完全复用阶段2 公平协议】的前提下, 把模型集合扩展到:
  baseline(A) / mask_aware(B) / hyper(C, 可变维度 D) / mask_only(对照)

关键函数:
  - build_specs          生成 7 个模型规格 (3 主模型 + 4 个 Hyper 维度)
  - build_model_s3       按规格构建模型 (hyper 支持指定 D)
  - forward_model_s3     统一前向: 先 x_i'=m_i·x_i 置零, 再按类型 forward (可 return_rep)
  - train_stage3_model   训练协议与 stage2 逐行一致 -> baseline/mask_aware 可复现阶段2结果
  - evaluate_with_mask_s3 / extract_representation_s3
  - pca_2d               numpy SVD 实现的 PCA (无 sklearn 依赖), 供表征可视化

公平性 & 可复现 (继承阶段2约束):
  - 所有模型: 相同 seed / 相同超参数 / 相同数据划分 / 相同 L1Loss+Adam
  - 每次训练前重置 torch & numpy 种子 -> 相同初始化 + 相同 shuffle 顺序
  - Protocol B 训练 mask 用 seed=cfg.seed+global_step 逐 batch 生成,
    使所有模型看到【完全相同】的缺失序列
  - valid / test 的 mask 预生成整个 split 后按 batch 切片 (loader shuffle=False)
  - 仅用 valid MAE 选模型, 绝不用 test 调参
禁令 (阶段3): 无 Missing Embedding / Dynamic Fusion / Cross-Attention / MCAC / CCL / KL / 门控。
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import (FullModalBaseline, MaskAwareBaseline,
                    HyperRepresentationModel, MaskOnlyModel)
from utils import eval_regression, generate_missing_mask, apply_missing_mask
# 复用阶段2 中与模型无关的公共件, 保证协议一致 (单一事实来源)
from experiments.stage2_lib import FIXED_TEST_CONDITIONS, valid_mask_for

MODEL_TYPES_S3 = ("baseline", "mask_aware", "hyper", "mask_only")


def _mask_to_tensor(mask_np, device):
    return torch.tensor(np.asarray(mask_np), dtype=torch.float32, device=device)


def build_specs(cfg):
    """
    7 个模型规格: Baseline / MaskAware / MaskOnly + Hyper(D) for D in cfg.hyper_dim_ablation。
    主对比 (3.4) 用的 Hyper = hyper_D{cfg.hyper_dim} (默认 256), 已包含在消融列表中。
    """
    specs = [
        {"key": "baseline",   "type": "baseline",   "dim": None, "label": "Baseline(A)"},
        {"key": "mask_aware", "type": "mask_aware", "dim": None, "label": "MaskAware(B)"},
        {"key": "mask_only",  "type": "mask_only",  "dim": None, "label": "MaskOnly"},
    ]
    for d in cfg.hyper_dim_ablation:
        specs.append({"key": f"hyper_D{d}", "type": "hyper", "dim": d, "label": f"Hyper(D={d})"})
    return specs


def build_model_s3(model_type, cfg, device, hyper_dim=None):
    """按类型构建模型。baseline/mask_aware 沿用阶段2 维度(proj=128,hidden=256); hyper 用 D。"""
    if model_type == "baseline":
        return FullModalBaseline(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim, vision_dim=cfg.vision_dim,
                                 proj_dim=cfg.proj_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    if model_type == "mask_aware":
        return MaskAwareBaseline(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim, vision_dim=cfg.vision_dim,
                                 proj_dim=cfg.proj_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    if model_type == "hyper":
        D = int(hyper_dim if hyper_dim is not None else cfg.hyper_dim)
        return HyperRepresentationModel(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim, vision_dim=cfg.vision_dim,
                                        hyper_dim=D, hidden_dim=cfg.hyper_hidden_dim, dropout=cfg.dropout).to(device)
    if model_type == "mask_only":
        return MaskOnlyModel(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    raise ValueError(f"未知 model_type: {model_type}")


def forward_model_s3(model, model_type, text, audio, vision, mask, return_rep=False):
    """统一前向: 先置零缺失模态, 再按类型调用。baseline 不吃 mask; 其余吃 mask。"""
    tz, az, vz = apply_missing_mask(text, audio, vision, mask)
    if model_type == "baseline":
        return model(tz, az, vz, return_rep=return_rep)
    return model(tz, az, vz, mask, return_rep=return_rep)


def count_params(model):
    """可训练参数量 (用于三模型公平性透明化: 报告各自规模)。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_with_mask_s3(model, model_type, loader, mask_all, device, criterion=None):
    """shuffle=False loader 上评估; mask_all 为整 split 预生成 (N,3), 按 batch 切片。"""
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
            pred = forward_model_s3(model, model_type, text, audio, vision, mask)
            if criterion is not None:
                total += criterion(pred, label).item() * bs
            n += bs
            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
    metrics = eval_regression(np.concatenate(preds), np.concatenate(labels))
    avg_loss = (total / n) if criterion is not None else None
    return avg_loss, metrics


def extract_representation_s3(model, model_type, loader, mask_all, device):
    """
    抽取表征 (3.11/3.12): hyper -> z_hyper(D); baseline/mask_aware -> 融合向量 h; mask_only -> m。
    返回 dict: {'rep': (N,d) np, 'pred': (N,) np, 'label': (N,) np}
    """
    model.eval()
    reps, preds, labels = [], [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["label"].to(device)
            bs = label.size(0)
            mask = _mask_to_tensor(mask_all[offset:offset + bs], device)
            offset += bs
            pred, rep = forward_model_s3(model, model_type, text, audio, vision, mask, return_rep=True)
            reps.append(rep.cpu().numpy())
            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
    return {"rep": np.concatenate(reps), "pred": np.concatenate(preds), "label": np.concatenate(labels)}


def train_stage3_model(model_type, protocol, cfg, datasets, device, hyper_dim=None, verbose=False):
    """
    训练一个阶段3模型; 训练协议与 stage2_lib.train_stage2_model 逐行一致,
    因此 baseline / mask_aware 的结果应【精确复现】阶段2 (内置一致性校验)。
      model_type: 'baseline' | 'mask_aware' | 'hyper' | 'mask_only'
      protocol  : 'A' (完整模态训练) | 'B' (训练即随机模拟缺失)
      hyper_dim : 仅 hyper 用, 指定 z_hyper 维度 D
    模型选择: 仅用 valid MAE。返回 (装载最佳权重的 model, info dict)。
    """
    # 重置种子 -> 相同初始化 + 相同 shuffle + 相同 mask 序列 (跨模型可比 & 可复现)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = build_model_s3(model_type, cfg, device, hyper_dim=hyper_dim)
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
    info = {"model_type": model_type, "protocol": protocol, "hyper_dim": hyper_dim,
            "best_epoch": best_epoch, "best_valid_mae": best_mae,
            "n_params": count_params(model), "history": history}
    return model, info


def pca_2d(X):
    """
    numpy SVD 实现的 PCA -> 2D (无 sklearn 依赖)。
    返回 (coords (N,2), explained_var_ratio (2,))。供 3.11/3.12 表征可视化。
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # 经济型 SVD: Xc = U S Vt, 前两个右奇异向量即主成分方向
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    var = (S ** 2) / ((S ** 2).sum() + 1e-12)
    return coords, var[:2]
