"""
阶段3C 训练库: Missing-Pattern Consistency Regularization (B3)
--------------------------------------------------
以 H0 (HyperRepresentationModel) 为基础, 【只增加】一个缺失模式一致性正则项:

    L_total = L_task + λ_cons · L_cons
    L_cons  = mean_i ‖ z_full(x_i, m=[1,1,1]) − z_miss(x_i, m_cons_i) ‖₂²

设计严格遵循用户 Stage 3C 指令 (2026-09-05):
  - 模型结构 = H0 原封不动; 不引入 Attention / MCAC / CCL / KL / Missing Embedding /
    Dynamic Fusion / 任何门控。唯一改动是训练目标多了 L_cons。
  - View1 = 完整 m=[1,1,1]; View2 = 单模态缺失 (B3-1: 仅 T/A/V-missing 三选一, 每样本均匀采样)。
    All-Missing 不参与 L_cons; 双模态缺失 (B3-2) 暂不加入。
  - L_task 与 Protocol A/B 既有口径【逐行一致】:
      Protocol A: task mask = [1,1,1] (完整)
      Protocol B: task mask = 随机缺失 (p=cfg.protocol_b_train_missing_prob, seed=cfg.seed+step)
    ⇒ λ_cons=0 时完全退化为 train_stage3_model('hyper', ...) 的 H0, 可作 w/o-consistency 对照。
  - 模型选择仍仅用 valid MAE; test 仅最终评估。

可复现性:
  - task mask 沿用 seed=cfg.seed+global_step (与 H0/baseline/mask 完全相同的缺失序列)。
  - consistency 的缺失视图 mask 用【独立确定性种子流】 seed=cfg.seed+CONS_SEED_OFF+global_step,
    与 task mask 互不干扰。
  - Protocol A 下 task 视图本身即完整视图, 故直接复用 task 前向的 z 作为 z_full (省一次前向,
    且不改变任何 RNG 消耗顺序)。
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import HyperRepresentationModel
from utils import generate_missing_mask
from experiments.stage2_lib import valid_mask_for
from experiments.stage3_lib import forward_model_s3, evaluate_with_mask_s3, count_params

# 一致性缺失视图 mask 的独立种子偏移 (与 task mask 流分离, 保证互不干扰且可复现)
CONS_SEED_OFF = 777777

# B3-1: 单模态缺失 (恰好 drop 一个模态) 的三种 mask, 顺序 [m_T, m_A, m_V]
#   [0,1,1]=T missing  [1,0,1]=A missing  [1,1,0]=V missing
# 不含 all-missing [0,0,0], 不含双模态缺失 (B3-2 再议)。
SINGLE_MISSING_MASKS = np.array([[0, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 0]], dtype=np.int64)


def _mask_to_tensor(mask_np, device):
    return torch.tensor(np.asarray(mask_np), dtype=torch.float32, device=device)


def _single_missing_cons_mask(bs, seed):
    """每样本均匀采样一个 '单模态缺失' 视图 mask (B3-1); 返回 (bs,3) int64。"""
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, SINGLE_MISSING_MASKS.shape[0], size=bs)
    return SINGLE_MISSING_MASKS[pick]


def train_stage3c_consistency(protocol, cfg, datasets, device, lam_cons=0.05,
                              hyper_dim=None, stop_grad_full=False, cons_norm=None,
                              model_factory=None, epoch_diag_hook=None, verbose=False):
    """
    训练 H0 + 缺失模式一致性正则 (B3)。返回 (装载最佳权重的 model, info dict)。

      protocol       : 'A' (完整模态训练) | 'B' (训练即随机模拟缺失) —— L_task 口径与 stage3 一致
      lam_cons       : λ_cons; =0 时退化为纯 H0 (w/o consistency 对照, 与 train_stage3_model 逐位一致)
      hyper_dim      : z_hyper 维度 D (默认 cfg.hyper_dim)
      stop_grad_full : True -> 对 full 视图停梯度 (sg(z_full) 作锚点, 用户 2026-09-05 裁决 B)。
      cons_norm      : None -> 直接用原始 z (未归一化);
                       'l2' -> 先把 z_full/z_miss 各自 L2 单位化再算一致性 (BYOL 式)。
                       注: 单独的 stop_grad_full + 未归一化 z 会因 "共享在线编码器使 sg 目标棘轮上抬" 而【发散爆炸】
                       (已实测: ‖z‖→1e8, Corr→0); 必须配合 cons_norm='l2' 才数值稳定。
      model_factory  : Stage 4 用。None -> 默认构建 H0(HyperRepresentationModel), F0 逐位复现封版 B3;
                       否则调用 model_factory() 得模型实例 (在种子重置后构建, 保证确定性 init)。
                       【不改动】下方 consistency loss / 种子流 / mask 流任何一行, F1/F2/F3 共享同一 consistency。
      epoch_diag_hook: Stage 4 Round-2 用。None -> 不记录额外诊断 (与封版行为逐位一致);
                       否则每 epoch 调用 hook(model, valid_loader, device) -> dict, 合并进该 epoch 的
                       history 条目。hook 仅做 eval+no_grad 前向 (dropout 关闭), 【不消耗 RNG、不改梯度】,
                       故不影响训练轨迹与可复现性。
    模型选择: 仅用 valid MAE (绝不用 test)。
    训练期额外追踪【原始 z】三范数 (epoch 均值): ‖z_full‖ / ‖z_miss‖ / ‖z_full−z_miss‖,
    用于判断一致性是靠 "真正靠近" 还是 "整体缩小/爆炸" 实现。
    """
    # 重置种子 -> 与 stage3 相同的初始化 + 相同 shuffle + 相同 task mask 序列
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    D = int(hyper_dim if hyper_dim is not None else cfg.hyper_dim)
    if model_factory is None:
        model = HyperRepresentationModel(text_dim=cfg.text_dim, audio_dim=cfg.audio_dim,
                                         vision_dim=cfg.vision_dim, hyper_dim=D,
                                         hidden_dim=cfg.hyper_hidden_dim,
                                         dropout=cfg.dropout).to(device)
    else:
        model = model_factory().to(device)   # Stage 4 融合模型 (种子重置后构建 -> 确定性 init, backbone 与 F0 逐字节一致)
    criterion = nn.L1Loss()   # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(datasets["train"], batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    valid_loader = DataLoader(datasets["valid"], batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)
    v_mask = valid_mask_for(len(datasets["valid"]), protocol, cfg)
    p_train = cfg.protocol_b_train_missing_prob
    lam = float(lam_cons)
    use_cons = lam > 0.0
    sg = bool(stop_grad_full)
    norm = cons_norm            # None | 'l2'

    best_mae, best_state, best_epoch = float("inf"), None, 0
    history = []
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_task, run_cons, n = 0.0, 0.0, 0
        run_nf, run_nm, run_nd = 0.0, 0.0, 0.0   # 三范数累计 (‖z_full‖/‖z_miss‖/‖z_full−z_miss‖)
        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["label"].to(device)
            bs = label.size(0)

            # ---- L_task: 与 stage3 Protocol A/B 逐行一致 ----
            if protocol == "A":
                task_mask_np = np.ones((bs, 3), dtype=np.int64)      # 完整模态
            else:
                task_mask_np = generate_missing_mask(bs, missing_probability=p_train,
                                                     allow_all_missing=False,
                                                     seed=cfg.seed + global_step)
            # ---- L_cons 的单模态缺失视图 mask (独立种子流) ----
            cons_mask_np = (_single_missing_cons_mask(bs, cfg.seed + CONS_SEED_OFF + global_step)
                            if use_cons else None)
            global_step += 1

            task_mask = _mask_to_tensor(task_mask_np, device)
            optimizer.zero_grad()

            # Protocol A 且启用一致性且未停梯度: task 视图 == 完整视图, 复用其 z 作为 z_full (省一次前向)。
            # 停梯度模式下不复用 (否则 detach 会切断 task 前向的梯度)。
            reuse_full = use_cons and protocol == "A" and not sg
            if reuse_full:
                pred, z_full = forward_model_s3(model, "hyper", text, audio, vision,
                                                task_mask, return_rep=True)
            else:
                pred = forward_model_s3(model, "hyper", text, audio, vision, task_mask)
            loss_task = criterion(pred, label)

            if use_cons:
                if not reuse_full:
                    full_mask = _mask_to_tensor(np.ones((bs, 3), dtype=np.int64), device)
                    _, z_full = forward_model_s3(model, "hyper", text, audio, vision,
                                                 full_mask, return_rep=True)
                cons_mask = _mask_to_tensor(cons_mask_np, device)
                _, z_miss = forward_model_s3(model, "hyper", text, audio, vision,
                                             cons_mask, return_rep=True)
                # 停梯度: full 视图作锚点 (sg(z_full)), 一致性梯度只更新 z_miss 通路
                z_anchor = z_full.detach() if sg else z_full
                # 可选 L2 单位化 (BYOL 式): 防止未归一化 sg 的 "目标棘轮上抬" 发散
                if norm == "l2":
                    z_a = z_anchor / (z_anchor.norm(dim=1, keepdim=True) + 1e-8)
                    z_m = z_miss / (z_miss.norm(dim=1, keepdim=True) + 1e-8)
                else:
                    z_a, z_m = z_anchor, z_miss
                # L_cons = mean_i ‖ z_a_i − z_m_i ‖₂²  (按维度求平方和, 再对 batch 取均值)
                loss_cons = ((z_a - z_m) ** 2).sum(dim=1).mean()
                loss = loss_task + lam * loss_cons
                cons_val = float(loss_cons.item())
                # 三范数追踪 (无梯度, 基于【原始 z】): 判断 "靠近" vs "整体缩小/爆炸"
                with torch.no_grad():
                    zf = z_full.detach(); zm = z_miss.detach()
                    run_nf += float(zf.norm(dim=1).mean().item()) * bs
                    run_nm += float(zm.norm(dim=1).mean().item()) * bs
                    run_nd += float((zf - zm).norm(dim=1).mean().item()) * bs
            else:
                loss = loss_task
                cons_val = 0.0

            loss.backward()
            optimizer.step()
            run_task += float(loss_task.item()) * bs
            run_cons += cons_val * bs
            n += bs
        train_task = run_task / n
        train_cons = run_cons / n
        nf = run_nf / n if use_cons else 0.0
        nm = run_nm / n if use_cons else 0.0
        nd = run_nd / n if use_cons else 0.0

        valid_loss, vm = evaluate_with_mask_s3(model, "hyper", valid_loader, v_mask, device, criterion)
        entry = {"epoch": epoch, "train_task": train_task, "train_cons": train_cons,
                 "norm_full": nf, "norm_miss": nm, "norm_diff": nd,
                 "valid_loss": valid_loss, "valid_MAE": vm["MAE"], "valid_Corr": vm["Corr"]}
        if epoch_diag_hook is not None:
            entry.update(epoch_diag_hook(model, valid_loader, device))   # 逐 epoch 融合诊断 (仅 eval 前向)
        history.append(entry)
        if vm["MAE"] < best_mae:
            best_mae, best_epoch = vm["MAE"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    [B3|P{protocol}|λ={lam}|sg={int(sg)}|norm={norm}] Epoch {epoch:02d}/{cfg.epochs} "
                  f"task={train_task:.4f} cons={train_cons:.4f} "
                  f"‖zf‖={nf:.3f} ‖zm‖={nm:.3f} ‖zf-zm‖={nd:.3f} "
                  f"valid_MAE={vm['MAE']:.4f} valid_Corr={vm['Corr']:.4f}")

    model.load_state_dict(best_state)
    info = {"model_type": "hyper_b3", "protocol": protocol, "hyper_dim": D,
            "lam_cons": lam, "stop_grad_full": sg, "cons_norm": norm, "best_epoch": best_epoch,
            "best_valid_mae": best_mae, "n_params": count_params(model), "history": history}
    return model, info
