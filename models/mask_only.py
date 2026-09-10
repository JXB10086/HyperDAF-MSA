"""
Mask-only 对照模型  (阶段3 · 3.5 Hyper vs Mask-only)
--------------------------------------------------
用途: 回答 "Hyper 的收益是否仅仅来自『缺失模式编码』本身?"

    Mask-only : m -> MLP -> y           (完全不看模态内容 x_T/x_A/x_V)
    Hyper     : [x_T,x_A,x_V,m] -> z_hyper -> Predictor -> y

若 Hyper >> Mask-only, 说明收益主要来自 "用 Hyper 组织可用模态信息",
而不是单纯把 mask 编码进网络。

严格约束: 与阶段3 其它模型一致的禁令 (无 Missing Embedding / Dynamic Fusion /
Cross-Attention / MCAC / CCL / KL / 门控)。本模型仅作为下界对照。
"""
import torch
import torch.nn as nn


class MaskOnlyModel(nn.Module):
    """只用 3 维 modality mask 预测情感 (忽略模态内容), 作为对照下界。"""

    def __init__(self, hidden_dim=256, dropout=0.1, mask_dim=3):
        super().__init__()
        self.mask_dim = mask_dim
        self.mlp = nn.Sequential(
            nn.Linear(mask_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text, audio, vision, mask, return_rep=False):
        # 保持与其它模型一致的调用签名, 但【忽略】模态内容, 只用 mask
        m = mask.to(torch.float32)          # (B, 3)
        y = self.mlp(m).squeeze(-1)         # (B,)
        if return_rep:
            return y, m
        return y
