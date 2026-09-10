"""
Mask-aware Baseline (Model B)  (阶段2)
--------------------------------------------------
与 FullModalBaseline(Model A) 的【唯一区别】:
    Model A:  [x_T', x_A', x_V']          -> MLP
    Model B:  [x_T', x_A', x_V', m_T,m_A,m_V] -> MLP
其中缺失模态特征同样已置零 (x_i' = m_i·x_i, 在外部完成),
此处仅额外把 3 维 modality mask 拼接到融合向量, 显式告知模型"哪些模态缺失"。

严格约束 (阶段2第五、十一节):
  - proj_T/proj_A/proj_V 与 Model A 完全相同 (保证对比公平, 唯一变量是 mask 输入)
  - 不含 Missing Embedding / Cross-Attention / Hyper / 任何门控
"""
import torch
import torch.nn as nn


class MaskAwareBaseline(nn.Module):
    """Text/Audio/Vision 投影 + 3维缺失mask 拼接 -> MLP -> 标量情感回归。"""

    def __init__(self, text_dim=300, audio_dim=5, vision_dim=20,
                 proj_dim=128, hidden_dim=256, dropout=0.1, mask_dim=3):
        super().__init__()
        self.mask_dim = mask_dim
        # 与 Model A 相同的三个投影层
        self.proj_T = nn.Linear(text_dim, proj_dim)
        self.proj_A = nn.Linear(audio_dim, proj_dim)
        self.proj_V = nn.Linear(vision_dim, proj_dim)
        # 唯一区别: MLP 输入维度 +mask_dim
        self.mlp = nn.Sequential(
            nn.Linear(3 * proj_dim + mask_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text, audio, vision, mask, return_rep=False):
        # (B, L, Dm) -> (B, Dm): 时间维平均池化 (缺失模态已置零, 均值亦为0)
        t = text.mean(dim=1)
        a = audio.mean(dim=1)
        v = vision.mean(dim=1)
        xt = self.proj_T(t)   # (B, D)
        xa = self.proj_A(a)   # (B, D)
        xv = self.proj_V(v)   # (B, D)
        mask = mask.to(xt.dtype)                 # (B, 3)
        h = torch.cat([xt, xa, xv, mask], dim=-1)  # (B, 3D+3)
        y = self.mlp(h).squeeze(-1)              # (B,)
        # return_rep: 向后兼容地返回融合表征 h (阶段3 t-SNE/PCA 对比用), 默认路径不变
        if return_rep:
            return y, h
        return y
