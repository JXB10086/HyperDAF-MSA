"""
完整模态 Baseline  (第一阶段 · Step 5)
--------------------------------------------------
结构 (严格按项目说明第二十节):
    Text   (B,50,300) --mean over time--> (B,300) --Linear--> (B,D)
    Audio  (B,50,5)   --mean over time--> (B,5)   --Linear--> (B,D)
    Vision (B,50,20)  --mean over time--> (B,20)  --Linear--> (B,D)
    Concat -> (B,3D) -> MLP -> (B,1) -> squeeze -> (B,)

约束:
  - 第一阶段仅跑完整模态 T+A+V, 不涉及缺失/mask
  - 时间维用最简单的 mean pooling (Mask-aware pooling 属于后续阶段)
  - 不含 Hyper Representation / MCAC / CCL / KL 等创新模块
"""
import torch
import torch.nn as nn


class FullModalBaseline(nn.Module):
    """Text/Audio/Vision 各自 Linear 投影 -> Concat -> MLP -> 标量情感回归。"""

    def __init__(self, text_dim=300, audio_dim=5, vision_dim=20,
                 proj_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.proj_T = nn.Linear(text_dim, proj_dim)
        self.proj_A = nn.Linear(audio_dim, proj_dim)
        self.proj_V = nn.Linear(vision_dim, proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text, audio, vision, return_rep=False):
        # (B, L, Dm) -> (B, Dm): 时间维平均池化得到全局模态表示
        t = text.mean(dim=1)
        a = audio.mean(dim=1)
        v = vision.mean(dim=1)
        # 统一投影到 proj_dim
        xt = self.proj_T(t)   # (B, D)
        xa = self.proj_A(a)   # (B, D)
        xv = self.proj_V(v)   # (B, D)
        h = torch.cat([xt, xa, xv], dim=-1)   # (B, 3D)
        y = self.mlp(h).squeeze(-1)           # (B,)
        # return_rep: 向后兼容地返回融合表征 h (阶段3 t-SNE/PCA 对比用), 默认路径不变
        if return_rep:
            return y, h
        return y
