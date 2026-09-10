"""
Missing-Aware Hyper Representation  (阶段3 · 第一项核心创新的最小可行版本)
--------------------------------------------------
目标 (阶段3): 单独验证 z_hyper 是否比 Mask-aware Representation 提供更 valuable 的
全局条件信息, 因此本版【刻意保持最纯粹】, 严格按用户 3.1~3.3 实现:

    Text   (B,50,300) --mean over time--> (B,300) --Linear--> (B,D)
    Audio  (B,50,5)   --mean over time--> (B,5)   --Linear--> (B,D)
    Vision (B,50,20)  --mean over time--> (B,20)  --Linear--> (B,D)

    h_input = [x_T, x_A, x_V, m]           (B, 3D+3)
    z_hyper = HyperEncoder(h_input)        (B, D)     <- 统一的高层情感条件表征
    y       = Predictor(z_hyper)           (B,)

与 Mask-aware Baseline(Model B) 的【唯一结构差异】:
    Mask-aware : [x_T,x_A,x_V,m] -> MLP -> y                     (单一映射)
    Hyper      : [x_T,x_A,x_V,m] -> HyperEncoder -> z_hyper -> Predictor -> y
    即 Hyper 显式地把 "可用模态+缺失模式" 压缩成一个 D 维统一表征 z_hyper, 再预测。

严格约束 (阶段3 第十一节禁令, 逐一核查):
  - 缺失模态仍严格 x_i'=m_i·x_i (在外部 apply_missing_mask 完成), 【不引入 Missing Embedding】
  - z_hyper 末端【不加激活】, 保持为纯表征向量 (利于后续 t-SNE / PCA)
  - 【不含】 Dynamic Fusion / Cross-Attention / MCAC / CCL / KL Prior / 任何门控
"""
import torch
import torch.nn as nn


class HyperRepresentationModel(nn.Module):
    """三模态投影到统一维度 D -> HyperEncoder 得到 z_hyper(D) -> Predictor 回归标量。"""

    def __init__(self, text_dim=300, audio_dim=5, vision_dim=20,
                 hyper_dim=256, hidden_dim=256, dropout=0.1, mask_dim=3):
        super().__init__()
        self.hyper_dim = hyper_dim
        self.mask_dim = mask_dim
        # 1) 三模态各自投影到统一维度 D (与 Baseline/MaskAware 相同的 mean-pooling + Linear 风格)
        self.proj_T = nn.Linear(text_dim, hyper_dim)
        self.proj_A = nn.Linear(audio_dim, hyper_dim)
        self.proj_V = nn.Linear(vision_dim, hyper_dim)
        # 2) HyperEncoder: [x_T,x_A,x_V,m] -> z_hyper (最小 2 层 MLP, 末端无激活)
        self.hyper_encoder = nn.Sequential(
            nn.Linear(3 * hyper_dim + mask_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hyper_dim),
        )
        # 3) Prediction MLP: z_hyper -> y (阶段3 不用 Dynamic Attention)
        self.predictor = nn.Sequential(
            nn.Linear(hyper_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text, audio, vision, mask, return_rep=False):
        # (B,L,Dm) -> (B,Dm): 时间维平均池化 (缺失模态已置零, 均值亦为0)
        t = text.mean(dim=1)
        a = audio.mean(dim=1)
        v = vision.mean(dim=1)
        xt = self.proj_T(t)   # (B, D)
        xa = self.proj_A(a)   # (B, D)
        xv = self.proj_V(v)   # (B, D)
        mask = mask.to(xt.dtype)                       # (B, 3)
        h = torch.cat([xt, xa, xv, mask], dim=-1)      # (B, 3D+3)
        z_hyper = self.hyper_encoder(h)                # (B, D)  <- 全局条件表征
        y = self.predictor(z_hyper).squeeze(-1)        # (B,)
        if return_rep:
            return y, z_hyper
        return y


class HyperAvailablePoolingModel(nn.Module):
    """阶段3B · B1 (H1): 可用模态池化 (Available-Modal Mean Pooling)。

    针对阶段3-D 诊断出的【文本依赖】问题: 不再把 [x_T,x_A,x_V,m] 当固定整体,
    而是先对【当前真正可用】的模态投影做 mask 加权平均:
        h_avail = sum_i m_i * x_i / max(sum_i m_i, 1)
        z_hyper = MLP([h_avail, m])
    与 H0 的唯一差异即 encoder 输入由 3D+3 变为 D+3 (聚合后)。
    不含 Attention / Dynamic Fusion / MCAC / CCL / KL / 门控。
    """

    def __init__(self, text_dim=300, audio_dim=5, vision_dim=20,
                 hyper_dim=256, hidden_dim=256, dropout=0.1, mask_dim=3):
        super().__init__()
        self.hyper_dim = hyper_dim
        self.mask_dim = mask_dim
        self.proj_T = nn.Linear(text_dim, hyper_dim)
        self.proj_A = nn.Linear(audio_dim, hyper_dim)
        self.proj_V = nn.Linear(vision_dim, hyper_dim)
        # HyperEncoder: [h_avail, m] -> z_hyper (最小 2 层 MLP, 末端无激活)
        self.hyper_encoder = nn.Sequential(
            nn.Linear(hyper_dim + mask_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hyper_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hyper_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _available_pool(self, xt, xa, xv, mask):
        """仅聚合 m_i=1 的模态投影 (mean), 缺失模态不贡献。"""
        x = torch.stack([xt, xa, xv], dim=1)     # (B,3,D)
        w = mask.unsqueeze(-1)                    # (B,3,1)
        cnt = w.sum(dim=1).clamp(min=1.0)         # (B,1)  防全缺(模拟器已禁止)除零
        return (x * w).sum(dim=1) / cnt           # (B,D)

    def forward(self, text, audio, vision, mask, return_rep=False):
        t = text.mean(dim=1); a = audio.mean(dim=1); v = vision.mean(dim=1)
        xt = self.proj_T(t); xa = self.proj_A(a); xv = self.proj_V(v)
        mask = mask.to(xt.dtype)
        h_avail = self._available_pool(xt, xa, xv, mask)   # (B,D)
        h = torch.cat([h_avail, mask], dim=-1)             # (B,D+3)
        z_hyper = self.hyper_encoder(h)                    # (B,D)
        y = self.predictor(z_hyper).squeeze(-1)
        if return_rep:
            return y, z_hyper
        return y


class HyperAvailablePoolingResModel(HyperAvailablePoolingModel):
    """阶段3B · B2 (H2): 在 B1 基础上加轻量 LayerNorm + Residual。

        z0      = MLP([h_avail, m])      (与 H1 完全相同的 encoder)
        z_hyper = z0 + LayerNorm(z0)     (抗维度塌缩的轻量修正)
    针对阶段3-D 诊断出的【有效秩过低(1.5)】问题。唯一新增参数: 一个 LayerNorm(D),
    其余与 H1 一致, 便于把增益归因到 "抗塌缩结构" 而非其他改动。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.hyper_dim)

    def forward(self, text, audio, vision, mask, return_rep=False):
        t = text.mean(dim=1); a = audio.mean(dim=1); v = vision.mean(dim=1)
        xt = self.proj_T(t); xa = self.proj_A(a); xv = self.proj_V(v)
        mask = mask.to(xt.dtype)
        h_avail = self._available_pool(xt, xa, xv, mask)
        h = torch.cat([h_avail, mask], dim=-1)
        z0 = self.hyper_encoder(h)
        z_hyper = z0 + self.ln(z0)                 # residual + LN
        y = self.predictor(z_hyper).squeeze(-1)
        if return_rep:
            return y, z_hyper
        return y
