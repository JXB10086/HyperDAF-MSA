"""
Stage 4 · Hyper-Guided Dynamic Fusion (第一轮 · 最小实现)
--------------------------------------------------
用户 2026-09-05 裁决: B3 封版 (λ_cons=0.005), 进入 Stage 4。本文件【只】实现最小动态融合,
在已封版的 H0 backbone (proj_T/A/V + hyper_encoder + predictor, z_hyper 计算逐字节不变) 之上,
增加一个动态融合头:

    Q    = W_Q z_hyper                        (仅 F3; 样本级"统一情感条件")
    K_i  = W_K x_i ,  V_i = W_V x_i           (i∈{T,A,V}; x_i = proj_i(mean-pool) ∈ R^D)
    s_i  = Q·K_i^T / sqrt(D)
    s_i  = -inf   当 m_i = 0                   (缺失模态 attention logit 置 -inf)
    α    = softmax(s)                          (α_missing 严格 = 0)
    h_fuse = z_hyper + Σ_i α_i V_i
    y      = Predictor(h_fuse)

四对照 (仅 α 的来源不同, 其余 backbone / V_i / h_fuse / consistency 完全共享):
    F0 = H0 (z_hyper -> Predictor, 无融合)             —— 即 HyperRepresentationModel, 不在本文件
    F1 = MaskGateFusion     α <- MLP([x_T,x_A,x_V,m])          (简单 mask-aware gate, 不用 z_hyper)
    F2 = GlobalQueryFusion  α <- softmax(q_global·K_i/√D)      (静态可学习 query, 样本无关)
    F3 = HyperGuidedFusion  α <- softmax(W_Q z_hyper·K_i/√D)   (样本级 hyper-guided, 本项创新)

严格约束 (Stage 4 第一轮禁令): 不含 MCAC / CCL / KL / MoE / 专家网络 / Prompt / Reliability Network。
唯一新增 = 动态融合头。z_hyper 仍由【未修改】的 hyper_encoder 产生, 并继续受 B3 归一化
stop-gradient 一致性 (λ=0.005) 约束 (在 stage3c_lib 中, 与本文件无关)。

公平性保证: 所有子类 __init__ 先调用 super().__init__() 构建 H0 backbone (相同 RNG 消耗顺序 ->
与 F0/H0 的 backbone 初始化逐字节一致), 之后才创建融合头参数; 因此 F0/F1/F2/F3 共享完全相同的
backbone 初始权重, 唯一变量是融合头的 α 计算方式。训练轨迹因结构不同而不同 (符合预期)。
"""
import torch
import torch.nn as nn

from models.hyper_representation import HyperRepresentationModel


class HyperFusionBase(HyperRepresentationModel):
    """Stage 4 融合基类: 复用 H0 backbone, 在其上加动态融合头。子类只需实现 _alpha_logits。"""

    def __init__(self, text_dim=300, audio_dim=5, vision_dim=20, hyper_dim=256,
                 hidden_dim=256, dropout=0.1, mask_dim=3):
        # 先构建 H0 backbone (proj_T/A/V + hyper_encoder + predictor), RNG 顺序与 F0 完全一致
        super().__init__(text_dim=text_dim, audio_dim=audio_dim, vision_dim=vision_dim,
                         hyper_dim=hyper_dim, hidden_dim=hidden_dim, dropout=dropout,
                         mask_dim=mask_dim)
        self._hidden_dim = hidden_dim
        self._dropout = dropout
        D = self.hyper_dim
        # 所有融合变体共享的 V 投影 (把"α 来源"隔离为唯一变量)
        self.W_V = nn.Linear(D, D)
        self._scale = D ** 0.5
        self._neg = -1e9                        # 用大负数代替 -inf, softmax 后严格下溢为 0, 且不产生 NaN

    # ---- 子类实现: 返回融合前 logit s (B,3) ----
    def _alpha_logits(self, xt, xa, xv, z_hyper, maskf):
        raise NotImplementedError

    def _backbone(self, text, audio, vision, mask):
        """与 H0.forward 逐字节一致地产出 x_T/x_A/x_V 与 z_hyper (缺失模态已在外部置零)。"""
        t = text.mean(dim=1); a = audio.mean(dim=1); v = vision.mean(dim=1)
        xt = self.proj_T(t); xa = self.proj_A(a); xv = self.proj_V(v)
        maskf = mask.to(xt.dtype)                       # (B,3)
        h = torch.cat([xt, xa, xv, maskf], dim=-1)      # (B,3D+3)
        z_hyper = self.hyper_encoder(h)                 # (B,D)  <- 与 H0 完全相同
        return xt, xa, xv, z_hyper, maskf

    def forward(self, text, audio, vision, mask, return_rep=False, return_alpha=False):
        xt, xa, xv, z_hyper, maskf = self._backbone(text, audio, vision, mask)
        # ---- 动态融合头 ----
        s = self._alpha_logits(xt, xa, xv, z_hyper, maskf)              # (B,3)
        s = s.masked_fill(maskf == 0, self._neg)                        # 缺失位 -> -inf
        alpha = torch.softmax(s, dim=1)                                 # (B,3), 缺失位严格 0
        V = torch.stack([self.W_V(xt), self.W_V(xa), self.W_V(xv)], dim=1)   # (B,3,D)
        h_fuse = z_hyper + (alpha.unsqueeze(-1) * V).sum(dim=1)         # (B,D) = z_hyper + Σ α_i V_i
        y = self.predictor(h_fuse).squeeze(-1)                          # (B,)
        if return_rep and return_alpha:
            return y, z_hyper, alpha
        if return_alpha:
            return y, alpha
        if return_rep:
            return y, z_hyper
        return y


class MaskGateFusion(HyperFusionBase):
    """F1 · 简单 Mask Gate: α 来自 MLP([x_T,x_A,x_V,m]) (不使用 z_hyper)。

    对照意义: 与 F3 共享 backbone/V_i/h_fuse, 唯一差异是 α 由"直接读原始投影+mask 的 MLP"给出,
    而非由压缩后的全局情感条件 z_hyper 引导 -> 直接回答 "Hyper 是否比 Mask Gate 更适合做融合条件"。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        D = self.hyper_dim
        self.gate = nn.Sequential(
            nn.Linear(3 * D + self.mask_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._hidden_dim, 3),
        )

    def _alpha_logits(self, xt, xa, xv, z_hyper, maskf):
        h = torch.cat([xt, xa, xv, maskf], dim=-1)      # (B,3D+3), 与 hyper_encoder 同输入
        return self.gate(h)                             # (B,3)


class GlobalQueryFusion(HyperFusionBase):
    """F2 · 普通 Attention: α 来自静态可学习全局 query q_global (样本无关)。

    对照意义: 与 F3 唯一差异是 query 来源 —— F2 用对所有样本相同的可学习向量 q_global,
    F3 用样本级 Q=W_Q z_hyper。α 在 F2 中只随各模态内容 K_i 变化, 不随"该样本的全局情感条件"变化。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        D = self.hyper_dim
        self.q_global = nn.Parameter(torch.empty(D).uniform_(-0.02, 0.02))   # 全局可学习 query
        self.W_K = nn.Linear(D, D)

    def _alpha_logits(self, xt, xa, xv, z_hyper, maskf):
        K = torch.stack([self.W_K(xt), self.W_K(xa), self.W_K(xv)], dim=1)   # (B,3,D)
        return (K * self.q_global.view(1, 1, -1)).sum(dim=-1) / self._scale  # (B,3)


class HyperGuidedFusion(HyperFusionBase):
    """F3 · Hyper-Guided Dynamic Fusion (本项核心创新 2): α 来自样本级 Q=W_Q z_hyper 的 attention。

    s_i = (W_Q z_hyper)·(W_K x_i)^T / √D, 缺失位 -inf, softmax -> α; h_fuse = z_hyper + Σ α_i V_i。
    z_hyper 由封版的 HyperEncoder 产生并受 B3 一致性约束, 作为"统一情感条件"动态决定各可用模态权重。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        D = self.hyper_dim
        self.W_Q = nn.Linear(D, D)
        self.W_K = nn.Linear(D, D)

    def _alpha_logits(self, xt, xa, xv, z_hyper, maskf):
        q = self.W_Q(z_hyper)                                              # (B,D)
        K = torch.stack([self.W_K(xt), self.W_K(xa), self.W_K(xv)], dim=1)  # (B,3,D)
        return (K * q.unsqueeze(1)).sum(dim=-1) / self._scale              # (B,3) = Q·K_i/√D


class HyperGuidedNormFusion(HyperGuidedFusion):
    """F3_norm · Stage 4 Round-2 (A 干预实验, 用户 2026-09-06): 仅对进入 Query 的 z 副本做 L2 归一化。

        Q = W_Q ẑ_hyper,  ẑ_hyper = z_hyper / (‖z_hyper‖₂ + ε)
        融合残差仍用【原始】z_hyper:  h_fuse = z_hyper + Σ α_i V_i   (forward 未改, 只改 _alpha_logits)

    动机 (C 诊断): F3 raw 的 Q=W_Q z 幅度无界 -> logit s 随 ‖z‖ 放大 -> softmax 饱和 one-hot ->
    α 梯度消失 -> ‖z‖ 正反馈爆炸 (onset ep10, peak≈196, 29× 基线)。归一化 query 只修尺度、
    不改各维分布, 是最小单变量干预; 不引入 LayerNorm/MCAC/CCL/KL/MoE/Reliability 等任何新组件。
    与 F3 raw 唯一差异 = query 是否归一化; backbone/V_i/残差/consistency 完全共享。
    """

    _q_eps = 1e-8

    def _alpha_logits(self, xt, xa, xv, z_hyper, maskf):
        zhat = z_hyper / (z_hyper.norm(dim=1, keepdim=True) + self._q_eps)  # 只归一化 query 副本
        return super()._alpha_logits(xt, xa, xv, zhat, maskf)


# 变体注册表: key -> (类, 展示名)。F0 用原生 HyperRepresentationModel (类=None)。
FUSION_VARIANTS = {
    "F0": (None,                 "F0 H0 (z_hyper→Pred)"),
    "F1": (MaskGateFusion,       "F1 MaskGate α=MLP[x,m]"),
    "F2": (GlobalQueryFusion,    "F2 GlobalQuery Attn"),
    "F3": (HyperGuidedFusion,    "F3 Hyper-Guided Fusion"),
    "F3n": (HyperGuidedNormFusion, "F3_norm Hyper-Guided (Q-norm)"),
}
