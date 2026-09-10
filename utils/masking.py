"""
缺失模态置零工具  (阶段2)
--------------------------------------------------
严格按项目说明阶段2第五节: 只做
    x_i' = m_i · x_i
即缺失模态(m_i=0)的时序特征整体置零, 可用模态(m_i=1)保持不变。

【严禁】在此引入 Missing Embedding (x_i' = m_i·x_i + (1-m_i)·e_i),
那属于后续的 Missing-Aware Hyper Representation, 会破坏阶段2消融的干净性。
"""
import torch


def apply_missing_mask(text, audio, vision, mask):
    """
    对 batch 特征按模态 mask 置零缺失模态。

    参数:
        text   : (B, L, 300) float tensor
        audio  : (B, L,   5) float tensor
        vision : (B, L,  20) float tensor
        mask   : (B, 3)      float tensor, [m_T, m_A, m_V], 1=可用 0=缺失
    返回:
        (text', audio', vision')  缺失模态已置零, 形状不变。
    """
    if mask.dim() != 2 or mask.size(1) != 3:
        raise ValueError(f"mask 形状必须为 (B,3), 收到 {tuple(mask.shape)}")
    mask = mask.to(text.dtype)
    # (B,3) -> 每个模态 (B,1,1), 以便广播到 (B,L,Dm)
    m_t = mask[:, 0].view(-1, 1, 1)
    m_a = mask[:, 1].view(-1, 1, 1)
    m_v = mask[:, 2].view(-1, 1, 1)
    return text * m_t, audio * m_a, vision * m_v
