"""
Missing Mask Simulator  (第一阶段 · Step 9)
--------------------------------------------------
独立函数 generate_missing_mask, 生成模态缺失掩码:
    m = [m_T, m_A, m_V],   1 = 模态可用,  0 = 模态缺失
    (顺序固定为 text, audio, vision)

支持两种模式 (对应项目说明第十一节 & 第二十三节):

  1) 固定模式 (missing_pattern):
       "T+A+V" -> [1,1,1]      完整
       "T+A"   -> [1,1,0]      缺 vision
       "T+V"   -> [1,0,1]      缺 audio
       "A+V"   -> [0,1,1]      缺 text
       "T"     -> [1,0,0]      仅 text
       "A"     -> [0,1,0]      仅 audio
       "V"     -> [0,0,1]      仅 vision
       "none"  -> [0,0,0]      全部缺失 (极端, 单独报告)
     也可直接传长度3的 0/1 序列。整个 batch 使用同一 mask。

  2) 随机模式 (missing_probability = p):
       每个样本、每个模态独立地以概率 p 缺失。
       allow_all_missing=False(默认) 时, 保证每个样本【至少一个模态可用】,
       即输出限定在第二十三节列举的 7 种非全缺组合内;
       全缺 [0,0,0] 作为极端情况由固定模式 "none" 单独报告。

注意: 本模块只负责【生成 mask】, 不负责把 mask 应用到特征
      (Mask-aware pooling / 融合属于后续阶段)。
"""
import numpy as np

# mask 顺序: [text, audio, vision]
MODALITIES = ("text", "audio", "vision")

# 固定缺失模式 -> (m_T, m_A, m_V); key 统一大写匹配
PATTERN_MAP = {
    "T+A+V": (1, 1, 1),
    "T+A": (1, 1, 0),
    "T+V": (1, 0, 1),
    "A+V": (0, 1, 1),
    "T": (1, 0, 0),
    "A": (0, 1, 0),
    "V": (0, 0, 1),
    "NONE": (0, 0, 0),
}


def _resolve_pattern(pattern):
    """把 missing_pattern 解析为长度3的 (m_T, m_A, m_V) 元组。"""
    if isinstance(pattern, str):
        key = pattern.strip().upper().replace(" ", "")
        if key in ("", "NONE", "ALL_MISSING", "∅"):
            return (0, 0, 0)
        if key not in PATTERN_MAP:
            raise ValueError(
                f"未知 missing_pattern='{pattern}', 可选: "
                f"{['T+A+V','T+A','T+V','A+V','T','A','V','none']} 或长度3的0/1序列"
            )
        return PATTERN_MAP[key]
    # 序列形式
    seq = tuple(int(x) for x in pattern)
    if len(seq) != 3 or any(x not in (0, 1) for x in seq):
        raise ValueError(f"missing_pattern 序列必须为长度3的0/1, 收到: {pattern}")
    return seq


def generate_missing_mask(batch_size, missing_probability=None, missing_pattern=None,
                          allow_all_missing=False, seed=None):
    """
    生成模态缺失掩码。

    参数:
        batch_size (int): 样本数, 决定输出第0维。
        missing_probability (float|None): 随机模式下每个模态的缺失概率 p∈[0,1]。
        missing_pattern (str|sequence|None): 固定模式, 优先级高于随机模式。
        allow_all_missing (bool): 随机模式下是否允许出现全缺 [0,0,0]; 默认 False。
        seed (int|None): 随机种子, 保证可复现。

    返回:
        np.ndarray, shape=(batch_size, 3), dtype=int64, 每行 [m_T, m_A, m_V]。

    说明:
        - missing_pattern 与 missing_probability 都为 None 时, 返回完整模态 [1,1,1]。
        - 两者都提供时, 以 missing_pattern(固定模式) 为准。
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数, 收到: {batch_size}")
    rng = np.random.default_rng(seed)

    # ---- 模式1: 固定 pattern (优先) ----
    if missing_pattern is not None:
        m = _resolve_pattern(missing_pattern)
        return np.tile(np.asarray(m, dtype=np.int64), (batch_size, 1))

    # ---- 模式2: 随机缺失 ----
    if missing_probability is not None:
        p = float(missing_probability)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"missing_probability 必须在 [0,1], 收到: {p}")
        # 每个位置以概率 p 缺失(0), 以 1-p 可用(1)
        mask = (rng.random((batch_size, 3)) >= p).astype(np.int64)
        if not allow_all_missing:
            all_missing = mask.sum(axis=1) == 0
            n_fix = int(all_missing.sum())
            if n_fix > 0:
                # 为每个全缺样本随机保留一个模态, 保证至少一个可用
                idxs = np.where(all_missing)[0]
                keep_modality = rng.integers(0, 3, size=n_fix)
                mask[idxs, keep_modality] = 1
        return mask

    # ---- 默认: 完整模态 ----
    return np.ones((batch_size, 3), dtype=np.int64)
