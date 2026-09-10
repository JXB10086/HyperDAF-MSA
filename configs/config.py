"""
第一阶段配置  (Baseline · T+A+V 完整模态)
--------------------------------------------------
所有维度均来自实测数据, 禁止随意更改 (见项目说明第十四节)。
训练目标: 确认 loss 能下降, 记录 MAE / Corr。
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # ---------- 数据 ----------
    data_path = os.path.join(ROOT, "data", "mosi", "mosi_data.pkl")
    # 特征维度 (实测: text=300, audio=5, vision=20, seq_len=50) —— 不可改
    text_dim = 300
    audio_dim = 5
    vision_dim = 20
    seq_len = 50

    # ---------- 模型 ----------
    proj_dim = 128      # 统一投影维度 D (项目说明: 128 或 256)
    hidden_dim = 256    # MLP 隐层
    dropout = 0.1

    # ---------- 训练 ----------
    batch_size = 32
    lr = 1e-3
    weight_decay = 0.0
    epochs = 30
    seed = 42
    num_workers = 0     # Windows 下小数据用 0, 避免多进程开销

    # ---------- 输出 ----------
    exp_dir = os.path.join(ROOT, "experiments")
    ckpt_name = "baseline_full.pt"

    # ---------- 阶段2: 模态缺失实验 ----------
    # Protocol B 训练时每个模态的随机缺失概率 (每样本独立, 至少保留一个模态)
    protocol_b_train_missing_prob = 0.5
    # 随机缺失测试的 p 列表 (阶段2第八节)
    random_test_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    # 阶段2结果输出目录 (表格 / 曲线 / checkpoint)
    stage2_dir = os.path.join(ROOT, "experiments", "stage2")

    # ---------- 阶段3: Missing-Aware Hyper Representation ----------
    # z_hyper 维度 D (同时也是三模态统一投影维度), 默认 256 (用户 3.1)
    hyper_dim = 256
    # HyperEncoder / Predictor 内部隐层宽度 (最小可行, 保持与 baseline 同量级)
    hyper_hidden_dim = 256
    # Hyper Dimension Ablation (用户 3.6): 看性能是否随 D 稳定
    hyper_dim_ablation = [64, 128, 256, 512]
    # 阶段3结果输出目录 (表格 / 曲线 / z_hyper / checkpoint)
    stage3_dir = os.path.join(ROOT, "experiments", "stage3")

    # ---------- 阶段3B: Hyper Representation 修正实验 (D->B1->B2) ----------
    # H0=原始Hyper / H1=可用模态池化 / H2=池化+LayerNorm+Residual
    stage3b_dir = os.path.join(ROOT, "experiments", "stage3b")

    # ---------- MOSEI 跨数据集验证 ----------
    # 处理后的 CMU-MOSEI pkl (train/valid/test × text/audio/vision/labels/id)
    # 实测: train 16265 / valid 1869 / test 4643; text=300 audio=74 vision=35 L=50
    mosei_data_path = os.path.join(ROOT, "data", "mosei", "mosei_senti_data.pkl")
    # inspect_mosei.py 产出的 float32 .npy (mmap 后端, 避免载入 3.5GB pkl)
    mosei_conv_dir = os.path.join(ROOT, "data", "mosei", "converted")
    # MOSEI 实验输出目录 (表格 / 诊断 / z_hyper / checkpoint / 实验记录)
    mosei_dir = os.path.join(ROOT, "experiments", "mosei")
    # audio 第 7 列存在 -Inf (train 1249 / valid 191 / test 425, 全在有效帧) 的处置策略:
    #   用户裁决 = "train_p01" (train 有效帧该列有限值的 0.1% 分位数 = -0.536847)
    #   理由: train_min=-8.97 是约 -95σ 孤立离群点, 0 在 +3.43σ, 取有界低分位数最稳;
    #         统计量只在 train 有效帧上算, valid/test 不参与 (无泄漏)。
    #   可选: "none"(检测到即报错) | "zero" | "train_min" | "train_median" | "train_p01"
    mosei_inf_policy = "train_p01"
    # 注: MOSEI 原始特征维度与 MOSI 不同 (audio 74 vs 5, vision 35 vs 20), 这是允许的;
    #     统一发生在模型 Proj_i -> D=256, 数据层不强行对齐维度。

    @property
    def ckpt_path(self):
        return os.path.join(self.exp_dir, self.ckpt_name)
