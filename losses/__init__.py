"""
losses 包  (第一阶段: 占位)
--------------------------------------------------
第一阶段 Baseline 直接使用 PyTorch 内置 nn.L1Loss (MAE), 无自定义损失。

后续阶段的【可选】损失将放在此处, 但必须遵循项目说明第九节:
  - Label-Aware Contrastive Loss (L_ccl): 连续标签按 |y_i-y_j|<τ 构造正样本
  - Latent Prior (L_prior): KL(N(μ,σ²) ‖ N(0,I)) 辅助正则
  两者均以【消融实验结果】决定是否保留, 第一阶段不实现。
"""
