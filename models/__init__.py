"""models 包: Baseline(A) + Mask-aware(B) + Hyper Representation(C/H0) + 阶段3B 修正(H1/H2) + Mask-only 对照。"""
from .baseline import FullModalBaseline
from .mask_baseline import MaskAwareBaseline
from .hyper_representation import (HyperRepresentationModel,
                                   HyperAvailablePoolingModel,
                                   HyperAvailablePoolingResModel)
from .mask_only import MaskOnlyModel
from .stage4_fusion import (HyperFusionBase, MaskGateFusion,
                            GlobalQueryFusion, HyperGuidedFusion,
                            HyperGuidedNormFusion, FUSION_VARIANTS)

__all__ = ["FullModalBaseline", "MaskAwareBaseline",
           "HyperRepresentationModel", "HyperAvailablePoolingModel",
           "HyperAvailablePoolingResModel", "MaskOnlyModel",
           "HyperFusionBase", "MaskGateFusion", "GlobalQueryFusion",
           "HyperGuidedFusion", "HyperGuidedNormFusion", "FUSION_VARIANTS"]
