# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss


@LOSSES.register_module()
class LucasCrossEntropyLoss(CrossEntropyLoss):
    """LucasCrossEntropyLoss.
    """

    def __init__(self,
                 loss_name='loss_ce_lucas',
                 forests_disambiguation=True,
                 **kwargs):
        super().__init__(loss_name=loss_name, **kwargs)
        self.forests_disambiguation = forests_disambiguation

    def forward(self, cls_score, label, gt_lucas, weight, **kwargs):
        """Forward function."""

        agg_cls_score = cls_score.clone()

        # Coniferous and Broadleaves aggregation
        if not self.forests_disambiguation:
            agg_cls_score[:, 6] += agg_cls_score[:, 7]
            agg_cls_score[:, 7] = 0
        loss_cls = super().forward(agg_cls_score, gt_lucas, **kwargs)
        return loss_cls
