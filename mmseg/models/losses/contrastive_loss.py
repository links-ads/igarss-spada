# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def compute_ema(ema, new, alpha):
    new = new.detach()
    return alpha * new + (1 - alpha) * ema


@LOSSES.register_module()
class ConstrastiveIntraViewLucasLoss(nn.Module):
    """Lucas Intra-view Contrastive Loss.
    """

    def __init__(self,
                 loss_name='loss_civ_lucas',
                 num_classes=9,
                 temperature=0.1,
                 alpha=0.99,
                 use_sigmoid=False,
                 use_mask=False,
                 loss_weight=None,
                 feature_dim=768):
        super(ConstrastiveIntraViewLucasLoss, self).__init__()
        self._loss_name = loss_name
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = torch.rand((num_classes, feature_dim),
                                     device=torch.device('cuda'))
        self.alpha = alpha

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                gt_lucas=None,
                **kwargs):
        """Forward function."""

        assert "features" in kwargs
        features = kwargs.pop("features")

        # features devono essere della stessa shape dell'immagine

        # Prototypes: select from gt_lucas the pixels of the class c
        # label: BS x 512 x 512
        # features: BS x 768 x 128 x 128

        # rescale gt from 512x512 to 128x128
        downsampled_gt_lucas = gt_lucas[:, ::4, ::4]
        mask_lucas_pts = downsampled_gt_lucas != 255
        labels_lucas_pts = downsampled_gt_lucas[mask_lucas_pts]
        features_lucas_pts = torch.permute(features,
                                           (0, 2, 3, 1))[mask_lucas_pts]

        for id, c in enumerate(torch.unique(labels_lucas_pts)):
            self.prototypes[id] = compute_ema(
                self.prototypes[id],
                features_lucas_pts[labels_lucas_pts == c].mean(dim=0),
                self.alpha)

        self.prototypes = F.normalize(self.prototypes)

        prot_feat = F.normalize(
            torch.matmul(self.prototypes, features_lucas_pts.transpose(0, 1)),
            dim=-1) / self.temperature

        A1 = torch.exp(prot_feat)[labels_lucas_pts,
                                  torch.arange(len(labels_lucas_pts))]
        A2 = torch.exp(prot_feat).sum(dim=0)
        loss = torch.mean(-1 * torch.log(A1 / A2))

        return loss


@LOSSES.register_module()
class ConstrastiveCrossViewLucasVSCorineLoss(nn.Module):
    """Lucas vs Corine Cross-view Contrastive Loss.
    """

    def __init__(self,
                 loss_name='loss_ccv_lucas_corine',
                 num_classes=9,
                 temperature=0.1,
                 use_sigmoid=False,
                 use_mask=False,
                 loss_weight=None,
                 alpha=0.99,
                 feature_dim=768):
        super(ConstrastiveCrossViewLucasVSCorineLoss, self).__init__()
        self._loss_name = loss_name
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.prototypes = torch.rand((num_classes, feature_dim),
                                     device=torch.device('cuda'))

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                gt_lucas=None,
                **kwargs):
        """Forward function."""

        assert "features" in kwargs
        features = kwargs.pop("features")

        # features devono essere della stessa shape dell'immagine

        # Prototypes: select from gt_lucas the pixels of the class c
        # label: BS x 512 x 512
        # features: BS x 768 x 128 x 128

        downsampled_gt_corine = label[:, ::4, ::4]
        downsampled_gt_lucas = gt_lucas[:, ::4, ::4]

        mask_corine_pts = downsampled_gt_corine != 255
        mask_lucas_pts = downsampled_gt_lucas != 255

        labels_corine_pts = downsampled_gt_corine[mask_corine_pts]
        labels_lucas_pts = downsampled_gt_corine[mask_lucas_pts]
        labels_corine_pts[labels_corine_pts == 7] = 6

        features_corine_pts = torch.permute(features,
                                            (0, 2, 3, 1))[mask_corine_pts]

        features_lucas_pts = torch.permute(features,
                                           (0, 2, 3, 1))[mask_lucas_pts]

        for id, c in enumerate(torch.unique(labels_lucas_pts)):
            self.prototypes[id] = compute_ema(
                self.prototypes[id],
                features_lucas_pts[labels_lucas_pts == c].mean(dim=0),
                self.alpha)

        self.prototypes = F.normalize(self.prototypes)
        prot_feat = F.normalize(
            torch.matmul(self.prototypes, features_corine_pts.transpose(0, 1)),
            dim=-1) / self.temperature
        prot_feat[2] = labels_corine_pts == 2
        A1 = torch.exp(prot_feat)[labels_corine_pts,
                                  torch.arange(len(labels_corine_pts))]
        A2 = torch.exp(prot_feat).sum(dim=0)
        loss = torch.mean(-1 * torch.log(A1 / A2))

        return loss


@LOSSES.register_module()
class ConstrastiveIntraViewCorineVSCorineLoss(nn.Module):
    """Corine vs Corine Intra-view Contrastive Loss.
    """

    def __init__(self,
                 loss_name='loss_iv_corine_corine',
                 num_classes=9,
                 temperature=0.1,
                 use_sigmoid=False,
                 use_mask=False,
                 loss_weight=None,
                 alpha=0.99,
                 feature_dim=768):
        super(ConstrastiveIntraViewCorineVSCorineLoss, self).__init__()
        self._loss_name = loss_name
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.prototypes = torch.rand((num_classes, feature_dim),
                                     device=torch.device('cuda'))

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                gt_lucas=None,
                **kwargs):
        """Forward function."""

        assert "features" in kwargs
        features = kwargs.pop("features")

        # features devono essere della stessa shape dell'immagine

        # Prototypes: select from gt_lucas the pixels of the class c
        # label: BS x 512 x 512
        # features: BS x 768 x 128 x 128

        downsampled_gt_corine = label[:, ::4, ::4]

        mask_corine_pts = downsampled_gt_corine != 255

        labels_corine_pts = downsampled_gt_corine[mask_corine_pts]

        features_corine_pts = torch.permute(features,
                                            (0, 2, 3, 1))[mask_corine_pts]

        for id, c in enumerate(torch.unique(labels_corine_pts)):
            self.prototypes[id] = compute_ema(
                self.prototypes[id],
                features_corine_pts[labels_corine_pts == c].mean(dim=0),
                self.alpha)

        self.prototypes = F.normalize(self.prototypes)
        prot_feat = F.normalize(
            torch.matmul(self.prototypes, features_corine_pts.transpose(0, 1)),
            dim=-1) / self.temperature

        A1 = torch.exp(prot_feat)[labels_corine_pts,
                                  torch.arange(len(labels_corine_pts))]
        A2 = torch.exp(prot_feat).sum(dim=0)
        loss = torch.mean(-1 * torch.log(A1 / A2))

        return loss
