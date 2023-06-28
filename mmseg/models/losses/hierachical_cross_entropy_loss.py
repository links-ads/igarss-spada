# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    if class_weight is not None:
        class_weight = pred.new_tensor(class_weight)
    else:
        class_weight = None

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def clustered_cross_entropy(pred,
                            label,
                            weight=None,
                            class_weight=None,
                            reduction='mean',
                            avg_factor=None,
                            ignore_index=-100,
                            avg_non_ignore=False,
                            clusters=None):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    assert clusters
    new_pred = torch.empty(pred.shape[0], len(clusters), pred.shape[2],
                           pred.shape[3]).to(pred.device)
    new_label = label.clone()

    for i, cluster in enumerate(clusters):
        new_pred[:, i] = torch.sum(pred[:, cluster[0]:cluster[1] + 1], dim=1)
        new_label[torch.logical_and(label >= cluster[0],
                                    label <= cluster[1])] = i

    loss = cross_entropy(
        new_pred,
        new_label,
        weight,
        class_weight,
        reduction=reduction,
        avg_factor=avg_factor,
        avg_non_ignore=avg_non_ignore,
        ignore_index=ignore_index)

    return loss


def hierarchical_cross_entropy(pred,
                               label,
                               weight=None,
                               class_weight=None,
                               reduction='mean',
                               avg_factor=None,
                               ignore_index=-100,
                               avg_non_ignore=False,
                               hierarchy=None):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    first_level_loss = clustered_cross_entropy(
        pred,
        label,
        weight,
        class_weight=class_weight['first_level'],
        reduction=reduction,
        avg_factor=avg_factor,
        avg_non_ignore=avg_non_ignore,
        ignore_index=ignore_index,
        clusters=hierarchy['first_level'])

    second_level_loss = clustered_cross_entropy(
        pred,
        label,
        weight,
        class_weight=class_weight['second_level'],
        reduction=reduction,
        avg_factor=avg_factor,
        avg_non_ignore=avg_non_ignore,
        ignore_index=ignore_index,
        clusters=hierarchy['second_level'])

    third_level_loss = cross_entropy(
        pred,
        label,
        weight,
        class_weight=class_weight['third_level'],
        reduction=reduction,
        avg_factor=avg_factor,
        avg_non_ignore=avg_non_ignore,
        ignore_index=ignore_index)

    loss = first_level_loss + second_level_loss + third_level_loss

    return loss


@LOSSES.register_module()
class HierarchicalCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_hier_ce',
                 avg_non_ignore=False,
                 hierarchy_path=None,
                 class_weight_path=None):
        super(HierarchicalCrossEntropyLoss, self).__init__()
        assert hierarchy_path
        self.reduction = reduction
        self.loss_weight = loss_weight
        # self.class_weight = get_class_weight(class_weight)
        with open(class_weight_path) as class_weight_file:
            self.class_weight = json.load(class_weight_file)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        with open(hierarchy_path) as hierarchy_file:
            self.hierarchy = json.load(hierarchy_file)
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # if self.class_weight is not None:
        #     class_weight = cls_score.new_tensor(self.class_weight)
        # else:
        #     class_weight = None
        # Note: for BCE loss, label < 0 is invalid.

        loss = hierarchical_cross_entropy(
            cls_score,
            label,
            weight,
            class_weight=self.class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            hierarchy=self.hierarchy)

        loss_cls = self.loss_weight * loss
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
