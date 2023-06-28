from copy import deepcopy
from typing import List

import numpy as np
import torch
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.models.builder import MODELS
from mmseg.models.custom import CustomModel
from mmseg.models.wrapper import get_module


@MODELS.register_module()
class CustomDACS(CustomModel):

    def __init__(self, model_config: dict, **kwargs):
        self.dacs = model_config.pop('dacs', None)
        assert self.dacs is not None, "DACS configuration missing"
        super(CustomDACS, self).__init__(model_config, **kwargs)
        self.alpha = self.dacs['alpha']
        self.pseudo_threshold = self.dacs['pseudo_threshold']
        self.local_iter = 0

        self.ema_model = deepcopy(self.model)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def adapt_input(self, num_channels: int = 1):
        super().adapt_input(num_channels=num_channels)
        self.ema_model = deepcopy(self.model)

    def forward_train(self,
                      img: torch.Tensor,
                      img_metas: List[dict],
                      gt_semantic_seg: torch.Tensor,
                      gt_lucas: torch.Tensor = None,
                      seg_weight: torch.Tensor = None):

        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()

        if self.local_iter > 0:
            self._update_ema(self.local_iter)

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits = self.get_ema_model().encode_decode(img, img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size

        # Scribble and pseudo labels mixing
        pseudo_label[~ps_large_p] = 255
        valid_mask = gt_semantic_seg != 255
        gt_semantic_seg = gt_semantic_seg.long()
        pseudo_label[valid_mask] = gt_semantic_seg[valid_mask]
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)
        pseudo_weight[valid_mask] = 1

        losses = super().forward_train(
            img,
            img_metas,
            pseudo_label,
            gt_lucas=gt_lucas,
            seg_weight=pseudo_weight)

        self.local_iter += 1

        return losses
