# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: Model construction with loop
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with
    Transformers
    """

    def __init__(self, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        conv_kernel_size = decoder_params['conv_kernel_size']

        self.mlps = []
        for i, in_channels in zip(self.in_index, self.in_channels):
            self.mlps.append(MLP(input_dim=in_channels, embed_dim=embedding_dim))
        self.mlps = nn.ModuleList(self.mlps)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * len(self.in_index),
                                      out_channels=embedding_dim,
                                      kernel_size=conv_kernel_size,
                                      padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
                                      norm_cfg=kwargs['norm_cfg'])

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, return_feat=False):
        batch_size = inputs[-1].shape[0]
        # size of each one after MLP: [batch, 768, 128, 128]
        embeddings = []
        for i in self.in_index:
            embedding = self.mlps[i](inputs[i]).permute(0, 2, 1).contiguous()
            embedding = embedding.reshape(batch_size, -1, inputs[i].shape[2], inputs[i].shape[3])
            if i != 0:
                embedding = resize(embedding, size=inputs[0].size()[2:], mode='bilinear', align_corners=False)
            embeddings.append(embedding)
        embeddings = self.linear_fuse(torch.cat(embeddings, dim=1))

        features = embeddings
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        out = self.linear_pred(embeddings)

        if return_feat:
            return out, features
        return out