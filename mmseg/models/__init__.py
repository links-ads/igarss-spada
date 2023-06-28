# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, build_backbone,
                      build_head, build_loss, build_segmentor)
from .custom import CustomModel
from .custom_aux_conv import CustomAuxConv
from .custom_dacs import CustomDACS
from .decode_heads import *  # noqa: F401,F403
from .decode_heads.legacy import SegFormerHead
from .kernel_custom import KernelCustomModel
from .logits_custom import LogitsCustomModel
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .wrapper import SegmentorWrapper

__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'build_backbone',
    'build_head', 'build_loss', 'build_segmentor', 'CustomModel',
    'LogitsCustomModel', 'KernelCustomModel', 'CustomAuxConv', 'CustomDACS',
    'SegmentorWrapper', 'SegFormerHead'
]
