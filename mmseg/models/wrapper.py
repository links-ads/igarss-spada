from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.parallel import MMDistributedDataParallel

from mmseg.models.builder import build_segmentor
from mmseg.models.segmentors import BaseSegmentor


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.
    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.
    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


def _copy_channel(layer: nn.Module,
                  channel: int = 0,
                  num_channels: int = 1) -> torch.Tensor:
    """Extracts weights from the given layer, it extracts the weights at channel `channel`,
    and it attaches it at the end.
    Args:
        layer (nn.Module): torch layer with weight params
        channel (int, optional): index of the channel to duplicate. Defaults to 0.
    Returns:
        torch.Tensor: layer weights, expanded
    """
    input_weights = layer.weight
    extra_weights = input_weights[:, channel].unsqueeze(
        dim=1)  # make it  [64, 1, 7, 7]
    if (num_channels > 1):
        extra_weights = extra_weights.repeat(1, num_channels, 1, 1)
    return torch.cat((input_weights, extra_weights),
                     dim=1)  # obtain  [64, 4, 7, 7]


def expand_input(model: nn.Module,
                 input_layer: str = None,
                 copy_channel: int = 0,
                 num_channels: int = 1) -> nn.Module:
    """Recursively iterates the layers of the model, until the input layer is retrieved.
    Last, it expands the layer weights to accomodate for extra channels.
    Args:
        model (nn.Module): torch model
        input_layer (str, optional): name of input, if known a priori. Defaults to None.
        copy_channel (int, optional): which channel to copy. Defaults to 0.
    Returns:
        nn.Module: the same model, whose input has been extended
    """
    # when we know the layer name
    if input_layer is not None:
        model[input_layer].weight = nn.Parameter(
            _copy_channel(
                model[input_layer],
                channel=copy_channel,
                num_channels=num_channels))
    else:
        children = list(model.children())
        input_layer = children[0]
        while children and len(children) > 0:
            input_layer = children[0]
            children = list(children[0].children())

        assert not list(input_layer.children()
                        ), f"layer '{input_layer}' still has children!"
        input_layer.weight = nn.Parameter(
            _copy_channel(
                input_layer, channel=copy_channel, num_channels=num_channels))

    return model


class SegmentorWrapper(BaseSegmentor):

    def __init__(self, model_config: dict, max_iters: int, resume_iters: int,
                 num_channels: int, work_dir: str, **kwargs):
        super(BaseSegmentor, self).__init__()
        self.max_iters = max_iters
        self.local_iters = resume_iters
        self.num_channels = num_channels
        self.work_dir = work_dir
        self.model = build_segmentor(deepcopy(model_config))
        self.train_cfg = model_config['train_cfg']
        self.test_cfg = model_config['test_cfg']
        head_cfg = model_config['decode_head']
        if isinstance(head_cfg, list):
            head_cfg = head_cfg[0]
        self.num_classes = head_cfg['num_classes']
        # self.adapt_input(num_channels - 3)

    def get_model(self):
        return get_module(self.model)

    def adapt_input(self, num_channels: int = 1):
        """Called by train.py before the training loop.
        Scales the input to the expected size (adds the NIR weights copying from red)
        """
        if self.num_channels > 3:
            self.model = expand_input(
                self.model, copy_channel=0, num_channels=num_channels)

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      gt_lucas=None,
                      seg_weight=None,
                      return_feat=False):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = self.get_model().forward_train(img, img_metas,
                                                gt_semantic_seg, gt_lucas,
                                                seg_weight)
        return losses

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.
        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.
        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.
        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)
