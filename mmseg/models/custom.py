from typing import List

import torch
import torch.nn.functional as F

from mmseg.models.builder import MODELS
from mmseg.models.wrapper import SegmentorWrapper
from scripts.tiling import _spline_2d


@MODELS.register_module()
class CustomModel(SegmentorWrapper):

    def __init__(self,
                 model_config: dict,
                 max_iters: int,
                 resume_iters: int,
                 num_channels: int = 3,
                 aug: dict = None,
                 **kwargs):
        super().__init__(model_config, max_iters, resume_iters, num_channels,
                         **kwargs)
        self.num_channels = num_channels

    def forward_train(self,
                      img: torch.Tensor,
                      img_metas: List[dict],
                      gt_semantic_seg: torch.Tensor,
                      gt_lucas: torch.Tensor = None,
                      seg_weight: torch.Tensor = None):
        # mmcv.print_log(f"iter: {self.local_iter}")
        # mmcv.print_log(f"input shape: {img.shape}")
        # mmcv.print_log(f"label shape: {gt_semantic_seg.shape}")

        # Forward on the (possibly augmented) batch with standard segmentation
        if gt_lucas is not None:
            gt_lucas = gt_lucas.long()
        losses = super().forward_train(
            img,
            img_metas,
            gt_semantic_seg.unsqueeze(1).long(),
            gt_lucas=gt_lucas,
            seg_weight=seg_weight)

        return losses

    def augment_crop(self, crop_img: torch.Tensor):
        crop_img = crop_img.squeeze(0)
        crop_img_90 = torch.rot90(crop_img, k=1, dims=[1, 2])
        crop_img_180 = torch.rot90(crop_img, k=2, dims=[1, 2])
        crop_img_270 = torch.rot90(crop_img, k=3, dims=[1, 2])
        crop_img_hflip = torch.flip(crop_img, dims=(2, ))
        crop_img_vflip = torch.flip(crop_img, dims=(1, ))
        return torch.stack((crop_img, crop_img_90, crop_img_180, crop_img_270,
                            crop_img_hflip, crop_img_vflip),
                           dim=0)

    def fuse_aug(self, out: torch.Tensor):
        out[1] = torch.rot90(out[1], k=3, dims=[1, 2])
        out[2] = torch.rot90(out[2], k=2, dims=[1, 2])
        out[3] = torch.rot90(out[3], k=1, dims=[1, 2])
        out[4] = torch.flip(out[4], dims=(2, ))
        out[5] = torch.flip(out[5], dims=(1, ))

        return out.mean(dim=0)

    def chunk_crop(self, crop_img: torch.Tensor, splits: int = 2):
        chunks = torch.stack(torch.chunk(crop_img, splits, dim=-2), dim=0)
        chunks = torch.cat(torch.chunk(chunks, splits, dim=-1), dim=0)
        return F.upsample(chunks, scale_factor=splits, mode='bilinear')

    def fuse_chunks(self, chunks: torch.Tensor, splits: int = 2):
        fused = F.upsample(chunks, scale_factor=1 / splits, mode='bilinear')
        fused = torch.cat(torch.chunk(fused, splits, dim=0), dim=-1)
        fused = torch.cat(torch.chunk(fused, splits, dim=0), dim=-2)
        return fused.squeeze(dim=0)

    def weighted_slide_inference(self, img, img_meta):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        # Spline
        spline = torch.Tensor(
            _spline_2d(window_size=h_crop, power=2).squeeze(-1))

        # Chunks splits
        splits = 2
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # Augmentations batch
                crop_img = self.augment_crop(crop_img)
                # Chunks
                crop_chunks = self.chunk_crop(crop_img[0], splits=splits)
                crop_img = torch.cat((crop_img, crop_chunks), dim=0)

                out = self.encode_decode(crop_img, img_meta)

                # Augmentations fusion
                fused_augs = self.fuse_aug(out[:-splits**2])
                # Chunks fusion
                fused_chunks = self.fuse_chunks(
                    out[-splits**2:], splits=splits)
                # Final fusion
                fused_out = torch.stack((fused_augs, fused_chunks), dim=0)
                out = fused_out.mean(dim=0)

                if isinstance(out, tuple):
                    crop_seg_logit = out[0]
                else:
                    crop_seg_logit = out
                # Weighting
                crop_seg_logit *= spline.to(crop_seg_logit.device)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        return preds

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

        assert self.test_cfg.mode in ['slide', 'whole', 'weighted_slide']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = super().get_model().slide_inference(
                img, img_meta, rescale)
        elif self.test_cfg.mode == 'weighted_slide':
            seg_logit = self.weighted_slide_inference(img, img_meta)
        else:
            seg_logit = super().get_model().whole_inference(
                img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
