import numpy as np
from albumentations import Compose
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations.augmentations.geometric.rotate import (RandomRotate90,
                                                           Rotate)
from albumentations.augmentations.geometric.transforms import (Affine,
                                                               Perspective,
                                                               ShiftScaleRotate
                                                               )
from albumentations.augmentations.transforms import (ChannelShuffle,
                                                     ColorJitter, GaussianBlur,
                                                     HorizontalFlip,
                                                     RandomBrightnessContrast,
                                                     VerticalFlip)
from cv2 import BORDER_REFLECT_101

from ..builder import PIPELINES


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@PIPELINES.register_module()
class ReplaceNan:

    def __call__(self, results):
        results['img'] = np.nan_to_num(results['img'])
        return results


@PIPELINES.register_module()
class Divide10k:

    def __call__(self, results):
        results['img'] /= 10000
        return results


@PIPELINES.register_module()
class Multiply10K:

    def __call__(self, results):
        results['img'] = results['img'] * 10000
        return results


@PIPELINES.register_module()
class TiffHorizontalFlip:

    def __init__(self, prob, enable_lucas=False):
        if enable_lucas:
            self.transform = Compose([HorizontalFlip(p=prob)],
                                     additional_targets={'gt_lucas': 'mask'})
        else:
            self.transform = HorizontalFlip(p=prob)

        self.enable_lucas = enable_lucas

    def __call__(self, results):
        if self.enable_lucas:
            aug = self.transform(
                image=results['img'],
                mask=results['gt_semantic_seg'],
                gt_lucas=results['gt_lucas'])

            results['img'] = aug.get('image')
            results['gt_semantic_seg'] = aug.get('mask')
            results['gt_lucas'] = aug.get('gt_lucas')
        else:
            aug = self.transform(
                image=results['img'], mask=results['gt_semantic_seg'])
            results['img'] = aug.get('image')
            results['gt_semantic_seg'] = aug.get('mask')
        return results


@PIPELINES.register_module()
class TiffVerticalFlip:

    def __init__(self, prob, enable_lucas=False):
        if enable_lucas:
            self.transform = Compose([VerticalFlip(p=prob)],
                                     additional_targets={'gt_lucas': 'mask'})
        else:
            self.transform = VerticalFlip(p=prob)

        self.enable_lucas = enable_lucas

    def __call__(self, results):
        if self.enable_lucas:
            aug = self.transform(
                image=results['img'],
                mask=results['gt_semantic_seg'],
                gt_lucas=results['gt_lucas'])

            results['img'] = aug.get('image')
            results['gt_semantic_seg'] = aug.get('mask')
            results['gt_lucas'] = aug.get('gt_lucas')
        else:
            aug = self.transform(
                image=results['img'], mask=results['gt_semantic_seg'])
            results['img'] = aug.get('image')
            results['gt_semantic_seg'] = aug.get('mask')
        return results


@PIPELINES.register_module()
class TiffRandomRotate90:

    def __init__(self, prob):
        self.transform = RandomRotate90(p=prob)

    def __call__(self, results):
        pair = self.transform(
            image=results['img'], mask=results['gt_semantic_seg'])
        results['img'] = pair.get('image')
        results['gt_semantic_seg'] = pair.get('mask')
        return results


@PIPELINES.register_module()
class TiffPerspective:

    def __init__(self, dist, prob):
        self.transform = Perspective(scale=dist, p=prob)

    def __call__(self, results):
        pair = self.transform(
            image=results['img'], mask=results['gt_semantic_seg'])
        results['img'] = pair.get('image')
        results['gt_semantic_seg'] = pair.get('mask')
        return results


@PIPELINES.register_module()
class TiffRandomRotate:

    def __init__(self, prob):
        self.transform = Rotate(p=prob)

    def __call__(self, results):
        pair = self.transform(
            image=results['img'], mask=results['gt_semantic_seg'])
        results['img'] = pair.get('image')
        results['gt_semantic_seg'] = pair.get('mask')
        return results


@PIPELINES.register_module()
class TiffGaussianBlur:

    def __init__(self, blur_limit, sigma_limit, prob):
        self.transform = GaussianBlur(
            blur_limit=blur_limit, sigma_limit=sigma_limit, p=prob)

    def __call__(self, results):
        pair = self.transform(
            image=results['img'], mask=results['gt_semantic_seg'])
        results['img'] = pair.get('image')
        results['gt_semantic_seg'] = pair.get('mask')
        return results


@PIPELINES.register_module()
class TiffRandomCrop:

    def __init__(self, img_shape, scale, prob):
        self.transform = RandomResizedCrop(
            height=img_shape[0], width=img_shape[1], scale=scale, p=prob)

    def __call__(self, results):
        pair = self.transform(
            image=results['img'], mask=results['gt_semantic_seg'])
        results['img'] = pair.get('image')
        results['gt_semantic_seg'] = pair.get('mask')
        return results


@PIPELINES.register_module()
class TiffColorJitter:

    def __init__(self, brightness, contrast, saturation, hue, prob):
        self.transform = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=prob)

    def __call__(self, results):
        pair = self.transform(
            image=results['img'], mask=results['gt_semantic_seg'])
        results['img'] = pair.get('image')
        results['gt_semantic_seg'] = pair.get('mask')
        return results


@PIPELINES.register_module()
class TiffRandomBrightnessContrast:

    def __init__(self, brightness_limit, contrast_limit, prob):
        self.transform = RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            brightness_by_max=False,
            p=prob)

    def __call__(self, results):
        pair = self.transform(
            image=results['img'], mask=results['gt_semantic_seg'])
        results['img'] = pair.get('image')
        results['gt_semantic_seg'] = pair.get('mask')
        return results


@PIPELINES.register_module()
class TiffAffine:

    def __init__(self, scale, translate_percent, rotate, shear, prob):
        self.transform = Affine(
            scale=scale,
            translate_percent=translate_percent,
            rotate=rotate,
            shear=shear,
            p=prob,
            mode=BORDER_REFLECT_101)

    def __call__(self, results):
        pair = self.transform(
            image=results['img'], mask=results['gt_semantic_seg'])
        results['img'] = pair.get('image')
        results['gt_semantic_seg'] = pair.get('mask')
        return results


@PIPELINES.register_module()
class TiffShiftScaleRotate:

    def __init__(self,
                 shift_limit,
                 scale_limit,
                 rotate_limit,
                 prob,
                 enable_lucas=False):

        if enable_lucas:
            self.transform = Compose([
                ShiftScaleRotate(
                    shift_limit=shift_limit,
                    scale_limit=scale_limit,
                    rotate_limit=rotate_limit,
                    border_mode=BORDER_REFLECT_101,
                    p=prob)
            ],
                                     additional_targets={'gt_lucas': 'mask'})
        else:
            self.transform = ShiftScaleRotate(
                shift_limit=shift_limit,
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                border_mode=BORDER_REFLECT_101,
                p=prob)

        self.enable_lucas = enable_lucas

    def __call__(self, results):
        if self.enable_lucas:
            aug = self.transform(
                image=results['img'],
                mask=results['gt_semantic_seg'],
                gt_lucas=results['gt_lucas'])
            results['img'] = aug.get('image')
            results['gt_semantic_seg'] = aug.get('mask')
            results['gt_lucas'] = aug.get('gt_lucas')
        else:
            aug = self.transform(
                image=results['img'], mask=results['gt_semantic_seg'])
            results['img'] = aug.get('image')
            results['gt_semantic_seg'] = aug.get('mask')
        return results


@PIPELINES.register_module()
class TiffChannelShuffle:

    def __init__(self, prob):
        self.transform = ChannelShuffle(p=prob)

    def __call__(self, results):
        pair = self.transform(
            image=results['img'], mask=results['gt_semantic_seg'])
        results['img'] = pair.get('image')
        results['gt_semantic_seg'] = pair.get('mask')
        return results


@PIPELINES.register_module()
class TiffMinMaxNormalize:

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, results):
        results['img'] = (results['img'] - self.min) / np.subtract(
            self.max, self.min)
        return results


@PIPELINES.register_module()
class TiffNormalize:

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        results['img'] = (results['img'] - self.mean) / self.std
        return results


@PIPELINES.register_module()
class TiffClipNormalize:

    def __init__(self, min, max):
        self.min = np.array(min)
        self.max = np.array(max)

    def __call__(self, results):
        results['img'] = (results['img'] - self.min) / (self.max - self.min)
        results['img'] = results['img'].clip(0, 1).astype(np.float32)
        return results


@PIPELINES.register_module()
class TiffLogSigmoidNormalize:

    def __call__(self, results):
        results['img'] = (results['img'] + 1) / 10000
        results['img'] = sigmoid(np.log10(results['img']))
        return results


@PIPELINES.register_module()
class TiffTanHNormalize:

    def __call__(self, results):
        results['img'] = (results['img'] * 2) - 1
        results['img'] = np.tanh(results['img'])
        return results


@PIPELINES.register_module()
class TiffClip:

    def __call__(self, results):
        results['img'] = results['img'].clip(0, 1).astype(np.float32)
        return results


@PIPELINES.register_module()
class TiffClipDEM:

    def __call__(self, results):
        clipped_data = results['img'][:, :, :-1].clip(0, 1).astype(np.float32)
        normalized_dem_data = np.expand_dims(
            (results['img'][:, :, -1] / 5000).clip(0, 1), axis=2)
        results['img'] = np.concatenate((clipped_data, normalized_dem_data),
                                        axis=2)
        return results


@PIPELINES.register_module()
class SwapChannels:

    def __call__(self, results):
        results['img'] = results['img'].transpose(2, 0, 1)
        return results
