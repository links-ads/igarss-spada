import json
import os.path as osp

import numpy as np

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import LoadTiffAnnotation


@DATASETS.register_module()
class FuelMapDataset(CustomDataset):

    # 11 classes
    # CLASSES = ('Artificial', 'Bare', 'Wetlands', 'Water', 'Grassland',
    #            'Agricultural', 'Broadleaves', 'Coniferous', 'Mixed', 'Shrubs',
    #            'Transitional')

    # PALETTE = [[189, 189, 189], [204, 204, 204], [80, 197, 179], [17, 36, 204],
    #            [134, 214, 68], [229, 119, 34], [61, 117, 62], [49, 81, 55],
    #            [42, 131, 35], [94, 165, 121], [0, 128, 128]]

    # 9 classes
    CLASSES = ('Artificial', 'Bare', 'Wetlands', 'Water', 'Grassland',
               'Agricultural', 'Broadleaves', 'Coniferous', 'Shrubs')

    PALETTE = [[214, 58, 61], [154, 154, 154], [150, 107, 196], [43, 80, 198],
               [249, 159, 39], [253, 211, 39], [36, 152, 1], [8, 98, 0],
               [141, 140, 0]]

    def __init__(self,
                 image_size: int = 512,
                 num_channels: int = 12,
                 img_norm_cfg: dict = None,
                 weight_dir: str = None,
                 lucas_dir: str = None,
                 rcs_enabled=False,
                 samples_with_class_path=None,
                 rcs_classprob=None,
                 **kwargs):
        super().__init__(
            img_suffix='.tif', seg_map_suffix='_MAP.tif', **kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.dims = (image_size, image_size)
        self.img_norm_cfg = img_norm_cfg
        self.gt_seg_map_loader = LoadTiffAnnotation()

        self.weight_dir = osp.join(self.data_root,
                                   weight_dir) if weight_dir else None

        self.lucas_dir = lucas_dir

        self.rcs_enabled = rcs_enabled

        if rcs_enabled:
            assert samples_with_class_path is not None, 'Rare class sampling is enabled but samples file path is not given'
            with open(samples_with_class_path) as samples_with_class_file:
                self.samples_with_class = json.load(samples_with_class_file)
            self.rcs_classes = sorted(self.samples_with_class.keys())
            assert rcs_classprob is not None, 'Rare class sampling is enabled but class probabilities are not given'
            self.rcs_classprob = rcs_classprob

            self.rcs_min_pixels = 1

    def prepare_train_img(self, idx: int):
        img_info = self.img_infos[idx]

        results = dict(
            img_dir=self.img_dir,
            ann_dir=self.ann_dir,
            weight_dir=self.weight_dir,
            lucas_dir=self.lucas_dir,
            img_info=img_info,
            out_shape=(self.num_channels, *self.dims),
            img_norm_cfg=self.img_norm_cfg,
            ori_shape=self.dims,
            img_shape=self.dims)
        results = self.pipeline(results)
        return results

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(
            img_dir=self.img_dir,
            ann_dir=self.ann_dir,
            img_info=img_info,
            out_shape=(self.num_channels, *self.dims),
            img_norm_cfg=self.img_norm_cfg,
            ori_shape=self.dims,
            img_shape=self.dims,
            pad_shape=self.dims,
            ori_filename=img_info["filename"],
            flip=False)
        results = self.pipeline(results)
        return results

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        idx = np.random.choice(self.samples_with_class[c])
        for _ in range(10):
            source = self.prepare_train_img(idx)
            n_class = (source['gt_semantic_seg'] == int(c)).sum()
            if n_class >= self.rcs_min_pixels:
                break

        return source

    def __getitem__(self, idx):

        if self.test_mode:
            return self.prepare_test_img(idx)
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            return self.prepare_train_img(idx)
