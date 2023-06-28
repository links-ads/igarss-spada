from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import LoadTiffAnnotation


@DATASETS.register_module()
class FuelMapDEMDataset(CustomDataset):

    # 9 classes
    CLASSES = ('Artificial', 'Bare', 'Wetlands', 'Water', 'Grassland',
               'Agricultural', 'Broadleaves', 'Coniferous', 'Shrubs')

    PALETTE = [[214, 58, 61], [154, 154, 154], [150, 107, 196], [43, 80, 198],
               [249, 159, 39], [253, 211, 39], [36, 152, 1], [8, 98, 0],
               [141, 140, 0]]

    def __init__(self,
                 image_size: int = 512,
                 num_channels: int = 13,
                 img_norm_cfg: dict = None,
                 dem_dir: str = None,
                 lucas_dir: str = None,
                 **kwargs):
        super().__init__(
            img_suffix='.tif', seg_map_suffix='_MAP.tif', **kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.dims = (image_size, image_size)
        self.img_norm_cfg = img_norm_cfg
        self.gt_seg_map_loader = LoadTiffAnnotation()

        self.dem_dir = dem_dir
        self.lucas_dir = lucas_dir

    def prepare_train_img(self, idx: int):
        img_info = self.img_infos[idx]

        results = dict(
            img_dir=self.img_dir,
            ann_dir=self.ann_dir,
            lucas_dir=self.lucas_dir,
            dem_dir=self.dem_dir,
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
            dem_dir=self.dem_dir,
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

    def __getitem__(self, idx):

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)
