# dataset settings
dataset_type = 'FuelMapDEMDataset'
# Fuel Maps
data_root = 'data/FuelMap'

hflip_prob = 0.5
vflip_prob = 0.5
blur_limit = (3, 3)
sigma_limit = (0.1, 2.0)
blur_prob = 0.25
shift_limit = 0.2
scale_limit = (-0.5, 1)
rotate_limit = 180
shift_scale_rotate_prob = 0.5

crop_size = (512, 512)
num_channels = 13

train_image_size = 512
test_image_size = 2048
train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadTiffAnnotations'),
    dict(type='LoadLUCAS'),
    dict(type='Divide10k'),
    dict(type='LoadDEM'),
    dict(type='TiffHorizontalFlip', prob=hflip_prob, enable_lucas=True),
    dict(type='TiffVerticalFlip', prob=vflip_prob, enable_lucas=True),
    dict(
        type='TiffGaussianBlur',
        blur_limit=blur_limit,
        sigma_limit=sigma_limit,
        prob=blur_prob),
    dict(
        type='TiffShiftScaleRotate',
        shift_limit=shift_limit,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        prob=shift_scale_rotate_prob,
        enable_lucas=True),
    dict(type='TiffClipDEM'),
    dict(type='SwapChannels'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_lucas'],
        meta_keys=('img_info', 'img_shape', 'ori_shape'))
]
val_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadTiffAnnotations'),
    dict(type='Divide10k'),
    dict(type='LoadDEM'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='TiffClipDEM'),
            dict(type='SwapChannels'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('img_info', 'img_shape', 'ori_shape',
                           'ori_filename', 'pad_shape', 'flip'))
        ])
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='Divide10k'),
    dict(type='LoadDEM'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 2048),
        flip=False,
        transforms=[
            dict(type='TiffClipDEM'),
            dict(type='SwapChannels'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('img_info', 'img_shape', 'ori_shape',
                           'ori_filename', 'pad_shape', 'flip'))
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Train/Tiles/Img',
        # DEM
        dem_dir='data/DEM/Tiles/Train',
        # Scribble
        ann_dir='Train/Tiles/Scribble',
        # LUCAS
        lucas_dir='data/LUCAS/Tiles/DisTrain',
        pipeline=train_pipeline,
        num_channels=num_channels,
        image_size=train_image_size),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Validation/Tiles/Img',
        # DEM
        dem_dir='data/DEM/Tiles/Val',
        ann_dir='Validation/Tiles/Fuel',
        pipeline=val_pipeline,
        num_channels=num_channels,
        image_size=train_image_size),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Test/Sections/Img',
        # DEM
        dem_dir='data/DEM/Tiles/test',
        # Fuel Maps
        ann_dir='Test/Sections/Fuel',
        pipeline=test_pipeline,
        num_channels=num_channels,
        image_size=test_image_size))
