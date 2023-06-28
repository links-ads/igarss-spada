# dataset settings
dataset_type = 'FuelMapDataset'
# Fuel Maps
data_root = 'data/FuelMap'

# Filtered Fuel Rare Class Sampling Weights
rcs_classprob = [
    0.1217, 0.1232, 0.1231, 0.1211, 0.0947, 0.0961, 0.1096, 0.1122, 0.0983
]

hflip_prob = 0.5
vflip_prob = 0.5
rotation_prob = 0.25  # 0.5
blur_limit = (3, 3)
sigma_limit = (0.1, 2.0)
blur_prob = 0.25
crop_scale = (0.8, 1.0)
crop_prob = 0.25
shift_limit = 0.2
scale_limit = (-0.5, 1)
rotate_limit = 180
shift_scale_rotate_prob = 0.5
brightness_limit = 0.1
contrast_limit = 0.1
brightness_contrast_prob = 0.5
channel_shuffle_prob = 0.25

crop_size = (512, 512)
num_channels = 12
train_image_size = 512
test_image_size = 2048
train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadTiffAnnotations'),
    dict(type='TiffHorizontalFlip', prob=hflip_prob, enable_lucas=False),
    dict(type='TiffVerticalFlip', prob=vflip_prob, enable_lucas=False),
    dict(
        type='TiffShiftScaleRotate',
        shift_limit=shift_limit,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        prob=shift_scale_rotate_prob,
        enable_lucas=False),
    # Sigmoid Normalization
    dict(type='TiffLogSigmoidNormalize'),
    dict(type='SwapChannels'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('img_info', 'img_shape', 'ori_shape', 'img_norm_cfg'))
]
val_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadTiffAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            # Sigmoid Normalization
            dict(type='TiffLogSigmoidNormalize'),
            dict(type='SwapChannels'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('img_info', 'img_shape', 'ori_shape',
                           'ori_filename', 'img_norm_cfg', 'pad_shape',
                           'flip'))
        ])
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 2048),
        flip=False,
        transforms=[
            # Sigmoid Normalization
            dict(type='TiffLogSigmoidNormalize'),
            dict(type='SwapChannels'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('img_info', 'img_shape', 'ori_shape',
                           'ori_filename', 'img_norm_cfg', 'pad_shape',
                           'flip'))
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Train/Tiles/Img',
        # Filtered Fuel
        ann_dir='Train/Tiles/FilteredFuel',
        pipeline=train_pipeline,
        num_channels=num_channels,
        image_size=train_image_size,
        rcs_enabled=True,
        samples_with_class_path='tools/samples_with_class.json',
        rcs_classprob=rcs_classprob),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Validation/Tiles/Img',
        ann_dir='Validation/Tiles/Fuel',
        pipeline=val_pipeline,
        num_channels=num_channels,
        image_size=train_image_size),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # Fuel Maps
        img_dir='Test/Sections/Img',
        ann_dir='Test/Sections/Fuel',
        pipeline=test_pipeline,
        num_channels=num_channels,
        image_size=test_image_size))
