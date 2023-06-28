# dataset settings
dataset_type = 'FuelMapDataset'
# Fuel Maps
data_root = 'data'

# LUCAS
# data_root = 'data/CorineLandCover'

# Filtered Fuel Rare Class Sampling Weights
# rcs_classprob = [
#     0.1217, 0.1232, 0.1231, 0.1211, 0.0947, 0.0961, 0.1096, 0.1122, 0.0983
# ]

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
min = [
    1013.0, 676.0, 448.0, 247.0, 269.0, 253.0, 243.0, 189.0, 61.0, 4.0, 11.0,
    186.0
],
max = [
    2309.0, 4543.05, 4720.2, 5293.05, 3902.05, 4473.0, 5447.0, 5948.05, 1829.0,
    23.0, 4076.05, 5846.0
],
# Minmax normalization means and stds
mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Remove Bands
# remove_bands = [0, 8, 9]

img_norm_cfg = dict(min=min, max=max, to_rgb=False)
crop_size = (512, 512)
num_channels = 12
# Remove bands
# num_channels = 9
train_image_size = 512
test_image_size = 2048
train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadTiffAnnotations'),
    # dict(type='LoadLUCAS'),
    # dict(type='LoadTiffWeights'),
    dict(type='TiffHorizontalFlip', prob=hflip_prob, enable_lucas=False),
    dict(type='TiffVerticalFlip', prob=vflip_prob, enable_lucas=False),
    # dict(type='TiffRandomRotate90', prob=rotation_prob),
    # dict(type='TiffRandomRotate', prob=rotation_prob),
    # dict(
    #     type='TiffGaussianBlur',
    #     blur_limit=blur_limit,
    #     sigma_limit=sigma_limit,
    #     prob=blur_prob),
    # dict(
    #     type='TiffRandomCrop',
    #     img_shape=(image_size, image_size),
    #     scale=crop_scale,
    #     prob=crop_prob),
    dict(
        type='TiffShiftScaleRotate',
        shift_limit=shift_limit,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        prob=shift_scale_rotate_prob,
        enable_lucas=False),
    # dict(
    #     type='TiffRandomBrightnessContrast',
    #     brightness_limit=brightness_limit,
    #     contrast_limit=contrast_limit,
    #     prob=brightness_contrast_prob),
    # dict(type='TiffClipNormalize', min=min, max=max),
    # dict(type='TiffNormalize', mean=mean, std=std),
    # Sigmoid Normalization
    dict(type='TiffLogSigmoidNormalize'),
    dict(type='SwapChannels'),
    dict(
        type='Collect',
        # keys=['img', 'gt_semantic_seg', 'seg_weight'],
        # keys=['img', 'gt_semantic_seg', 'gt_lucas'],
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
            # dict(type='TiffClipNormalize', min=min, max=max),
            # dict(type='TiffNormalize', mean=mean, std=std),
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
            # dict(type='TiffClipNormalize', min=min, max=max),
            # dict(type='TiffNormalize', mean=mean, std=std),
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
        img_dir='FuelMap/Train/Tiles/Img',
        # LUCAS
        ann_dir='LUCAS/Tiles/DisTrain',
        pipeline=train_pipeline,
        img_norm_cfg=img_norm_cfg,
        num_channels=num_channels,
        image_size=train_image_size,
        rcs_enabled=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='FuelMap/Validation/Tiles/Img',
        ann_dir='LUCAS/Tiles/DisVal',
        pipeline=val_pipeline,
        img_norm_cfg=img_norm_cfg,
        num_channels=num_channels,
        image_size=train_image_size),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # Fuel Maps
        img_dir='FuelMap/Test/Sections/Img',
        ann_dir='FuelMap/Test/Sections/Fuel',
        pipeline=test_pipeline,
        img_norm_cfg=img_norm_cfg,
        num_channels=num_channels,
        image_size=test_image_size))
