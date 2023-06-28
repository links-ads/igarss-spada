# dataset settings
dataset_type = 'LandCoverDataset'
data_root = 'data/CorineLandCover'
hflip_prob = 0.5
vflip_prob = 0.5
rotation_prob = 0.25
blur_limit = (3, 3)
sigma_limit = (0.1, 2.0)
blur_prob = 0.25
crop_scale = (0.8, 1.0)
crop_prob = 0.25
min = [
    1013.0, 676.0, 448.0, 247.0, 269.0, 253.0, 243.0, 189.0, 61.0, 4.0, 11.0,
    186.0
],
max = [
    2309.0, 4543.05, 4720.2, 5293.05, 3902.05, 4473.0, 5447.0, 5948.05, 1829.0,
    23.0, 4076.05, 5846.0
],
mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
img_norm_cfg = dict(min=min, max=max, to_rgb=False)
crop_size = (512, 512)
num_channels = 12
image_size = 512
train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadTiffAnnotations'),
    dict(type='TiffHorizontalFlip', prob=hflip_prob),
    dict(type='TiffVerticalFlip', prob=vflip_prob),
    dict(type='TiffRandomRotate', prob=rotation_prob),
    dict(
        type='TiffGaussianBlur',
        blur_limit=blur_limit,
        sigma_limit=sigma_limit,
        prob=blur_prob),
    dict(
        type='TiffRandomCrop',
        img_shape=(image_size, image_size),
        scale=crop_scale,
        prob=crop_prob),
    dict(type='TiffClipNormalize', min=min, max=max),
    dict(type='TiffNormalize', mean=mean, std=std),
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
            dict(type='TiffClipNormalize', min=min, max=max),
            dict(type='TiffNormalize', mean=mean, std=std),
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
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='TiffClipNormalize', min=min, max=max),
            dict(type='TiffNormalize', mean=mean, std=std),
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
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/training',
        ann_dir='full_ann_dir/training',
        pipeline=train_pipeline,
        img_norm_cfg=img_norm_cfg,
        num_channels=num_channels,
        image_size=image_size),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/validation',
        ann_dir='full_ann_dir/validation',
        pipeline=val_pipeline,
        img_norm_cfg=img_norm_cfg,
        num_channels=num_channels,
        image_size=image_size),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='img_dir/testing',
        img_dir='img_dir/validation',
        ann_dir='full_ann_dir/validation',
        pipeline=test_pipeline,
        img_norm_cfg=img_norm_cfg,
        num_channels=num_channels,
        image_size=image_size))
