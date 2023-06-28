_base_ = [
    '../_base_/landcover_runtime.py',
    # Network Architecture
    '../_base_/models/segformer_mit-b5.py',
    # Dataset
    '../_base_/datasets/landcover_v2.py',
    # Customization
    '../landcover/custom_dacs.py',
    # Training schedule
    '../_base_/schedules/schedule_40k_adam.py'
]

# Random Seed
seed = 0

# Scribbles
scribble_class_weight = [1.07, 3.65, 6.86, 1.27, 1.11, 1.0, 1.51, 1.96, 1.01]

# Disambiguated LUCAS
lucas_class_weight = [3.75, 4.87, 7.67, 5.86, 2.05, 1.0, 1.02, 2.28, 1.76]

norm_cfg = dict(type='SyncBN', requires_grad=True)

# Model
model = dict(
    pretrained='pretrained/mit_b5.pth',
    dacs=dict(alpha=0.99, pseudo_threshold=0.968),
    backbone=dict(type='mit_b5', style='pytorch', img_size=512),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg)),
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                class_weight=scribble_class_weight,
                avg_non_ignore=True),
            dict(
                type='LucasCrossEntropyLoss',
                loss_weight=1.0,
                class_weight=lucas_class_weight,
                avg_non_ignore=True,
                forests_disambiguation=True)
        ]),

    # Sliding window on test
    test_cfg=dict(
        mode='weighted_slide', crop_size=(512, 512), stride=(256, 256)))

data = dict(samples_per_gpu=16, workers_per_gpu=2)
