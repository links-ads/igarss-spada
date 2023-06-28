_base_ = [
    '../_base_/landcover_runtime.py',
    # Network Architecture
    '../_base_/models/segformer_mit-b3.py',
    # Dataset
    '../_base_/datasets/landcover_fuel.py',
    # Customization
    '../_base_/custom/base.py',
    # Training schedule
    '../_base_/schedules/schedule_160k_adam.py'
]

# Random Seed
seed = 0

# Poly loss epsilon
epsilon = 1.0

# Peatbogs fuel maps
# class_weight = [
#     4.03, 6.56, 6.97, 1.0, 2.12, 1.0, 1.01, 1.68, 2.17, 3.32, 1.18, 3.1
# ]

# No peatbogs fuel maps
# class_weight = [4.03, 6.56, 6.97, 2.12, 1.0, 1.01, 1.68, 2.17, 3.32, 1.18, 3.1]

# No peatbogs, mixed and transitional fuel maps
# class_weight = [4.03, 6.56, 6.97, 2.12, 1.0, 1.01, 1.68, 2.17, 1.18]

# 10m fuel maps
# class_weight = [4.18, 6.73, 7.0, 3.61, 1.0, 1.07, 1.74, 2.27, 1.04]

# LUCAS
# class_weight = [4.64, 6.49, 7.67, 6.72, 2.74, 1.17, 1.0, 7.67, 2.11]

# Scribbles
class_weight = [1.0, 5.7, 6.58, 1.0, 1.01, 1.02, 1.56, 1.97, 1.07]

norm_cfg = dict(type='SyncBN', requires_grad=True)

# Model
model = dict(
    pretrained='pretrained/mit_b3.pth',
    backbone=dict(type='mit_b3', style='pytorch'),
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
                class_weight=class_weight,
                avg_non_ignore=True),
            dict(
                type='LucasCrossEntropyLoss',
                loss_weight=1.0,
                class_weight=class_weight,
                avg_non_ignore=True,
                forests_disambiguation=False)
        ]),

    # Sliding window on test
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))
    test_cfg=dict(
        mode='weighted_slide', crop_size=(512, 512), stride=(256, 256)))

data = dict(samples_per_gpu=2, workers_per_gpu=2)
