_base_ = [
    '../runtime/landcover_runtime.py',
    # Network Architecture
    '../../_base_/models/fcn_unet_s5-d16.py',
    # Dataset
    '../datasets/landcover_fuel.py',
    # Customization
    '../custom/base.py',
    # Training schedule
    '../schedules/schedule_160k_adam.py'
]

# Random Seed
seed = 0

# Model
model = dict(
    backbone=dict(in_channels=3),
    decode_head=dict(
        num_classes=9,

        # Cross entropy
        loss_decode=dict(
            type='CrossEntropyLoss',
            avg_non_ignore=True,
            use_sigmoid=False,
            loss_weight=1.0)),
    auxiliary_head=dict(
        num_classes=9,

        # Auxiliary head
        loss_decode=dict(
            type='CrossEntropyLoss',
            avg_non_ignore=True,
            use_sigmoid=False,
            loss_weight=0.4)),

    # Sliding window on test
    test_cfg=dict(
        mode='weighted_slide', crop_size=(512, 512), stride=(256, 256)))

data = dict(samples_per_gpu=3, workers_per_gpu=2)
