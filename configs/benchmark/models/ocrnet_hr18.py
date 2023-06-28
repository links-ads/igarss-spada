_base_ = [
    '../runtime/landcover_runtime.py',
    # Network Architecture
    '../base/ocrnet_hr18.py',
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

    # Sliding window on test
    test_cfg=dict(
        mode='weighted_slide', crop_size=(512, 512), stride=(256, 256)))

data = dict(samples_per_gpu=4, workers_per_gpu=2)
