# optimizer
optimizer = dict(
    type='AdamW',
    # lr=0.00006,
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()

# learning policy
lr_config = dict(
    # policy='CosineAnnealing',
    policy='poly',
    warmup='linear',
    # warmup_iters=1500,
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
