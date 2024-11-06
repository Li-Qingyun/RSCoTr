_base_ = 'MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py'
# data
data = dict(
    _delete_=True,
    potsdam=dict(
        task='seg',
        config='configs/_base_/seg/potsdam_IRRG_all.py',
        data=dict(samples_per_gpu=2)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=5e-5,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'query_embed': dict(decay_mult=0.0),
            'query_feat': dict(decay_mult=0.0),
            'level_embed': dict(decay_mult=0.0)}))

# learning policy
lr_config = dict(policy='step', step=[60000])
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(interval=100000)
log_config = dict(interval=100)

# evaluation
evaluation = dict(
    _delete_=True,
    interval=400,
    save_best={'potsdam.mFscore': 100},
    seg=dict(
        metric=['mFscore', 'mIoU'],
        pre_eval=True,
        classwise=True))

custom_imports = dict(
    imports='models.multi',
    allow_failed_imports=False)
