_base_ = [
    '../_base_/cls/swin-tiny.py',
    '../_base_/cls/resisc_swin_224.py',
    '../_base_/cls/resisc_adamw_swin.py',
    '../_base_/cls/default_runtime.py'
]

model = dict(
    neck=None,
    head=dict(
        type='SlvlClsHead',
        num_classes=45,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False))

checkpoint_config = dict(interval=50)
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # TODO: tune lr
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(_delete_=True, policy='step', step=[150])
runner = dict(type='EpochBasedRunner', max_epochs=200)
# evaluation
evaluation = dict(interval=1)

custom_imports = dict(
    imports='models.multi',
    allow_failed_imports=False)