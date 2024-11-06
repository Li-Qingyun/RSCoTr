_base_ = [
    './RSCoTrSeg_r50_512x512_80k_potsdam_IRRG_all.py',
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/' \
             'v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='RSCoTrSeg',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_queries=100,
    ))

data = dict(samples_per_gpu=1)  # on V100 16G

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'query_embed': dict(decay_mult=0.0),
            'query_feat': dict(decay_mult=0.0),
            'level_embed': dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0))
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=0.000001 / 2, by_epoch=False)
lr_config = dict(_delete_=True, policy='step', step=[60000])

optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
checkpoint_config = dict(by_epoch=False, interval=20000)
runner = dict(type='IterBasedRunner', max_iters=80000)
evaluation = dict(interval=8000, metric=['mIoU', 'mFscore'],
                  pre_eval=True,
                  save_best='auto')
