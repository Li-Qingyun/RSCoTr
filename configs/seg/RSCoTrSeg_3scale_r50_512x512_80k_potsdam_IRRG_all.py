_base_ = [
    '../_base_/seg/potsdam_IRRG_all.py',
    '../_base_/seg/schedule_80k.py',
    '../_base_/seg/default_runtime.py',
]
num_classes = 5
model = dict(
    type='RSCoTrSeg',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    decode_head=dict(
        type='RSCoTrSegHead',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=3,
                        dropout=0.0),
                    ffn_cfgs=dict(
                        type='FFN',
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                temperature=10000,
                normalize=True)),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=10000,
            normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            num_layers=9,
            return_intermediate=True,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    type='FFN',
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    num_fcs=2,
                    ffn_drop=0.0,  # 0.1 for DeformDETR
                    act_cfg=dict(type='ReLU', inplace=True)),
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm'))),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    init_cfg=None)

data = dict(samples_per_gpu=1)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001 / 2,
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


custom_imports = dict(
    imports=[
        'models.seg.rscotr_seg',
        'models.seg.rscotr_seg_head',
    ],
    allow_failed_imports=False)