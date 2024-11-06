_base_ = 'default_runtime.py'
# model
backbone_pretrained = 'https://github.com/SwinTransformer/storage/' \
                      'releases/download/v1.0.0/' \
                      'swin_tiny_patch4_window7_224.pth'
det_pretrainded = '/home/rs/Desktop/pretrained/dino_swin-t-p4-w7_mmdet.pth'
model = dict(
    type='MTL',
    backbone=dict(
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
        init_cfg=dict(type='Pretrained', checkpoint=backbone_pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    shared_encoder=dict(
        type='DetrTransformerEncoder',
        num_layers=6,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=dict(
                type='MultiScaleDeformableAttention',
                embed_dims=256,
                num_levels=4,
                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfgs=dict(
                type='FFN',
                feedforward_channels=2048,  # 1024 for DeformDETR
                num_fcs=2,
                ffn_drop=0.0,  # 0.1 for DeformDETR
                act_cfg=dict(type='ReLU', inplace=True)),
            operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
    cls_head=dict(
        type='SlvlClsHead',
        num_classes=45,
        in_channels=768,
        # init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    bbox_head=dict(
        type='DINOHead',
        num_query=600,
        num_classes=20,
        num_feature_levels=4,
        in_channels=2048,  # TODO
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=1.0),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='DinoTransformer',
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),  # 0.1 for DeformDETR
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=4,
                            dropout=0.0),  # 0.1 for DeformDETR
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        feedforward_channels=2048,  # 1024 for DeformDETR
                        num_fcs=2,
                        ffn_drop=0.0,  # 0.1 for DeformDETR
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    seg_head=dict(
        type='Mask2FormerHead',
        in_channels=[96, 192, 384, 768],  # pass to pixel_decoder inside
        scheme=2,
        feat_channels=256,
        out_channels=256,
        num_classes=5,
        num_queries=100,  # 100
        num_transformer_feat_level=4,
        align_corners=False,
        pixel_decoder=dict(
            type='MlvlSegPixelDecoder',
            num_outs=4,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
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
    task_weight=dict(cls=1, det=1, seg=0.1),
    train_cfg=dict(
        cls=dict(
            augments=[
                dict(type='BatchMixup', alpha=0.8, num_classes=45, prob=0.5),
                dict(type='BatchCutMix', alpha=1.0, num_classes=45, prob=0.5)
            ]),
        det=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        seg=dict(),
        task_pretrain=dict(rule='dino_mmdet', pretrained=det_pretrainded)),
        # shoule add a '--load-task-pretrain' in args to launch task_pretrain
    test_cfg=dict(
        cls=dict(),
        det=dict(max_per_img=300),
        seg=dict(mode='whole'))
)

# data
data = dict(
    resisc=dict(
        task='cls',
        config='configs/_base_/cls/resisc_swin_224.py',
        data=dict(
            samples_per_gpu=16)),
    dior=dict(
        task='det',
        config='configs/_base_/det/dior.py',
        data=dict(
            samples_per_gpu=1)),
    potsdam=dict(
        task='seg',
        config='configs/_base_/seg/potsdam_IRRG_all.py',
        data=dict(
            samples_per_gpu=2)))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'query_embed': dict(decay_mult=0.0),
            'query_feat': dict(decay_mult=0.0),
            'level_embed': dict(decay_mult=0.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[240000, 285000])  # 750000
runner = dict(type='IterBasedRunner', max_iters=300000)  # 900000
checkpoint_config = dict(interval=100000)
log_config = dict(interval=300)

# evaluation
evaluation = dict(
    interval=15000,
    save_best={
        'resisc.accuracy_top-1': 1,
        'dior.bbox_mAP': 100,
        'potsdam.mFscore': 100},
    cls=dict(
        metric='accuracy'),
    det=dict(
        metric='bbox',
        iou_thrs=[0.5],
        classwise=True),
    seg=dict(
        metric=['mFscore', 'mIoU'],
        pre_eval=True,
        classwise=True))

custom_imports = dict(
    imports='models.multi',
    allow_failed_imports=False)
