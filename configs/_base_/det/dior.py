# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/DIOR/'
classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt',
           'bridge', 'chimney', 'dam', 'Expressway-Service-area',
           'Expressway-toll-station', 'golffield', 'groundtrackfield',
           'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
           'tenniscourt', 'trainstation', 'vehicle', 'windmill')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_ann/DIOR_train_coco.json',
        img_prefix=data_root + 'JPEGImages-trainval',
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_ann/DIOR_val_coco.json',
        img_prefix=data_root + 'JPEGImages-trainval/',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_ann/DIOR_test_coco.json',
        img_prefix=data_root + 'JPEGImages-test/',
        pipeline=test_pipeline,
        classes=classes))
evaluation = dict(
    interval=1,
    metric='bbox',
    iou_thrs=[0.5],
    save_best='bbox_mAP_50',
    classwise=True)
