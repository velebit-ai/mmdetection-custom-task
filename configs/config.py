# dataset settings
dataset_type = 'ColorFashionDataset'
data_root = '/data/nas/users/jakov/projects/mmdetection-custom-task//data/'
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadColorAnnotations', with_bbox=True, with_colors=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ModifiedDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_colors'])
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0001,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    step=[31])
runner = dict(type='EpochBasedRunner', max_epochs=30)

checkpoint_config = dict(interval=50)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/data/nas/users/jakov/projects/mmdetection-custom-task/checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]

opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=1)

model = dict(
    type='ColorFashionModel',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='DilatedEncoder',
        in_channels=2048,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4,
        block_dilations=[2, 4, 6, 8]),
    bbox_head=dict(
        type='ColorFashionHead',
        num_classes=2,
        num_colors=2,
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2, 4, 8, 16],
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_col=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

custom_imports = dict(
    imports=[
        'color_tagging.datasets.color_fashion_dataset',
        'color_tagging.pipelines.loaders',
        'color_tagging.pipelines.formatting',
        'color_tagging.models.color_fashion_model',
        'color_tagging.models.heads.color_fashion_head'
    ],
    allow_failed_imports=False)
