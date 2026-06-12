# YOLOv5-n 实例分割,在 Labelme/X-AnyLabeling polygon 数据集(line / QFU)上训练。
# 用法:python tools/train.py configs/yolov5_n_seg_labelme.py
_base_ = ['./_base_/default_runtime.py']

# -------------------- 数据集 --------------------
dataset_type = 'LabelmeDetDataset'
data_root = 'data'
classes = ('line', 'QFU')
num_classes = len(classes)
img_scale = 640

metainfo = dict(classes=classes)

# -------------------- 模型 --------------------
deepen_factor = 0.33
widen_factor = 0.25
strides = [8, 16, 32]
anchors = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
]
head_channels = (64, 128, 256)
num_masks = 32

model = dict(
    type='YOLOv5SegDetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # 冻结整个 backbone(stem~stage4)+ 冻结 BN 统计,用官方权重
        # warm-start 时只微调 neck/head;-1 关闭冻结
        frozen_stages=4,
        norm_eval=True,
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        in_channels=head_channels,
        out_channels=head_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    head=dict(
        type='YOLOv5SegHead',
        num_classes=num_classes,
        in_channels=head_channels,
        strides=strides,
        prior_generator=dict(type='YOLOv5AnchorGenerator', base_sizes=anchors, strides=strides),
        bbox_coder=dict(type='YOLOv5BBoxCoder'),
        assigner=dict(type='YOLOv5BatchAssigner', num_classes=num_classes, strides=strides),
        # proto 中间通道 = 256 * widen_factor(0.25) = 64,需与官方 yolov5n-seg 权重一致
        num_masks=num_masks,
        proto_channels=64,
        loss_box_weight=0.05,
        loss_obj_weight=1.0,
        loss_cls_weight=0.5,
        loss_mask_weight=0.3,
        score_thr=0.001,
        nms_iou_thr=0.45,
        max_per_img=300,
    ),
)

# -------------------- 数据 pipeline --------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLabelmeAnnotations', box_as_mask=True),
    dict(type='LetterResize', scale=img_scale, pad_val=114),
    dict(type='PackDetInputs'),
]
test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_split.txt',
        data_prefix=dict(img='images', ann='jsons'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(img='images', ann='jsons'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

# COCO 风格 0.5:0.95 共 10 档:mAP 即 mAP@[.5:.95],同时输出 mAP_50;
# 复用同一 IoU 矩阵,多阈值几乎不增开销
val_evaluator = dict(
    type='LabelmeSegMetric',
    num_classes=num_classes,
    class_names=classes,
    iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    # mask IoU 评估边长上限:在原 1280² 上算 IoU 时每图 ~2s(GPU→CPU 搬运 + 大
    # matmul);降到 256² 后耗时降一个量级、精度几乎不变。仅影响评估速度,不改
    # 训练 / 推理 / 可视化用的全分辨率 mask。设 None 可恢复全分辨率评估。
    mask_eval_size=256,
)
test_evaluator = val_evaluator

# -------------------- 训练循环 / 优化器 / 调度 --------------------
max_epochs = 100
val_interval = 1

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=val_interval, max_keep_ckpts=3, save_best='labelme_seg/mAP', rule='greater'
    )
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.937, weight_decay=0.0005, nesterov=True),
    clip_grad=dict(max_norm=10.0),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=0.0001, begin=0, end=max_epochs, by_epoch=True, convert_to_iter_based=True),
]

# -------------------- 预训练权重 --------------------
# 优先用 tools/convert_ultralytics.py --seg 转换的 yolov5n-seg 权重(全命中);
# 若只有检测权重,strict=False 也能加载(backbone/neck warm-start,
# proto/mask 系数从零训)。
load_from = 'pretrained/yolov5n_seg_official.pth'
