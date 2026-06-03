# YOLOv5-n 在 Labelme/X-AnyLabeling 数据集(line / QFU 两类)上的训练配置。
# 用法:python tools/train.py configs/yolov5_n_labelme.py
_base_ = ['./_base_/default_runtime.py']

# -------------------- 数据集 --------------------
dataset_type = 'LabelmeDetDataset'
data_root = 'data'
classes = ('line', 'QFU')
num_classes = len(classes)
img_scale = 640

metainfo = dict(classes=classes)

# -------------------- 模型 --------------------
# yolov5n: deepen=0.33, widen=0.25 → P3/P4/P5 通道 (64, 128, 256)
deepen_factor = 0.33
widen_factor = 0.25
strides = [8, 16, 32]
anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)],  # P5/32
]
head_channels = (64, 128, 256)

model = dict(
    type='YOLOv5Detector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(type='YOLOv5CSPDarknet', deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        type='YOLOv5PAFPN',
        in_channels=head_channels,
        out_channels=head_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    head=dict(
        type='YOLOv5Head',
        num_classes=num_classes,
        in_channels=head_channels,
        strides=strides,
        prior_generator=dict(type='YOLOv5AnchorGenerator', base_sizes=anchors, strides=strides),
        bbox_coder=dict(type='YOLOv5BBoxCoder'),
        assigner=dict(type='YOLOv5BatchAssigner', num_classes=num_classes, strides=strides),
        loss_box_weight=0.05,
        loss_obj_weight=1.0,
        loss_cls_weight=0.5,
        score_thr=0.001,
        nms_iou_thr=0.45,
        max_per_img=300,
    ),
)

# -------------------- 数据 pipeline --------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLabelmeAnnotations'),
    dict(type='LetterResize', scale=img_scale, pad_val=114),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLabelmeAnnotations'),
    dict(type='LetterResize', scale=img_scale, pad_val=114),
    dict(type='PackDetInputs'),
]

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

val_evaluator = dict(
    type='LabelmeDetMetric',
    num_classes=num_classes,
    class_names=classes,
    iou_thrs=[0.5])
test_evaluator = val_evaluator

# -------------------- 训练循环 / 优化器 / 调度 --------------------
max_epochs = 100
val_interval = 10  # 每 10 个 epoch 评估一次

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs,
    val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
# 由 tools/convert_ultralytics.py 从官方 yolov5n.pt 转换而来(COCO 80 类)。
# strict=False 加载:backbone/neck 全部命中做 warm-start,head 因类别数
# 不同(80 → 2)自动跳过,等价 COCO 预训练 finetune。
load_from = 'pretrained/yolov5n_official.pth'
