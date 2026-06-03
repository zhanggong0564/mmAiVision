# YOLOv5-s 在 Labelme/X-AnyLabeling 数据集(line / QFU 两类)上的训练配置。
# 用法:python tools/train.py configs/yolov5_s_labelme.py
_base_ = ['./_base_/default_runtime.py']

# -------------------- 数据集 --------------------
dataset_type = 'LabelmeDetDataset'
data_root = 'data'
classes = ('line', 'QFU')
num_classes = len(classes)
img_scale = 640

metainfo = dict(classes=classes)

# -------------------- 模型 --------------------
# yolov5s: deepen=0.33, widen=0.5 → P3/P4/P5 通道 (128, 256, 512)
deepen_factor = 0.33
widen_factor = 0.5
strides = [8, 16, 32]
anchors = [
    [(10, 13), (16, 30), (33, 23)],       # P3/8
    [(30, 61), (62, 45), (59, 119)],      # P4/16
    [(116, 90), (156, 198), (373, 326)],  # P5/32
]
head_channels = (128, 256, 512)

model = dict(
    type='YOLOv5Detector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        type='YOLOv5PAFPN',
        in_channels=head_channels,
        out_channels=head_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    head=dict(
        type='YOLOv5Head',
        num_classes=num_classes,
        in_channels=head_channels,
        strides=strides,
        prior_generator=dict(
            type='YOLOv5AnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        bbox_coder=dict(type='YOLOv5BBoxCoder'),
        assigner=dict(
            type='YOLOv5BatchAssigner',
            num_classes=num_classes,
            strides=strides),
        loss_box_weight=0.05,
        loss_obj_weight=1.0,
        loss_cls_weight=0.5,
        score_thr=0.001,
        nms_iou_thr=0.45,
        max_per_img=300))

# -------------------- 数据 pipeline --------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLabelmeAnnotations'),
    dict(type='LetterResize', scale=img_scale, pad_val=114),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img='images', ann='jsons'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline))

# -------------------- 训练循环 / 优化器 / 调度 --------------------
max_epochs = 100

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=max_epochs)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True),
    clip_grad=dict(max_norm=10.0))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0001,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# 训练 only:不配置 val/test loop。
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
