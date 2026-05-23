# Labelme/X-AnyLabeling 风格目标检测数据集示例配置。
# 用户使用时把 data_root / classes / ann_file 替换为自己的值。

dataset_type = 'LabelmeDetDataset'
data_root = 'data/my_dataset'

metainfo = dict(classes=('dc_line', ))

# 最小 pipeline：仅读图。后续可在 _base_/dataset.py 之上叠加 transforms。
# 若安装了 mmdet，可在下游 config 里追加 mmdet.PackDetInputs 等。
train_pipeline = [dict(type='LoadImageFromFile')]
test_pipeline = [dict(type='LoadImageFromFile')]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img='images', ann='annotations'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(img='images', ann='annotations'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(type='DumpResults', out_file_path='val_results.pkl')
test_evaluator = val_evaluator
