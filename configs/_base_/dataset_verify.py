# 数据加载验证用 config：指向 data/（images + jsons），类别 line/QFU。
# 仅用于 tools/verify_dataset.py 校验标注解析与可视化。

dataset_type = 'LabelmeDetDataset'
data_root = 'data'

metainfo = dict(classes=('line', 'QFU'))

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
        data_prefix=dict(img='images', ann='jsons'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ))

val_dataloader = train_dataloader
test_dataloader = val_dataloader

val_evaluator = dict(type='DumpResults', out_file_path='val_results.pkl')
test_evaluator = val_evaluator
