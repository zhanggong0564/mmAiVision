# mmAiVision

基于 [MMEngine](https://github.com/open-mmlab/mmengine) 的轻量级视觉训练框架，内置从零实现的 **YOLOv5 目标检测 / 实例分割**，并原生支持 **LabelMe / X-AnyLabeling** polygon 标注数据集。

本项目脱胎于 MMEngine 最佳实践模板，沿用其 Registry + Config 的简洁开发范式（无需遵循 `BaseDataElement` 的严格数据流，目录扁平、模块自动注册），并在其上构建了一套完整、可独立训练与推理的 YOLOv5 检测/分割算法栈。

## 主要特性

- **YOLOv5 全栈复现**：CSPDarknet backbone + PAFPN neck + YOLOv5 检测头 / 分割头（Proto + mask 系数），结构与子模块命名与 ultralytics 对齐。
- **目标检测 + 实例分割**：`YOLOv5Detector` 与 `YOLOv5SegDetector` 两种检测器，配置切换。
- **LabelMe 原生数据集**：`LabelmeDetDataset` 直接读取 LabelMe / X-AnyLabeling 导出的 `json + 图片`，支持矩形框与多边形 mask。
- **官方权重转换**：`tools/convert_ultralytics.py` 把 ultralytics 官方 YOLOv5(-seg) 权重映射到本仓库命名体系，并做端到端数值等价校验。
- **MMEngine 基建**：自动注册、混合精度、分布式 / Slurm 训练、检查点恢复、可视化与日志开箱即用。

## 目录结构

```bash
├── configs
│   ├── yolov5_n_labelme.py          # YOLOv5-n 目标检测训练配置
│   ├── yolov5_n_seg_labelme.py      # YOLOv5-n 实例分割训练配置
│   └── _base_                       # dataset / scheduler / runtime 基础配置
├── demo
│   ├── yolov5_detect_demo.py        # 检测可视化推理
│   ├── yolov5_seg_demo.py           # 实例分割可视化推理(mask 叠加)
│   └── yolov5_inference.py          # 配置驱动的通用推理脚手架
├── mmaivision
│   ├── datasets                     # LabelmeDetDataset + transforms
│   ├── models
│   │   ├── base/single_stage.py     # 单阶段检测器基类
│   │   └── yolov5                   # backbone / neck / head / loss / task_utils
│   ├── evaluation                   # LabelmeDetMetric / LabelmeSegMetric
│   ├── engine                       # hooks / optimizers / schedulers
│   ├── infer                        # inferencer
│   └── registry.py                  # 各 Registry 与自动注册位置
├── tools
│   ├── train.py / test.py           # 训练 / 测试入口
│   ├── convert_ultralytics.py       # ultralytics 权重转换
│   ├── verify_dataset.py            # 数据集校验与可视化
│   └── dist_*.sh / slurm_*.sh       # 分布式 / Slurm 启动脚本
└── pretrained                       # 预训练 / 转换后权重
```

## 安装

1. 按[官方指南](https://pytorch.org/get-started/locally/)安装 PyTorch。
2. 安装 MMEngine 与 MMCV：
   ```bash
   pip install -U openmim
   mim install mmengine
   mim install "mmcv>=2.0.0"
   ```
3. 以开发模式安装本项目：
   ```bash
   pip install -e .
   ```

## 数据准备

数据集为 LabelMe / X-AnyLabeling 导出格式：图片与同名 `.json` 标注（矩形框或多边形）放在同一目录。配置中通过 `data_root` 指向数据根目录，`classes` 指定类别（示例为 `('line', 'QFU')`）。

训练前可先校验数据集并可视化标注：

```bash
python tools/verify_dataset.py configs/yolov5_n_seg_labelme.py --out-dir vis --num 5
```

## 训练

```bash
# 目标检测
python tools/train.py configs/yolov5_n_labelme.py

# 实例分割
python tools/train.py configs/yolov5_n_seg_labelme.py

# 常用选项
python tools/train.py <config> --amp        # 混合精度
python tools/train.py <config> --resume     # 从检查点恢复
python tools/train.py <config> --work-dir work_dirs/my_exp
```

分布式 / Slurm：

```bash
bash tools/dist_train.sh <config> <num_gpus>
bash tools/slurm_train.sh <partition> <job_name> <config>
```

## 测试

```bash
python tools/test.py <config> <checkpoint> [--out results.pkl]
```

评估指标：检测使用 `LabelmeDetMetric`（mAP），分割使用 `LabelmeSegMetric`（mask mAP，流式逐图匹配以避免验证集 mask 累积导致 OOM）。

## 推理演示

```bash
# 检测
python demo/yolov5_detect_demo.py <img|dir> configs/yolov5_n_labelme.py \
    <checkpoint> --out-dir work_dirs/infer

# 实例分割(mask 半透明叠加回原图)
python demo/yolov5_seg_demo.py <img|dir> configs/yolov5_n_seg_labelme.py \
    <checkpoint> --out-dir work_dirs/infer_seg --score-thr 0.3
```

## 加载官方 YOLOv5 权重

把 ultralytics 官方权重转换到本仓库命名体系（转换后默认做端到端数值等价校验）：

```bash
# 自动从 torch.hub 拉取并转换检测权重
python tools/convert_ultralytics.py --size s --out work_dirs/yolov5s_official.pth

# 转换实例分割权重(需本地 .pt)
python tools/convert_ultralytics.py --size n --seg \
    --weights yolov5n-seg.pt --out work_dirs/yolov5n_seg_official.pth
```

## 扩展开发

本项目沿用 MMEngine 的 Registry 自动注册机制：在 `models/` `datasets/` `evaluation/` `engine/` 等默认目录下用 `@XXX.register_module()` 注册新模块，并在对应 `__init__.py` 中导入即可被配置引用。如需在新位置注册，更新 `mmaivision/registry.py` 中的 `locations` 参数。

各 Registry 与对应位置：

| Registry | 位置 |
|----------|------|
| `MODELS` | `mmaivision/models/` |
| `DATASETS` | `mmaivision/datasets/datasets.py` |
| `TRANSFORMS` | `mmaivision/datasets/transforms.py` |
| `METRICS` | `mmaivision/evaluation/metrics.py` |
| `HOOKS` / `OPTIMIZERS` / `PARAM_SCHEDULERS` | `mmaivision/engine/` |

配置中通过 `default_scope = 'mmaivision'` 指定默认注册域。
