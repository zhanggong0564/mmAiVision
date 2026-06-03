# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Preference / 语言偏好

**All responses should be in Chinese / 所有回答请使用中文**

当用户用中文提问时，请用中文回答。代码注释和文档可以使用中英文混合。

## Git Commit Guidelines / Git 提交规范

### Core Principle / 核心原则

**IMPORTANT: All commits MUST be split by module/responsibility. Never mix multiple modules in a single commit.**

**重要：所有提交必须按模块/职责拆分。严禁在一次提交中混合多个模块。**

When modifying files across different modules, you MUST create separate commits for each module.

当修改涉及不同模块的文件时，必须为每个模块创建单独的提交。

### Commit Message Format / 提交消息格式

All commits must follow the conventional commit format with module-based prefixes:

所有提交必须遵循约定式提交格式，使用模块前缀：

```
<module>: <type>: <description>
```

### Module Prefixes / 模块前缀

Commits should be organized by module/responsibility:

提交应按模块/职责分类：

| Module Prefix | Scope / 范围 |
|--------------|-------------|
| `panel_label_ocr` | Panel label OCR detection / 线标 OCR 检测 |
| `plate_screw` | Plate screw detection / 铁片螺丝检测 |
| `dc_fuse` | DC fuse detection / 直流熔丝检测 |
| `indicator` | Indicator light detection / 指示灯检测 |
| `lap_surf` | Lap surface detection / 搭接面检测 |
| `router` | API routing layer / API 路由层 |
| `schema` | Data models & schemas / 数据模型 |
| `config` | Configuration files / 配置文件 |
| `service` | Service layer (shared) / 服务层（通用） |
| `utils` | Utility functions / 工具函数 |
| `test` | Test cases / 测试用例 |
| `docs` | Documentation / 文档 |
| `ci` | CI/CD configuration / CI/CD 配置 |
| `deps` | Dependencies / 依赖管理 |

### Commit Types / 提交类型

| Type | Description / 描述 | When to Use / 使用场景 |
|------|-------------------|----------------------|
| `feat` | New feature / 新功能 | Adding new functionality |
| `fix` | Bug fix / 修复 | Fixing existing bugs |
| `refactor` | Refactoring / 重构 | Code restructuring without behavior change |
| `style` | Code style / 代码风格 | Formatting, comments, naming |
| `chore` | Maintenance / 维护 | Tooling, gitignore, non-config changes |
| `update` | Update / 更新 | Updating models, data, or configs |
| `perf` | Performance / 性能 | Performance improvements |

### Commit Workflow / 提交流程

**CRITICAL: When modifying files from multiple modules, you MUST split them into separate commits.**

**关键：当修改涉及多个模块的文件时，必须拆分为单独的提交。**

#### Step-by-step Process / 逐步流程

1. **Identify modified files and their modules / 识别修改的文件及其模块**:
   ```bash
   git status
   ```

2. **Stage files for ONE module at a time / 每次只暂存一个模块的文件**:
   ```bash
   # Example: First commit - config module
   git add config/panel_label_config.py
   git commit -m "config: update: 新增 OCR 文本检测相关配置参数"
   
   # Second commit - service module
   git add services/panel_label/business_logic.py services/panel_label/panel_label_detect.py
   git commit -m "panel_label_ocr: refactor: 重构 OCRPipeline 为三阶段独立架构"
   
   # Third commit - product_type module
   git add services/panel_label/product_type.py
   git commit -m "panel_label_ocr: update: 更新 PRODUCT_guideline 产品类型定义"
   
   # Fourth commit - utils module
   git add services/panel_label/utils.py
   git commit -m "panel_label_ocr: style: 清理 utils.py 冗余代码"
   
   # Fifth commit - demo module
   git add demo/panel_label_demo.py
   git commit -m "panel_label_ocr: style: 调整 demo 调试代码注释"
   ```

3. **Verify all commits are properly separated / 验证所有提交已正确分离**:
   ```bash
   git log --oneline -5
   ```

### Real Example / 实际案例

When you have modifications like this:
当你有以下修改时：

```
Modified files:
- config/panel_label_config.py          → config module
- services/panel_label/business_logic.py → panel_label_ocr module
- services/panel_label/panel_label_detect.py → panel_label_ocr module
- services/panel_label/product_type.py  → panel_label_ocr module
- services/panel_label/utils.py         → panel_label_ocr module
- demo/panel_label_demo.py              → panel_label_ocr module
```

You MUST create 5 separate commits (grouping related service files):
你必须创建 5 个单独的提交（将相关的服务文件分组）：

```bash
# Commit 1: config
git add config/panel_label_config.py
git commit -m "config: update: 新增 OCR 文本检测相关配置参数"

# Commit 2: core service logic
git add services/panel_label/business_logic.py services/panel_label/panel_label_detect.py
git commit -m "panel_label_ocr: refactor: 重构 OCRPipeline 为三阶段独立架构"

# Commit 3: product type definitions
git add services/panel_label/product_type.py
git commit -m "panel_label_ocr: update: 更新 PRODUCT_guideline 产品类型定义"

# Commit 4: utility functions
git add services/panel_label/utils.py
git commit -m "panel_label_ocr: style: 清理 utils.py 冗余代码"

# Commit 5: demo scripts
git add demo/panel_label_demo.py
git commit -m "panel_label_ocr: style: 调整 demo 调试代码注释"
```

### Commit Examples / 提交示例

```bash
# Feature addition / 添加新功能
git commit -m "panel_label_ocr: feat: 新增 PRODUCT_guideline 条目 1017KM1_1 和 201X1_1"

# Bug fix / 修复 bug
git commit -m "panel_label_ocr: fix: 恢复 DATA_DIR 路径为 wind_power 批量模式"

# Style change / 代码风格调整
git commit -m "panel_label_ocr: style: 移除 export_ocr_dataset.py 中的 Stage 编号等冗余注释"

# Refactoring / 重构
git commit -m "panel_label_ocr: refactor: 重构 demo 支持批量处理和结果可视化"

# Config update / 配置更新
git commit -m "config: update: 升级模型路径至 v3，提高 confThreshold 至 0.75"

# Test update / 测试更新
git commit -m "test: refactor: 重构单元测试体系，统一使用 pytest 风格"

# Chore / 维护任务
git commit -m "chore: 更新 .gitignore 排除临时数据和 demo 数据目录"
```


## 项目概述

**MMEngine Template** 是一个基于 MMEngine 的最佳实践模板项目,用于简化深度学习项目的开发流程。它提供了一个轻量级的开发标准,比 OpenMMLab 下游算法仓库(如 MMDet、MMCls)更简洁灵活。

## 常用命令

### 安装依赖

```bash
# 安装 PyTorch (按照官方指南)
# https://pytorch.org/get-started/locally/

# 安装 MMEngine
pip install -U openmim
mim install mmengine

# 如果需要 MMCV
mim install "mmcv>=2.0.0"

# 以开发模式安装本项目
pip install -e .
```

### 训练模型

```bash
python tools/train.py <config> [options]
# 示例:
python tools/train.py configs/_base_/dataset.py --work-dir work_dirs/my_model
python tools/train.py configs/_base_/dataset.py --amp  # 启用混合精度训练
python tools/train.py configs/_base_/dataset.py --resume  # 从检查点恢复
```

### 测试模型

```bash
python tools/test.py <config> <checkpoint> [options]
# 示例:
python tools/test.py configs/_base_/dataset.py work_dirs/my_model/epoch_1.pth
python tools/test.py configs/_base_/dataset.py work_dirs/my_model/epoch_1.pth --out results.pkl
```

### 推理演示

```bash
python demo/mmengine_template_demo.py <image> <config> <checkpoint>
# 示例:
python demo/mmengine_template_demo.py test.jpg configs/_base_/dataset.py model.pth --out-file result
```

### 代码质量工具

```bash
# 运行 pre-commit 钩子
pre-commit run --all-files

# 单独运行代码检查
flake8 .
isort .
yapf -i *.py
```

## 架构概览

### 核心设计模式

本项目基于 MMEngine 的 **Registry 系统**实现模块自动注册和配置管理。关键特点:

1. **简化的目录结构**: 无需创建过多嵌套目录,便于维护
2. **灵活的数据流**: 不强制要求遵循 `BaseDataElement`,允许自定义数据格式
3. **自动注册**: 遵循默认目录结构的模块会自动注册到 Registry

### 注册表系统

所有模块通过 `mmengine_template.registry` 中定义的 Registry 进行注册和管理:

- **RUNNERS**: 训练/测试循环管理器
- **MODELS**: 模型定义 (位于 `models/model.py`)
- **DATASETS**: 数据集 (位于 `datasets/datasets.py`)
- **TRANSFORMS**: 数据转换 (位于 `datasets/transforms.py`)
- **HOOKS**: 训练钩子 (位于 `engine/hooks.py`)
- **METRICS**: 评估指标 (位于 `evaluation/metrics.py`)
- **OPTIMIZERS**: 优化器 (位于 `engine/optimizers.py`)
- **PARAM_SCHEDULERS**: 学习率调度器 (位于 `engine/schedulers.py`)

每个 Registry 都继承自 MMEngine 的对应 Registry,并通过 `locations` 参数指定自动注册路径。

### 配置文件系统

配置文件位于 `configs/` 目录,使用 MMEngine 的 Config 系统:

- `_base_/`: 基础配置模块
  - `dataset.py`: 数据集配置
  - `scheduler.py`: 学习率调度配置
  - `default_runtime.py`: 默认运行时配置(hooks、日志、可视化等)

配置合并优先级: CLI > 文件片段 > 文件名

### 关键组件位置

**自定义模型**: `mmengine_template/models/model.py`
- 继承 `mmengine.model.BaseModel`
- 使用 `@MODELS.register_module()` 装饰器注册

**自定义数据集**: `mmengine_template/datasets/datasets.py`
- 继承 `mmengine.dataset.BaseDataset`
- 实现 `load_data_list()` 方法加载数据

**自定义数据转换**: `mmengine_template/datasets/transforms.py`
- 实现 callable 对象或使用 `@TRANSFORMS.register_module()`

**自定义 Hook**: `mmengine_template/engine/hooks.py`
- 继承 MMEngine 的 Hook 基类

**评估指标**: `mmengine_template/evaluation/metrics.py`
- 实现自定义评估逻辑

### 训练流程

1. `tools/train.py` 解析命令行参数
2. 加载配置文件 (`Config.fromfile()`)
3. 根据配置构建 Runner (`Runner.from_cfg()` 或 `RUNNERS.build()`)
4. 调用 `runner.train()` 开始训练
5. Runner 自动处理:
   - 分布式训练设置
   - 优化器和调度器初始化
   - Hook 注册和执行
   - 日志记录
   - 检查点保存

### 扩展新模块

添加新模块时的步骤:

1. 在对应位置创建文件(如 `models/my_model.py`)
2. 实现类并使用 `@XXX.register_module()` 装饰
3. 在对应的 `__init__.py` 中导入该类(触发注册)
4. 如需在新位置注册模块,更新 `registry.py` 中的 `locations` 参数

**重要**: 确保模块在 `__init__.py` 中被导入,否则不会注册!

### 注意事项

- 项目中大量使用 `mmengine_template` 作为 scope 名称,使用时需替换为你的项目名称
- 不需要将数据格式化为 `BaseDataElement`,可以使用更灵活的数据结构
- 配置文件中设置 `default_scope = 'mmengine_template'` 来指定默认注册域
- 使用 `--cfg-options` 可以在命令行覆盖配置项,支持嵌套结构
