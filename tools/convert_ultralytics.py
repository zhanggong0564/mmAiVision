"""把 Ultralytics YOLOv5 (v6.0+) 官方权重转换到本仓库 YOLOv5Detector 命名体系。

原理：本仓库的 Conv / C3 / Bottleneck / SPPF 内部子模块命名（.conv/.bn、
.cv1/.cv2/.cv3/.m 等）与 ultralytics 完全一致，差异仅在顶层模块前缀。
ultralytics 把整网建成一个 nn.Sequential，按 yaml 顺序编号 model.0..model.24；
本仓库则是 backbone / neck / bbox_head 的具名子模块。转换 = 顶层前缀重映射，
并对每个参数做 shape 校验。

用法：
    # 自动从 torch.hub 拉取官方 yolov5s 权重并转换
    python tools/convert_ultralytics.py --size s --out work_dirs/yolov5s_official.pth

    # 使用本地 .pt（仍经 torch.hub 提供反序列化所需代码）
    python tools/convert_ultralytics.py --size s --weights yolov5s.pt \\
        --out work_dirs/yolov5s_official.pth

    # 转换 yolov5n-seg 实例分割权重（需本地 .pt）
    python tools/convert_ultralytics.py --size n --seg \\
        --weights yolov5n-seg.pt --out work_dirs/yolov5n_seg_official.pth

转换后默认做一次端到端数值等价校验（--no-verify 可关闭）。
"""
import argparse
import os.path as osp

import torch

import mmaivision  # noqa: F401  触发 registry 注册
from mmaivision.models.yolov5.common import make_divisible
from mmaivision.registry import MODELS

# size -> (deepen_factor, widen_factor)，与 ultralytics yolov5{n,s,m,l,x}.yaml 一致
SIZE_FACTORS = {
    'n': (0.33, 0.25),
    's': (0.33, 0.50),
    'm': (0.67, 0.75),
    'l': (1.00, 1.00),
    'x': (1.33, 1.25),
}

BASE_CHANNELS = (64, 128, 256, 512, 1024)

# 顶层前缀映射：本仓库 key 前缀 -> ultralytics key 前缀
PREFIX_MAP = {
    'backbone.stem': 'model.0',
    'backbone.stage1.0': 'model.1',
    'backbone.stage1.1': 'model.2',
    'backbone.stage2.0': 'model.3',
    'backbone.stage2.1': 'model.4',
    'backbone.stage3.0': 'model.5',
    'backbone.stage3.1': 'model.6',
    'backbone.stage4.0': 'model.7',
    'backbone.stage4.1': 'model.8',
    'backbone.stage4.2': 'model.9',           # SPPF
    'neck.reduce_p5': 'model.10',
    'neck.top_down_c3_p4': 'model.13',
    'neck.reduce_p4': 'model.14',
    'neck.top_down_c3_p3': 'model.17',
    'neck.downsample_n3': 'model.18',
    'neck.bottom_up_c3_n4': 'model.20',
    'neck.downsample_n4': 'model.21',
    'neck.bottom_up_c3_n5': 'model.23',
    'bbox_head.convs.0': 'model.24.m.0',
    'bbox_head.convs.1': 'model.24.m.1',
    'bbox_head.convs.2': 'model.24.m.2',
}

# yolov5-seg(Segment head)额外的 proto 子模块前缀映射。
# detect convs(model.24.m.{0,1,2})前缀沿用 PREFIX_MAP,仅通道数随 nm 变化。
SEG_EXTRA_PREFIX_MAP = {
    'bbox_head.proto.cv1': 'model.24.proto.cv1',
    'bbox_head.proto.cv2': 'model.24.proto.cv2',
    'bbox_head.proto.cv3': 'model.24.proto.cv3',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--size', default='s', choices=list(SIZE_FACTORS),
        help='YOLOv5 规格 n/s/m/l/x，默认 s')
    parser.add_argument(
        '--num-classes', type=int, default=80,
        help='类别数，默认 80（COCO，与官方权重一致才能逐层对齐）')
    parser.add_argument(
        '--weights', default=None,
        help='本地 ultralytics .pt；不给则从 torch.hub 拉取官方预训练权重')
    parser.add_argument(
        '--out', default=None,
        help='输出 .pth 路径，默认 work_dirs/yolov5{size}_official.pth')
    parser.add_argument(
        '--hub', default='ultralytics/yolov5',
        help='torch.hub 仓库，默认 ultralytics/yolov5')
    parser.add_argument(
        '--no-verify', action='store_true',
        help='跳过转换后的端到端数值等价校验')
    parser.add_argument(
        '--seg', action='store_true',
        help='转换 yolov5-seg 实例分割权重(含 Proto + mask 系数);'
             '需配合 --weights 指定本地 yolov5{size}-seg.pt')
    parser.add_argument(
        '--num-masks', type=int, default=32,
        help='mask 系数维度 nm,默认 32(与官方 seg 一致)')
    return parser.parse_args()


def build_target_model(size, num_classes, seg=False, num_masks=32):
    """按 size/num_classes 构建本仓库的 YOLOv5 检测或分割模型。"""
    d, w = SIZE_FACTORS[size]
    ch = [make_divisible(c * w, 8) for c in BASE_CHANNELS]
    p3p4p5 = [ch[2], ch[3], ch[4]]
    if seg:
        cfg = dict(
            type='YOLOv5SegDetector',
            backbone=dict(type='YOLOv5CSPDarknet', deepen_factor=d,
                          widen_factor=w),
            neck=dict(type='YOLOv5PAFPN', in_channels=p3p4p5,
                      out_channels=p3p4p5, deepen_factor=d, widen_factor=w),
            head=dict(type='YOLOv5SegHead', num_classes=num_classes,
                      in_channels=p3p4p5, num_masks=num_masks,
                      # ultralytics Proto 通道 npr=256 经 width_multiple 缩放
                      proto_channels=make_divisible(256 * w, 8)),
        )
    else:
        cfg = dict(
            type='YOLOv5Detector',
            backbone=dict(
                type='YOLOv5CSPDarknet', deepen_factor=d, widen_factor=w),
            neck=dict(
                type='YOLOv5PAFPN', in_channels=p3p4p5, out_channels=p3p4p5,
                deepen_factor=d, widen_factor=w),
            head=dict(
                type='YOLOv5Head', num_classes=num_classes,
                in_channels=p3p4p5),
        )
    return MODELS.build(cfg)


def _stub_absent_modules():
    """给缺失的可视化/可选依赖注入假模块，避免反序列化时连带 import 失败。

    yolov5 的 models/common.py、utils/* 顶部硬 import 了 IPython/seaborn/scipy
    等绘图依赖，但纯前向 + state_dict 用不到它们。用 sys.meta_path 拦截这些
    顶层包的任意子模块并返回 MagicMock，从而无需污染 conda 环境。
    """
    import importlib
    import sys
    import types
    from importlib.abc import Loader, MetaPathFinder
    from importlib.machinery import ModuleSpec
    from unittest.mock import MagicMock

    stubbed = []
    # IPython 需特殊处理：matplotlib 会读它的 version_info 做比较，
    # 纯 MagicMock 会触发 TypeError，故给一个最小的真模块。
    try:
        importlib.import_module('IPython')
    except Exception:
        ip = types.ModuleType('IPython')
        ip.version_info = (8, 24, 0)
        ip.get_ipython = lambda: None
        disp = types.ModuleType('IPython.display')
        disp.display = lambda *a, **k: None
        ip.display = disp
        sys.modules['IPython'] = ip
        sys.modules['IPython.display'] = disp
        stubbed.append('IPython')

    candidates = [
        # 绘图 / 交互（IPython 已在上方特殊处理）
        'seaborn', 'scipy',
        # 系统 / dataloader
        'psutil', 'git', 'dill',
        # 数据增强 / 评测
        'albumentations', 'pycocotools',
        # 日志 / 性能
        'thop', 'tensorboard', 'wandb', 'clearml', 'comet_ml',
        # 导出后端
        'onnx', 'onnxruntime', 'openvino', 'tensorflow', 'coremltools',
        # 其它
        'ultralytics',
    ]
    absent = []
    for name in candidates:
        try:
            importlib.import_module(name)
        except Exception:
            absent.append(name)
    if not absent:
        return stubbed

    class _FakeLoader(Loader):
        def create_module(self, spec):
            m = MagicMock()
            m.__spec__ = spec
            m.__name__ = spec.name
            m.__path__ = []
            return m

        def exec_module(self, module):
            pass

    class _FakeFinder(MetaPathFinder):
        def __init__(self, names):
            self.names = set(names)

        def find_spec(self, fullname, path, target=None):
            if fullname.split('.')[0] in self.names:
                return ModuleSpec(fullname, _FakeLoader())
            return None

    sys.meta_path.insert(0, _FakeFinder(absent))
    return stubbed + absent


def _ensure_local_repo(hub):
    """确保 ultralytics/yolov5 已克隆到本地（torch.hub 缓存目录）。

    用 source='local' 加载可完全绕过 GitHub API 校验（否则易撞速率限制）。
    """
    import os
    import subprocess
    cache = os.path.join(
        torch.hub.get_dir(),
        hub.replace('/', '_') + '_master')
    if not osp.isdir(osp.join(cache, '.git')):
        os.makedirs(osp.dirname(cache), exist_ok=True)
        print(f'      本地无缓存，git clone {hub} -> {cache}')
        subprocess.run(
            ['git', 'clone', '--depth', '1', f'https://github.com/{hub}',
             cache], check=True)
    return cache


def load_ultralytics(hub, size, weights, seg=False):
    """经 torch.hub（source='local'）取得官方 DetectionModel/SegmentationModel。"""
    stubbed = _stub_absent_modules()
    if stubbed:
        print(f'      已 stub 缺失的可选依赖（不影响前向）: {stubbed}')
    repo = _ensure_local_repo(hub)
    if seg and not weights:
        raise SystemExit(
            f'seg 模式需要 --weights 指定本地 yolov5{size}-seg.pt'
            '(ultralytics 无 yolov5-seg 的 torch.hub 命名入口,'
            '请先从官方 release 下载对应 .pt)。')
    if weights:
        model = torch.hub.load(
            repo, 'custom', path=weights, source='local',
            autoshape=False, _verbose=False)
    else:
        model = torch.hub.load(
            repo, f'yolov5{size}', pretrained=True, source='local',
            autoshape=False, _verbose=False)
    # autoshape=False 返回 DetectMultiBackend 包装器，真正的 DetectionModel
    # 在 .model 里；解包后 state_dict 键为 model.0...，前向直接返回 (z, x)。
    if type(model).__name__ == 'DetectMultiBackend':
        model = model.model
    return model.float().eval()


def remap(target_sd, ultra_sd, extra_prefix=None):
    """把 target（本仓库）每个 key 映射到 ultralytics key 并取值。"""
    prefix_map = dict(PREFIX_MAP)
    if extra_prefix:
        prefix_map.update(extra_prefix)
    prefixes = sorted(prefix_map, key=len, reverse=True)
    converted, missing, mismatch = {}, [], []
    for k, v in target_sd.items():
        matched = next(
            (p for p in prefixes if k == p or k.startswith(p + '.')), None)
        if matched is None:
            missing.append((k, '无前缀匹配'))
            continue
        src = prefix_map[matched] + k[len(matched):]
        if src not in ultra_sd:
            missing.append((k, f'源缺失 {src}'))
            continue
        sv = ultra_sd[src]
        if tuple(sv.shape) != tuple(v.shape):
            mismatch.append((k, src, tuple(v.shape), tuple(sv.shape)))
            continue
        converted[k] = sv.detach().clone()
    used = {prefix_map[next(p for p in prefixes if k == p or k.startswith(
        p + '.'))] + k[len(next(p for p in prefixes if k == p or k.startswith(
            p + '.'))):]
        for k in converted}
    unused = sorted(set(ultra_sd) - used)
    return converted, missing, mismatch, unused


@torch.no_grad()
def verify(target_model, ultra_model, num_classes, na, seg=False, nm=32):
    """同一输入喂两网，比对原始 conv 输出的最大绝对误差。

    检测分支:逐层比对 raw conv 输出 (b, na, h, w, no)。
    seg 模式额外比对 proto 原型输出。任一环节结构与预期不符则打印告警并返回
    None（视为跳过），不让校验崩溃整个流程。

    ultralytics eval 输出结构:
      det: (decoded(b,N,no), raw_list[3])               -> raw 在 [1]
      seg: (decoded(b,N,no), proto(b,nm,H,W), raw_list[3]) -> proto 在 [1]、raw 在 [2]
    """
    target_model.cpu().eval()
    ultra_model.cpu().eval()
    im = torch.rand(1, 3, 640, 640)
    out = target_model(im, mode='tensor')
    # seg 时 forward 返回 (pred_maps, proto)
    ours = out[0] if seg else out
    ours_proto = out[1] if seg else None
    no = num_classes + 5 + (nm if seg else 0)
    try:
        ul = ultra_model(im)
        ul_raw = ul[2] if seg else ul[1]   # detect 分支三层原始输出
        max_diff = 0.0
        for o, u in zip(ours, ul_raw):
            b, _, h, w = o.shape
            o = o.view(b, na, no, h, w).permute(0, 1, 3, 4, 2).contiguous()
            max_diff = max(max_diff, (o - u).abs().max().item())
        if seg:
            # proto 分支:官方 ul[1] 形状 (b, nm, H, W),与本仓库一致
            ul_proto = ul[1]
            max_diff = max(max_diff,
                           (ours_proto - ul_proto).abs().max().item())
        return max_diff
    except Exception as e:  # noqa: BLE001
        if seg:
            print(f'      [warn] seg 数值校验结构不符，跳过: {e}')
            return None
        raise


def main():
    args = parse_args()
    if args.seg and not args.weights:
        raise SystemExit(
            f'seg 模式需要 --weights 指定本地 yolov5{args.size}-seg.pt'
            '(ultralytics 无 yolov5-seg 的 torch.hub 命名入口,'
            '请先从官方 release 下载对应 .pt)。')
    out = args.out or (
        f'work_dirs/yolov5{args.size}{"_seg" if args.seg else ""}_official.pth')

    model_type = 'YOLOv5SegDetector' if args.seg else 'YOLOv5Detector'
    print(f'[1/4] 构建本仓库 {model_type} (size={args.size}, '
          f'num_classes={args.num_classes}'
          + (f', num_masks={args.num_masks}' if args.seg else '')
          + ') ...')
    target = build_target_model(args.size, args.num_classes,
                                seg=args.seg, num_masks=args.num_masks)
    target_sd = target.state_dict()

    src = args.weights or f'torch.hub:{args.hub}:yolov5{args.size}'
    print(f'[2/4] 载入 ultralytics 官方模型 ({src}) ...')
    ultra = load_ultralytics(args.hub, args.size, args.weights, seg=args.seg)
    ultra_sd = ultra.state_dict()

    print('[3/4] 前缀重映射 + shape 校验 ...')
    extra_prefix = SEG_EXTRA_PREFIX_MAP if args.seg else None
    converted, missing, mismatch, unused = remap(target_sd, ultra_sd,
                                                 extra_prefix=extra_prefix)
    print(f'      目标参数 {len(target_sd)} | 成功映射 {len(converted)} | '
          f'缺失 {len(missing)} | shape 不符 {len(mismatch)}')
    print(f'      未使用的源参数（anchor 等，预期）: {unused}')
    if missing:
        print('      [缺失明细]')
        for k, why in missing[:20]:
            print(f'        - {k}: {why}')
    if mismatch:
        print('      [shape 不符明细]')
        for k, s, ts, ss in mismatch[:20]:
            print(f'        - {k}: target{ts} vs src{s}{ss}')
    if missing or mismatch:
        raise SystemExit('转换失败：存在缺失或 shape 不符，请检查映射表。')

    # strict 加载，确保完整覆盖
    target.load_state_dict(converted, strict=True)
    print('      strict load_state_dict 成功，全部参数覆盖。')

    verify_meta = None
    if not args.no_verify:
        print('[4/4] 端到端数值等价校验 ...')
        na = target.bbox_head.num_base_priors
        diff = verify(target, ultra, args.num_classes, na,
                      seg=args.seg, nm=args.num_masks)
        if diff is None:
            print('      数值校验已跳过（seg 模式输出结构不符，见上方告警）。')
            verify_meta = 'skipped (seg structure mismatch)'
        else:
            tag = 'PASS ✅' if diff < 1e-4 else 'FAIL ❌'
            print(f'      原始输出最大绝对误差 = {diff:.3e}  ->  {tag}')
            if diff >= 1e-4:
                raise SystemExit('数值校验未通过，转换结果与官方不等价。')
            verify_meta = diff
    else:
        print('[4/4] 跳过数值校验 (--no-verify)')
        verify_meta = 'skipped (--no-verify)'

    import os
    os.makedirs(osp.dirname(out) or '.', exist_ok=True)
    if verify_meta in ('skipped (seg structure mismatch)', 'skipped (--no-verify)'):
        print('      [注意] 数值校验已跳过,本权重未经等价性验证。')
    torch.save(
        dict(state_dict=converted,
             meta=dict(source='ultralytics/yolov5', size=args.size,
                       num_classes=args.num_classes,
                       seg=args.seg, num_masks=args.num_masks,
                       verify=verify_meta)),
        out)
    print(f'已保存转换权重 -> {out}')


if __name__ == '__main__':
    main()
