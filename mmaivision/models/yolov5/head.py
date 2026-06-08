"""YOLOv5 Detect head:三层 1x1 Conv + loss_by_feat + predict_by_feat。"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torchvision.ops import batched_nms

from mmaivision.registry import MODELS, TASK_UTILS

from .common import Proto
from .iou_loss import bbox_ciou

# YOLOv5 默认 COCO anchors(像素)与 strides。
DEFAULT_ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
]


@MODELS.register_module()
class YOLOv5Head(BaseModule):
    """YOLOv5 Detect head。

    ``forward`` 输出三层原始 feature(纯网络结构,不做 decode);
    ``loss_by_feat`` / ``predict_by_feat`` 负责训练 / 推理的解码与匹配。

    Args:
        num_classes: 类别数 nc。
        in_channels: 三个输入通道,通常等于 neck.out_channels。
        num_base_priors: 每层 anchor 数,默认 3。
        strides: 三层 stride,默认 (8, 16, 32)。
        prior_generator: ``YOLOv5AnchorGenerator`` 配置,None 时用 DEFAULT_ANCHORS。
        bbox_coder: ``YOLOv5BBoxCoder`` 配置,None 时用默认。
        assigner: ``YOLOv5BatchAssigner`` 配置,None 时用默认。
        loss_box_weight / loss_obj_weight / loss_cls_weight: loss 权重。
        obj_level_weights: 三层 obj loss 缩放,默认 (4.0, 1.0, 0.4)。
        score_thr / nms_iou_thr / max_per_img: postprocess 阈值。
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Sequence[int],
                 num_base_priors: int = 3,
                 strides: Sequence[int] = (8, 16, 32),
                 prior_generator: Optional[dict] = None,
                 bbox_coder: Optional[dict] = None,
                 assigner: Optional[dict] = None,
                 loss_box_weight: float = 0.05,
                 loss_obj_weight: float = 1.0,
                 loss_cls_weight: float = 0.5,
                 obj_level_weights: Sequence[float] = (4.0, 1.0, 0.4),
                 score_thr: float = 0.001,
                 nms_iou_thr: float = 0.45,
                 max_per_img: int = 300,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert len(in_channels) == 3, \
            f'in_channels 长度必须为 3, got {len(in_channels)}'
        assert len(strides) == 3, \
            f'strides 长度必须为 3, got {len(strides)}'
        assert len(obj_level_weights) == 3, \
            f'obj_level_weights 长度必须为 3, got {len(obj_level_weights)}'
        if num_classes < 1:
            raise ValueError(f'num_classes 必须 >= 1, got {num_classes}')
        if num_base_priors < 1:
            raise ValueError(
                f'num_base_priors 必须 >= 1, got {num_base_priors}')
        if not (0 < score_thr < 1):
            raise ValueError(f'score_thr 必须 ∈ (0, 1), got {score_thr}')
        if not (0 < nms_iou_thr < 1):
            raise ValueError(f'nms_iou_thr 必须 ∈ (0, 1), got {nms_iou_thr}')
        if max_per_img < 1:
            raise ValueError(f'max_per_img 必须 >= 1, got {max_per_img}')
        for w_name, w in [('loss_box_weight', loss_box_weight),
                          ('loss_obj_weight', loss_obj_weight),
                          ('loss_cls_weight', loss_cls_weight)]:
            if w < 0:
                raise ValueError(f'{w_name} 必须 >= 0, got {w}')

        self.num_classes = num_classes
        self.num_base_priors = num_base_priors
        self.strides = list(strides)
        self.loss_box_weight = loss_box_weight
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.obj_level_weights = list(obj_level_weights)
        self.score_thr = score_thr
        self.nms_iou_thr = nms_iou_thr
        self.max_per_img = max_per_img

        if prior_generator is None:
            prior_generator = dict(
                type='YOLOv5AnchorGenerator',
                base_sizes=DEFAULT_ANCHORS,
                strides=list(strides))
        if bbox_coder is None:
            bbox_coder = dict(type='YOLOv5BBoxCoder')
        if assigner is None:
            assigner = dict(
                type='YOLOv5BatchAssigner',
                num_classes=num_classes,
                strides=list(strides))
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.assigner = TASK_UTILS.build(assigner)

        out_c = num_base_priors * (num_classes + 5)
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, out_c, kernel_size=1) for c in in_channels])

    def forward(self, feats: Tuple[Tensor, ...]) -> List[Tensor]:
        return [conv(f) for conv, f in zip(self.convs, feats)]

    def loss_by_feat(self,
                     pred_maps: List[Tensor],
                     batch_gt_instances,
                     batch_img_metas) -> dict:
        """计算三段 loss(bbox CIoU + obj BCE + cls BCE)。

        Args:
            pred_maps: 三层 ``[B, na*(nc+5), Hi, Wi]``。
            batch_gt_instances: 长度 B 的 InstanceData 列表。
            batch_img_metas: 长度 B 的 dict 列表(当前实现不读字段)。

        Returns:
            dict 含 ``loss_bbox / loss_obj / loss_cls``,已乘权重。
        """
        B = pred_maps[0].shape[0]
        assert len(batch_gt_instances) == B, \
            f'batch_gt_instances 长度 {len(batch_gt_instances)} ' \
            f'!= pred batch {B}'

        nc = self.num_classes
        na = self.num_base_priors
        device = pred_maps[0].device
        dtype = pred_maps[0].dtype
        del batch_img_metas

        featmap_sizes = [(p.shape[2], p.shape[3]) for p in pred_maps]
        anchors_per_layer = self.prior_generator.grid_priors(
            featmap_sizes, device=device, dtype=dtype)
        # grid_priors 返回 (na, ny, nx, 2);assigner 期望 (na, 2)
        anchors_for_assigner = [a[:, 0, 0, :] for a in anchors_per_layer]
        assignments = self.assigner(
            batch_gt_instances, anchors_for_assigner, featmap_sizes)

        loss_bbox = torch.zeros((), device=device, dtype=dtype)
        loss_obj = torch.zeros((), device=device, dtype=dtype)
        loss_cls = torch.zeros((), device=device, dtype=dtype)
        bce = F.binary_cross_entropy_with_logits

        for i, raw_map in enumerate(pred_maps):
            Hi, Wi = raw_map.shape[2], raw_map.shape[3]
            raw = raw_map.view(B, na, nc + 5, Hi, Wi).permute(
                0, 1, 3, 4, 2).contiguous()  # (B, na, Hi, Wi, nc+5)

            a = assignments[i]
            M = a['img_idx'].numel()
            obj_target = torch.zeros(
                (B, na, Hi, Wi), device=device, dtype=dtype)

            if M > 0:
                matched_anchor = anchors_for_assigner[i][a['anchor_idx']]
                pos = raw[a['img_idx'], a['anchor_idx'],
                          a['grid_y'], a['grid_x']]
                pos_xy = pos[:, 0:2].sigmoid() * 2 - 0.5
                pos_wh = (pos[:, 2:4].sigmoid() * 2) ** 2 * matched_anchor
                pos_cls = pos[:, 5:]

                grid_xy_int = torch.stack(
                    [a['grid_x'].to(dtype), a['grid_y'].to(dtype)], dim=-1)
                pred_cxcywh = torch.cat([pos_xy, pos_wh], dim=-1)
                target_cxcywh = torch.cat(
                    [a['gt_xy'] - grid_xy_int, a['gt_wh']], dim=-1)
                pred_xyxy = _cxcywh_to_xyxy(pred_cxcywh)
                target_xyxy = _cxcywh_to_xyxy(target_cxcywh)
                ciou = bbox_ciou(pred_xyxy, target_xyxy)
                loss_bbox = loss_bbox + (1.0 - ciou).mean()

                obj_target[a['img_idx'], a['anchor_idx'],
                           a['grid_y'], a['grid_x']] = ciou.detach().clamp(0)

                if nc > 1:
                    cls_target = F.one_hot(
                        a['gt_class'], num_classes=nc).to(dtype)
                    loss_cls = loss_cls + bce(pos_cls, cls_target).mean()

            loss_obj_layer = bce(raw[..., 4], obj_target).mean()
            loss_obj = loss_obj + loss_obj_layer * self.obj_level_weights[i]

        loss_bbox = loss_bbox * self.loss_box_weight * B
        loss_obj = loss_obj * self.loss_obj_weight * B
        loss_cls = loss_cls * self.loss_cls_weight * B
        return dict(loss_bbox=loss_bbox, loss_obj=loss_obj, loss_cls=loss_cls)

    def predict_by_feat(self,
                        pred_maps: List[Tensor],
                        batch_img_metas) -> list:
        """解码 + per-image NMS,返回 List[InstanceData]。"""
        B = pred_maps[0].shape[0]
        nc = self.num_classes
        na = self.num_base_priors
        device = pred_maps[0].device
        dtype = pred_maps[0].dtype
        del batch_img_metas

        featmap_sizes = [(p.shape[2], p.shape[3]) for p in pred_maps]
        anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device, dtype=dtype)
        grid_xy = self.prior_generator.grid_xy(
            featmap_sizes, device=device, dtype=dtype)

        all_xyxy = []
        all_scores = []
        for i, raw_map in enumerate(pred_maps):
            Hi, Wi = raw_map.shape[2], raw_map.shape[3]
            stride = self.strides[i]
            raw = raw_map.view(B, na, nc + 5, Hi, Wi).permute(
                0, 1, 3, 4, 2).contiguous()
            sig = torch.sigmoid(raw)
            xy = (sig[..., 0:2] * 2 - 0.5
                  + grid_xy[i].view(1, 1, Hi, Wi, 2)) * stride
            wh = (sig[..., 2:4] * 2) ** 2 \
                * anchors[i].view(1, na, Hi, Wi, 2) * stride
            obj = sig[..., 4:5]
            cls = sig[..., 5:]
            score = obj * cls

            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            xyxy = torch.cat([x1y1, x2y2], dim=-1).view(B, -1, 4)
            score = score.view(B, -1, nc)
            all_xyxy.append(xyxy)
            all_scores.append(score)

        all_xyxy = torch.cat(all_xyxy, dim=1)
        all_scores = torch.cat(all_scores, dim=1)

        results = []
        for b in range(B):
            scores_b, labels_b = all_scores[b].max(dim=-1)
            keep = scores_b > self.score_thr
            bboxes_kept = all_xyxy[b][keep]
            scores_kept = scores_b[keep]
            labels_kept = labels_b[keep]
            if bboxes_kept.shape[0] == 0:
                results.append(InstanceData(
                    bboxes=torch.zeros(0, 4, device=device, dtype=dtype),
                    scores=torch.zeros(0, device=device, dtype=dtype),
                    labels=torch.zeros(0, dtype=torch.int64, device=device)))
                continue
            keep_idx = batched_nms(
                bboxes_kept, scores_kept, labels_kept, self.nms_iou_thr)
            keep_idx = keep_idx[:self.max_per_img]
            results.append(InstanceData(
                bboxes=bboxes_kept[keep_idx],
                scores=scores_kept[keep_idx],
                labels=labels_kept[keep_idx]))
        return results


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


@MODELS.register_module()
class YOLOv5SegHead(YOLOv5Head):
    """YOLOv5 实例分割 head(YOLACT 风格)。

    在 ``YOLOv5Head`` 基础上:每 anchor 多输出 ``num_masks`` 维 mask 系数,
    并用 ``Proto`` 在 P3 上生成共享原型;实例 mask = sigmoid(系数 @ 原型) 经
    bbox 裁剪。检测三段 loss 复用父类逻辑(此处重写以同时算 mask)。
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_base_priors=3,
                 strides=(8, 16, 32),
                 prior_generator=None,
                 bbox_coder=None,
                 assigner=None,
                 loss_box_weight=0.05,
                 loss_obj_weight=1.0,
                 loss_cls_weight=0.5,
                 obj_level_weights=(4.0, 1.0, 0.4),
                 score_thr=0.001,
                 nms_iou_thr=0.45,
                 max_per_img=300,
                 num_masks=32,
                 proto_channels=256,
                 loss_mask_weight=0.05,
                 mask_ratio=4,
                 init_cfg=None):
        super().__init__(
            num_classes=num_classes, in_channels=in_channels,
            num_base_priors=num_base_priors, strides=strides,
            prior_generator=prior_generator, bbox_coder=bbox_coder,
            assigner=assigner, loss_box_weight=loss_box_weight,
            loss_obj_weight=loss_obj_weight, loss_cls_weight=loss_cls_weight,
            obj_level_weights=obj_level_weights, score_thr=score_thr,
            nms_iou_thr=nms_iou_thr, max_per_img=max_per_img,
            init_cfg=init_cfg)
        if num_masks < 1:
            raise ValueError(f'num_masks 必须 >= 1, got {num_masks}')
        if loss_mask_weight < 0:
            raise ValueError(
                f'loss_mask_weight 必须 >= 0, got {loss_mask_weight}')
        if mask_ratio <= 0:
            raise ValueError(f'mask_ratio 必须 > 0, got {mask_ratio}')
        self.num_masks = num_masks
        self.proto_channels = proto_channels
        self.loss_mask_weight = loss_mask_weight
        # mask_ratio = P3_stride / Proto上采样倍率 (默认 8/2=4);
        # 改 Proto 上采样或接入不同 stride 时需同步调整,确保 mask 上采样回输入分辨率。
        self.mask_ratio = mask_ratio
        # 重建 convs:每 anchor 输出 nc+5+nm(父类只建了 nc+5)
        out_c = num_base_priors * (num_classes + 5 + num_masks)
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, out_c, kernel_size=1) for c in in_channels])
        self.proto = Proto(in_channels[0], proto_channels, num_masks)

    def forward(self, feats):
        pred_maps = [conv(f) for conv, f in zip(self.convs, feats)]
        proto = self.proto(feats[0])
        return pred_maps, proto

    @staticmethod
    def _crop_mask(masks, boxes):
        """把 box 外的像素置 0。masks (N,H,W) float;boxes (N,4) xyxy(mask 坐标)。"""
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # 各 (n,1,1)
        r = torch.arange(
            w, device=masks.device, dtype=x1.dtype)[None, None, :]
        c = torch.arange(
            h, device=masks.device, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def loss_by_feat(self, pred_maps, proto, batch_gt_instances,
                     batch_img_metas):
        B = pred_maps[0].shape[0]
        assert len(batch_gt_instances) == B, \
            f'batch_gt_instances 长度 {len(batch_gt_instances)} != pred batch {B}'
        nc = self.num_classes
        na = self.num_base_priors
        nm = self.num_masks
        no = nc + 5 + nm
        device = pred_maps[0].device
        dtype = pred_maps[0].dtype
        del batch_img_metas

        featmap_sizes = [(p.shape[2], p.shape[3]) for p in pred_maps]
        anchors_per_layer = self.prior_generator.grid_priors(
            featmap_sizes, device=device, dtype=dtype)
        anchors_for_assigner = [a[:, 0, 0, :] for a in anchors_per_layer]
        assignments = self.assigner(
            batch_gt_instances, anchors_for_assigner, featmap_sizes)

        # 拼 batch 全部 gt mask,顺序须与 assigner 的 gt_idx 一致
        # (assigner 跳过 bboxes.numel()==0 的图,这里同样跳过)
        mask_list = []
        have_masks = True
        for gt in batch_gt_instances:
            if gt.bboxes.numel() == 0:
                continue
            m = getattr(gt, 'masks', None)
            if m is None:
                have_masks = False
                break
            mask_list.append(m.to(device=device, dtype=dtype))
        all_masks = torch.cat(mask_list, dim=0) \
            if (have_masks and mask_list) else None

        loss_bbox = torch.zeros((), device=device, dtype=dtype)
        loss_obj = torch.zeros((), device=device, dtype=dtype)
        loss_cls = torch.zeros((), device=device, dtype=dtype)
        loss_mask = torch.zeros((), device=device, dtype=dtype)
        bce = F.binary_cross_entropy_with_logits

        for i, raw_map in enumerate(pred_maps):
            Hi, Wi = raw_map.shape[2], raw_map.shape[3]
            raw = raw_map.view(B, na, no, Hi, Wi).permute(
                0, 1, 3, 4, 2).contiguous()  # (B, na, Hi, Wi, no)

            a = assignments[i]
            M = a['img_idx'].numel()
            obj_target = torch.zeros(
                (B, na, Hi, Wi), device=device, dtype=dtype)

            if M > 0:
                matched_anchor = anchors_for_assigner[i][a['anchor_idx']]
                pos = raw[a['img_idx'], a['anchor_idx'],
                          a['grid_y'], a['grid_x']]
                pos_xy = pos[:, 0:2].sigmoid() * 2 - 0.5
                pos_wh = (pos[:, 2:4].sigmoid() * 2) ** 2 * matched_anchor
                pos_cls = pos[:, 5:5 + nc]
                pos_coeff = pos[:, 5 + nc:]

                grid_xy_int = torch.stack(
                    [a['grid_x'].to(dtype), a['grid_y'].to(dtype)], dim=-1)
                pred_cxcywh = torch.cat([pos_xy, pos_wh], dim=-1)
                target_cxcywh = torch.cat(
                    [a['gt_xy'] - grid_xy_int, a['gt_wh']], dim=-1)
                pred_xyxy = _cxcywh_to_xyxy(pred_cxcywh)
                target_xyxy = _cxcywh_to_xyxy(target_cxcywh)
                ciou = bbox_ciou(pred_xyxy, target_xyxy)
                loss_bbox = loss_bbox + (1.0 - ciou).mean()

                obj_target[a['img_idx'], a['anchor_idx'],
                           a['grid_y'], a['grid_x']] = ciou.detach().clamp(0)

                if nc > 1:
                    cls_target = F.one_hot(
                        a['gt_class'], num_classes=nc).to(dtype)
                    loss_cls = loss_cls + bce(pos_cls, cls_target).mean()

                if all_masks is not None:
                    loss_mask = loss_mask + self._mask_loss_layer(
                        pos_coeff, a['gt_idx'], a['img_idx'], proto,
                        all_masks, self.strides[i], a['gt_xy'], a['gt_wh'])

            loss_obj_layer = bce(raw[..., 4], obj_target).mean()
            loss_obj = loss_obj + loss_obj_layer * self.obj_level_weights[i]

        loss_bbox = loss_bbox * self.loss_box_weight * B
        loss_obj = loss_obj * self.loss_obj_weight * B
        loss_cls = loss_cls * self.loss_cls_weight * B
        loss_mask = loss_mask * self.loss_mask_weight * B
        return dict(loss_bbox=loss_bbox, loss_obj=loss_obj,
                    loss_cls=loss_cls, loss_mask=loss_mask)

    def _mask_loss_layer(self, coeff, gt_idx, img_idx, proto, all_masks,
                         stride, gt_xy, gt_wh):
        """单层正样本 mask loss(BCE + bbox 裁剪 + 面积归一)。"""
        B, nm, Hm, Wm = proto.shape
        proto_flat = proto.view(B, nm, Hm * Wm)
        proto_sel = proto_flat[img_idx]                       # (M, nm, Hm*Wm)
        pred = torch.bmm(
            coeff.unsqueeze(1), proto_sel).view(-1, Hm, Wm)   # logits

        gt = all_masks[gt_idx].unsqueeze(1)                   # (M,1,H,W) 输入分辨率
        gt = F.interpolate(gt, size=(Hm, Wm), mode='nearest').squeeze(1)
        gt = (gt > 0.5).to(pred.dtype)

        # gt_xy/gt_wh 为该层 grid 单位 → proto 像素:× stride / mask_ratio
        ratio = stride / self.mask_ratio
        cx, cy = gt_xy[:, 0] * ratio, gt_xy[:, 1] * ratio
        bw, bh = gt_wh[:, 0] * ratio, gt_wh[:, 1] * ratio
        boxes = torch.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)
        area = (bw * bh).clamp(min=1e-6)

        loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        # 框内 BCE 之和 / 框面积 = 框内像素平均 BCE(O(1));用 sum 而非 mean
        # 避免再除以整张 proto 面积(Hm*Wm)导致 loss 被额外缩小 ~2.5e4 倍。
        return (self._crop_mask(loss, boxes).sum(dim=(1, 2)) / area).mean()

    def predict_by_feat(self, pred_maps, proto, batch_img_metas):
        B = pred_maps[0].shape[0]
        nc = self.num_classes
        na = self.num_base_priors
        nm = self.num_masks
        no = nc + 5 + nm
        device = pred_maps[0].device
        dtype = pred_maps[0].dtype
        del batch_img_metas

        featmap_sizes = [(p.shape[2], p.shape[3]) for p in pred_maps]
        anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device, dtype=dtype)
        grid_xy = self.prior_generator.grid_xy(
            featmap_sizes, device=device, dtype=dtype)

        all_xyxy, all_scores, all_coeff = [], [], []
        for i, raw_map in enumerate(pred_maps):
            Hi, Wi = raw_map.shape[2], raw_map.shape[3]
            stride = self.strides[i]
            raw = raw_map.view(B, na, no, Hi, Wi).permute(
                0, 1, 3, 4, 2).contiguous()
            sig = torch.sigmoid(raw[..., :5 + nc])
            xy = (sig[..., 0:2] * 2 - 0.5
                  + grid_xy[i].view(1, 1, Hi, Wi, 2)) * stride
            wh = (sig[..., 2:4] * 2) ** 2 \
                * anchors[i].view(1, na, Hi, Wi, 2) * stride
            obj = sig[..., 4:5]
            cls = sig[..., 5:5 + nc]
            score = obj * cls
            coeff = raw[..., 5 + nc:]                 # 系数不过 sigmoid
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            all_xyxy.append(
                torch.cat([x1y1, x2y2], dim=-1).view(B, -1, 4))
            all_scores.append(score.view(B, -1, nc))
            all_coeff.append(coeff.reshape(B, -1, nm))

        all_xyxy = torch.cat(all_xyxy, dim=1)
        all_scores = torch.cat(all_scores, dim=1)
        all_coeff = torch.cat(all_coeff, dim=1)

        Hm, Wm = proto.shape[2], proto.shape[3]
        in_h, in_w = Hm * self.mask_ratio, Wm * self.mask_ratio
        results = []
        for b in range(B):
            scores_b, labels_b = all_scores[b].max(dim=-1)
            keep = scores_b > self.score_thr
            boxes = all_xyxy[b][keep]
            sc = scores_b[keep]
            lb = labels_b[keep]
            cf = all_coeff[b][keep]
            if boxes.shape[0] == 0:
                results.append(InstanceData(
                    bboxes=torch.zeros(0, 4, device=device, dtype=dtype),
                    scores=torch.zeros(0, device=device, dtype=dtype),
                    labels=torch.zeros(0, dtype=torch.int64, device=device),
                    masks=torch.zeros(0, in_h, in_w, dtype=torch.bool,
                                      device=device)))
                continue
            keep_idx = batched_nms(boxes, sc, lb, self.nms_iou_thr)
            keep_idx = keep_idx[:self.max_per_img]
            boxes = boxes[keep_idx]
            sc = sc[keep_idx]
            lb = lb[keep_idx]
            cf = cf[keep_idx]

            proto_b = proto[b].view(nm, -1)                  # (nm, Hm*Wm)
            masks = (cf @ proto_b).sigmoid().view(-1, Hm, Wm)
            masks = F.interpolate(
                masks.unsqueeze(1), size=(in_h, in_w),
                mode='bilinear', align_corners=False).squeeze(1)
            masks = self._crop_mask(masks, boxes) > 0.5
            results.append(InstanceData(
                bboxes=boxes, scores=sc, labels=lb, masks=masks))
        return results
