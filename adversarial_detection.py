# -*- coding: utf-8 -*-
"""
对抗样本检测（简化版）

核心思想：对同一输入做多次“轻微随机扰动/增强”（如微弱高斯噪声、亮度微调、翻转），
如果模型预测在这些扰动下非常不稳定，则判为“可疑对抗样本”。

输出：
- stability_score：0~1，越高越稳定（越不像对抗样本）

用法（示例）：
  python adversarial_detection.py --index 0 --runs 6 --conf 0.5
  python adversarial_detection.py --pt "./adv_outputs/pgd/adv_0000.pt" --runs 6 --conf 0.5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from attack_utils import (
    compute_iou,
    ensure_tensor_01,
    filter_pred,
    load_fasterrcnn_coco,
    load_voc2007_trainval_dataset,
)


@torch.no_grad()
def infer(model: torch.nn.Module, img: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    return model([img])[0]


def jitter_variants(
    img_01: torch.Tensor,
    runs: int,
    sigma: float,
    brightness: float,
) -> List[torch.Tensor]:
    """
    生成轻微扰动版本（不改变语义）：
    - 加微弱高斯噪声
    - 亮度微调（乘性）
    - 小概率水平翻转（为了让检测更敏感，默认不翻转；这里只加开关但默认0）
    """
    xs: List[torch.Tensor] = []
    for _ in range(int(runs)):
        x = img_01.detach().clone()
        if sigma > 0:
            x = x + torch.randn_like(x) * float(sigma)
        if brightness > 0:
            # factor in [1-brightness, 1+brightness]
            f = (1.0 - float(brightness)) + (2.0 * float(brightness)) * torch.rand((), device=x.device)
            x = x * f
        x = ensure_tensor_01(x)
        xs.append(x)
    return xs


@dataclass(frozen=True)
class DetItem:
    box: Tuple[float, float, float, float]
    label: int


def pred_to_items(pred: Dict[str, torch.Tensor], conf: float) -> List[DetItem]:
    fp = filter_pred(pred, conf_thresh=conf)
    items: List[DetItem] = []
    for box, label in zip(fp["boxes"].cpu(), fp["labels"].cpu()):
        b = tuple(float(v) for v in box.tolist())
        items.append(DetItem(box=b, label=int(label)))
    return items


def match_score(a: List[DetItem], b: List[DetItem], iou_thresh: float) -> float:
    """
    简单集合匹配得分：对于 a 中每个 item，若 b 中存在同 label 且 IoU>=阈值，则算匹配。
    返回 matched / max(1, len(a))，范围 0~1。
    """
    if len(a) == 0:
        return 1.0 if len(b) == 0 else 0.0
    used = [False] * len(b)
    matched = 0
    for ia in a:
        best_j = -1
        best_iou = 0.0
        for j, ib in enumerate(b):
            if used[j]:
                continue
            if ia.label != ib.label:
                continue
            iou = compute_iou(ia.box, ib.box)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            matched += 1
    return matched / max(1, len(a))


def stability_score(
    model: torch.nn.Module,
    img_01: torch.Tensor,
    runs: int,
    conf: float,
    iou_thresh: float,
    sigma: float,
    brightness: float,
) -> float:
    """
    以“原图预测”为基准，计算多次 jitter 后预测与基准的平均匹配得分。
    """
    base_pred = infer(model, img_01)
    base_items = pred_to_items(base_pred, conf=conf)
    xs = jitter_variants(img_01, runs=runs, sigma=sigma, brightness=brightness)
    scores: List[float] = []
    for x in xs:
        p = infer(model, x)
        items = pred_to_items(p, conf=conf)
        scores.append(match_score(base_items, items, iou_thresh=iou_thresh))
    if len(scores) == 0:
        return 1.0
    return float(sum(scores) / len(scores))


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--index", type=int, help="VOC trainval 的样本索引")
    src.add_argument("--pt", type=str, help="从对抗样本 .pt 文件读取（如 adv_outputs/**/adv_0000.pt）")

    p.add_argument("--runs", type=int, default=6)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=0.02, help="检测阶段 jitter 的高斯噪声强度")
    p.add_argument("--brightness", type=float, default=0.03, help="检测阶段亮度随机范围")
    p.add_argument("--threshold", type=float, default=0.55, help="稳定性阈值：低于该值判为可疑")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    bundle = load_fasterrcnn_coco(device)
    model = bundle.model

    if args.index is not None:
        ds = load_voc2007_trainval_dataset()
        img, _ = ds[int(args.index)]
        img = img.to(device)
        name = f"voc_index={int(args.index)}"
    else:
        obj = torch.load(args.pt, map_location="cpu")
        if "adv" not in obj:
            raise ValueError("pt 文件中未找到 key='adv' 的张量。")
        img = obj["adv"].to(device)
        name = f"pt={args.pt}"

    s = stability_score(
        model=model,
        img_01=img,
        runs=args.runs,
        conf=args.conf,
        iou_thresh=args.iou,
        sigma=args.sigma,
        brightness=args.brightness,
    )
    verdict = "可疑（可能是对抗样本）" if s < args.threshold else "正常（较稳定）"
    print(f"[{name}] stability_score={s:.4f} threshold={args.threshold:.4f} => {verdict}")


if __name__ == "__main__":
    main()

