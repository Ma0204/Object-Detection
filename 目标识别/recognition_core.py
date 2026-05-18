# -*- coding: utf-8 -*-
"""
「场景里有什么、每类几个」——只暴露 **类别→数量** 的表，不把画框当作对外结果。

VOC 标注在磁盘上仍是按实例存的；这里在内部读入后，只汇总成 **真值类别计数表**。
预测侧：底层仍用同一套前向，在内部把高分候选聚成 **预测类别计数表**（不返回框给用户）。
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import torch

from recognition_backend import ensure_project_root_on_path


def class_vocabulary() -> Dict[str, int]:
    ensure_project_root_on_path()
    from attack_utils import VOC_CLASSES

    return {name: idx for idx, name in enumerate(VOC_CLASSES)}


def truth_table_from_voc_target(target) -> Dict[int, int]:
    """真值：每张图一个「类 id → 实例数」字典（不含 background）。"""
    ensure_project_root_on_path()
    from attack_utils import voc_target_to_boxes_and_labels

    _, names = voc_target_to_boxes_and_labels(target)
    vocab = class_vocabulary()
    c: Counter[int] = Counter()
    for name in names:
        if name not in vocab:
            continue
        lid = int(vocab[name])
        if lid <= 0:
            continue
        c[lid] += 1
    return dict(c)


def predicted_table_from_scene(
    model: torch.nn.Module,
    scene_rgb01: torch.Tensor,
    *,
    score_cutoff: float,
    image_side: int,
) -> Dict[int, int]:
    """对当前场景图做一次前向，得到「类 id → 预测实例数」（内部用框做聚合，对外只有表）。"""
    ensure_project_root_on_path()
    from attack_utils import filter_pred, infer_yolo

    raw = infer_yolo(model, scene_rgb01, imgsz=int(image_side))
    slim = filter_pred(raw, conf_thresh=float(score_cutoff))
    labels = slim["labels"].cpu().tolist()
    c: Counter[int] = Counter()
    for lid in labels:
        i = int(lid)
        if i <= 0:
            continue
        c[i] += 1
    return dict(c)


def tables_match(truth: Dict[int, int], hypothesis: Dict[int, int]) -> bool:
    keys = set(truth) | set(hypothesis)
    for k in keys:
        if int(truth.get(k, 0)) != int(hypothesis.get(k, 0)):
            return False
    return True


def unpack_voc_scene(dataset, index: int, device: torch.device) -> Tuple[torch.Tensor, object, Dict[int, int]]:
    """返回 (图像张量, 原始 target, 真值表)。无前景目标时真值表为空 dict。"""
    img_chw, target = dataset[index]
    img_chw = img_chw.to(device)
    tt = truth_table_from_voc_target(target)
    return img_chw, target, tt
