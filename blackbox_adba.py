# -*- coding: utf-8 -*-
"""
黑盒攻击：ADBA（Approximation Decision Boundary Approach）— YOLO + VOC 检测适配。

论文：Feiyang Wang, Xingquan Zuo, Hai Huang, Gang Chen,
「ADBA: Approximation decision boundary approach for black-box adversarial attacks」,
AAAI 2024. 参考代码：https://github.com/BUPTAIOC/ADBA

与原文对照（写对的部分 / 差异）：
- 一致思想：在「两个候选扰动方向」之间，用**少量前向查询**比较哪一侧更易使目标变坏，
  避免对决策边界做完整二分（省查询）；再在优选方向上沿 L∞ 球做一步更新。
- ADBA-md 原文：用决策边界距离的**分布与中位数**定义近似决策边界（ADB），再比较两候选方向
  （约 4 次查询量级，见 AAAI 论文与 BUPTAIOC/ADBA）。
  本检测版：**未**估计边界距离分布或中位数；对方向 d1、d2 各做对称探测 f(x±αd)，用
  min(f(x+αd), f(x−αd)) 作启发式标量以二选一（4 次查询），再 2 次查询做 ±α 线搜索一步。
  属于「少查询方向比较 + 线搜索」的松散类比，**非** ADBA-md 统计定义上的忠实实现。
  若需与官方逐行一致，请 fork BUPTAIOC/ADBA 并将 oracle 换为本文的 _f()。

检测化（与「多分类拼接」的关系）：检测共享骨干 + NMS，各框耦合，并非独立分类器之和；但每个
GT 邻域仍有「检出/漏检」过渡。--gt_focus_expand 将扰动方向与随机起点限制在 GT 外扩区域内，
使查询集中在目标附近，便于化用 ADBA 的「方向比较 + 小步更新」框架。更深：可按 GT 序做短
ADBA 或 per-GT 硬判定 oracle，逐步逼近 ADBA-md 精神。
可选 `--adba_aux_fp` / `--adba_aux_misclass`：在原有「压制匹配框」项上叠加促 FP、促错类代理，
与漏检向目标加权组合（默认权重 0 行为与原先一致）。
`--adba_per_gt_local`：全图前向前提下，≥2 个有效 GT 时等价于按 GT 串联（每段仅该 GT 邻域掩膜 + x_init
链式合并）；单 GT 时仅在该 GT 邻域内扰动（仍用整图 hybrid ref），优于「裁 patch 再拼」。
- **按 GT 串联模式**（`--adba_sequential_gt`）：对每个 GT 仅用该框为 ref、邻域掩膜下跑一段 ADBA，
  输出作为下一段 `x_init`，实现「先定位再在框内/邻域化用方向比较」。总查询随 GT 数线性增加。
- **`--adba_per_gt_local`**：不裁 patch；始终对整图 `infer`，多 GT 时自动启用上述串联逻辑（可与 sequential
  叠加）；单 GT 时仅在单框外扩邻域内扰动、仍用完整 hybrid ref 与 `--steps`。

注意：论文在分类上的高成功率**不能**等价推出 VOC+YOLO+GT-ASR 也能到 90%；检测耦合 NMS、
多尺度与评估定义，上界需实验重新标定。
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from attack_utils import (
    build_yolo_voc_model,
    infer_yolo,
    load_voc2007_dataset,
    voc_target_to_boxes_and_labels,
    compute_iou,
    filter_pred,
    build_blackbox_attack_refs,
    yolo_matched_suppression_objective,
    yolo_spurious_promotion_objective,
    yolo_wrongclass_at_gt_objective,
    project_linf_01,
    spatial_mask_from_boxes_xyxy,
)
from evaluation_metrics import compute_model_metrics, compute_perturbation_metrics

VOC_CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}


@torch.no_grad()
def infer(model, img: torch.Tensor) -> Dict[str, torch.Tensor]:
    return infer_yolo(model, img)


def _f(
    model,
    x: torch.Tensor,
    ref_boxes: torch.Tensor,
    ref_labels: torch.Tensor,
    ref_scores: torch.Tensor,
    topk: int,
    iou_match: float,
    objective_conf: float,
    *,
    aux_gt_boxes: Optional[torch.Tensor] = None,
    aux_gt_label_ids: Optional[torch.Tensor] = None,
    w_aux_fp: float = 0.0,
    w_aux_misclass: float = 0.0,
    topk_aux: int = 64,
) -> float:
    pred = infer(model, x)
    loss = yolo_matched_suppression_objective(
        pred, ref_boxes, ref_labels, ref_scores, topk, iou_match, eval_conf=float(objective_conf)
    )
    if (
        aux_gt_boxes is not None
        and aux_gt_label_ids is not None
        and int(aux_gt_boxes.shape[0]) > 0
        and (float(w_aux_fp) > 0.0 or float(w_aux_misclass) > 0.0)
    ):
        if float(w_aux_fp) > 0.0:
            loss = loss + float(w_aux_fp) * yolo_spurious_promotion_objective(
                pred, aux_gt_boxes, aux_gt_label_ids, float(iou_match), int(topk_aux)
            )
        if float(w_aux_misclass) > 0.0:
            loss = loss + float(w_aux_misclass) * yolo_wrongclass_at_gt_objective(
                pred, aux_gt_boxes, aux_gt_label_ids, float(iou_match)
            )
    return float(loss.item())


def _rademacher_pair_decorr(
    x0: torch.Tensor, spatial_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """两个符号方向；若给定 spatial_mask [1,H,W]，仅在目标区域采样扰动方向。"""
    m3: Optional[torch.Tensor] = None
    if spatial_mask is not None and spatial_mask.shape[1:] == x0.shape[1:]:
        m3 = spatial_mask.expand_as(x0)

    def _sample_u() -> torch.Tensor:
        if m3 is None:
            return torch.sign(torch.randn_like(x0))
        noise = torch.randn_like(x0) * m3
        u = torch.sign(noise)
        if float(u.abs().sum()) < 1e-3:
            return torch.sign(torch.randn_like(x0))
        return u

    u1 = _sample_u()
    v1 = u1.reshape(-1).float()
    gn = torch.randn_like(x0.reshape(-1), dtype=torch.float32, device=x0.device)
    if m3 is not None:
        gn = gn * m3.reshape(-1)
    dot = (gn * v1).sum()
    denom = (v1 * v1).sum().clamp(min=1.0)
    orth = gn - (dot / denom) * v1
    if float(orth.abs().sum()) < 1e-5:
        orth = torch.randn_like(orth) * (m3.reshape(-1) if m3 is not None else 1.0)
    u2 = torch.sign(orth.reshape_as(x0))
    if m3 is not None:
        u2 = u2 * m3
        if float(u2.abs().sum()) < 1e-3:
            u2 = _sample_u()
    return u1, u2


def _ternary_minimize_on_ray(
    model,
    x0: torch.Tensor,
    x_base: torch.Tensor,
    du: torch.Tensor,
    lo_lam: float,
    hi_lam: float,
    eps: float,
    ref_boxes: torch.Tensor,
    ref_labels: torch.Tensor,
    ref_scores: torch.Tensor,
    topk: int,
    iou_match: float,
    objective_conf: float,
    inner_iters: int,
    fq,
) -> torch.Tensor:
    """沿 x_base + λ·du（再投影到 L∞ 球）对连续目标 fq 做短三分搜索；借鉴官方在双成功方向上的 ADB 收缩思想。"""
    best_x = project_linf_01(x_base, x0, eps)
    best_v = fq(best_x)
    if best_v is None:
        return best_x
    lo, hi = float(lo_lam), float(hi_lam)
    for _ in range(max(0, int(inner_iters))):
        if hi - lo < 1e-7:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        x1 = project_linf_01(x_base + m1 * du, x0, eps)
        x2 = project_linf_01(x_base + m2 * du, x0, eps)
        v1 = fq(x1)
        if v1 is None:
            return best_x
        v2 = fq(x2)
        if v2 is None:
            return best_x
        if v1 <= v2:
            hi = m2
            if v1 < best_v:
                best_v, best_x = v1, x1
        else:
            lo = m1
            if v2 < best_v:
                best_v, best_x = v2, x2
    return project_linf_01(best_x, x0, eps)


def adba_attack(
    model,
    x0: torch.Tensor,
    eps: float,
    steps: int,
    probe_scale: float,
    alpha: float,
    ref_boxes: torch.Tensor,
    ref_labels: torch.Tensor,
    ref_scores: torch.Tensor,
    topk: int,
    iou_match: float,
    random_start: bool,
    stop_ratio: float,
    objective_conf: float,
    spatial_mask: Optional[torch.Tensor] = None,
    x_init: Optional[torch.Tensor] = None,
    *,
    adb_ray_refine_iters: int = 0,
    adb_improve_only: bool = True,
    query_used: Optional[List[int]] = None,
    query_budget: int = 0,
    aux_gt_boxes: Optional[torch.Tensor] = None,
    aux_gt_label_ids: Optional[torch.Tensor] = None,
    w_aux_fp: float = 0.0,
    w_aux_misclass: float = 0.0,
    topk_aux: int = 64,
) -> torch.Tensor:
    """ADBA 风格方向比较 + L∞ 投影更新（见模块 docstring 与论文差异说明）。

    x_init：非空时从「已累积的对抗图」出发（须已满足相对 x0 的 L∞ 约束），用于按 GT 串联攻击；
    再结合 spatial_mask 即「先定位再在框邻域内做方向比较」。

    adb_ray_refine_iters：>0 时在选定 du 上沿 λ 做短三分搜索（借鉴官方 ADB 在双成功子方向上的区间收缩）；
    adb_improve_only：为真时仅当线搜索能严格降低 _f 才更新 x。
    query_used / query_budget：可选全局查询计数，每计一次对应一次 _f（与官方 budget 概念对齐）。

    aux_gt_* + w_aux_fp / w_aux_misclass：在压制项之外叠加「促 FP」「促 GT 处错类」代理（默认权重 0 即原行为）。
    """
    def fq(xx: torch.Tensor):
        if int(query_budget) > 0 and query_used is not None and query_used[0] >= int(query_budget):
            return None
        if query_used is not None:
            query_used[0] += 1
        return _f(
            model,
            xx,
            ref_boxes,
            ref_labels,
            ref_scores,
            topk,
            iou_match,
            objective_conf,
            aux_gt_boxes=aux_gt_boxes,
            aux_gt_label_ids=aux_gt_label_ids,
            w_aux_fp=w_aux_fp,
            w_aux_misclass=w_aux_misclass,
            topk_aux=topk_aux,
        )

    if x_init is not None:
        x = project_linf_01(x_init.detach().clone(), x0, eps)
    else:
        x = x0.detach().clone()
    if random_start:
        rs = torch.rand_like(x) * 2.0 - 1.0
        if spatial_mask is not None and spatial_mask.shape[1:] == x0.shape[1:]:
            rs = rs * spatial_mask.expand_as(x0)
        x = x + rs * float(eps)
        x = project_linf_01(x, x0, eps)
    init_v = fq(x)
    if init_v is None:
        return x.detach()
    init_f = init_v
    n_steps = max(1, int(steps))
    sr = float(stop_ratio)
    n_ref = max(0, int(adb_ray_refine_iters))
    for si in range(n_steps):
        if int(query_budget) > 0 and query_used is not None and query_used[0] >= int(query_budget):
            break
        prog = float(si) / float(max(n_steps - 1, 1))
        scale = float(eps) * float(probe_scale) * (1.0 + 0.38 * (1.0 - prog))
        alpha_t = float(alpha) * (1.08 - 0.28 * prog)
        alpha_t = max(float(alpha) * 0.72, min(float(alpha) * 1.08, alpha_t))

        u1, u2 = _rademacher_pair_decorr(x0, spatial_mask)
        x_p1 = project_linf_01(x + scale * u1, x0, eps)
        x_m1 = project_linf_01(x - scale * u1, x0, eps)
        x_p2 = project_linf_01(x + scale * u2, x0, eps)
        x_m2 = project_linf_01(x - scale * u2, x0, eps)
        f_p1 = fq(x_p1)
        if f_p1 is None:
            break
        f_m1 = fq(x_m1)
        if f_m1 is None:
            break
        f_p2 = fq(x_p2)
        if f_p2 is None:
            break
        f_m2 = fq(x_m2)
        if f_m2 is None:
            break
        m1 = min(f_p1, f_m1)
        m2 = min(f_p2, f_m2)
        if m1 <= m2:
            du = u1 if f_p1 <= f_m1 else -u1
        else:
            du = u2 if f_p2 <= f_m2 else -u2

        fx_here: Optional[float] = None
        if adb_improve_only or n_ref > 0:
            fx_here = fq(x)
            if fx_here is None:
                break

        x_a = project_linf_01(x + alpha_t * du, x0, eps)
        x_b = project_linf_01(x - alpha_t * du, x0, eps)
        fa = fq(x_a)
        if fa is None:
            break
        fb = fq(x_b)
        if fb is None:
            break

        if fx_here is not None and adb_improve_only and min(fa, fb) >= fx_here:
            if sr > 0.0 and init_f > 1e-8:
                cur = fq(x)
                if cur is None:
                    break
                if cur <= init_f * sr:
                    break
            continue

        if n_ref > 0 and fx_here is not None and fa < fx_here and fb < fx_here:
            x = _ternary_minimize_on_ray(
                model,
                x0,
                x,
                du,
                -alpha_t,
                alpha_t,
                eps,
                ref_boxes,
                ref_labels,
                ref_scores,
                topk,
                iou_match,
                objective_conf,
                n_ref,
                fq,
            )
        elif n_ref > 0 and fx_here is not None and fa < fx_here:
            x = _ternary_minimize_on_ray(
                model,
                x0,
                x,
                du,
                0.0,
                alpha_t,
                eps,
                ref_boxes,
                ref_labels,
                ref_scores,
                topk,
                iou_match,
                objective_conf,
                n_ref,
                fq,
            )
        elif n_ref > 0 and fx_here is not None and fb < fx_here:
            x = _ternary_minimize_on_ray(
                model,
                x0,
                x,
                du,
                -alpha_t,
                0.0,
                eps,
                ref_boxes,
                ref_labels,
                ref_scores,
                topk,
                iou_match,
                objective_conf,
                n_ref,
                fq,
            )
        else:
            x = x_a if fa <= fb else x_b
        x = project_linf_01(x, x0, eps)
        if sr > 0.0 and init_f > 1e-8:
            cur = fq(x)
            if cur is None:
                break
            if cur <= init_f * sr:
                break
    return x.detach()


def _gt_aux_label_tensors(
    gt_boxes: torch.Tensor, gt_labels: List[str], device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """构造辅助目标用的 GT 框与类别 id（与 VOC_CLASS_TO_ID 一致）；无有效 GT 时返回 (None, None)。"""
    if gt_boxes.numel() == 0:
        return None, None
    gb_list: List[torch.Tensor] = []
    gl_list: List[int] = []
    for i, name in enumerate(gt_labels):
        if i >= int(gt_boxes.shape[0]):
            break
        if name not in VOC_CLASS_TO_ID:
            continue
        lid = int(VOC_CLASS_TO_ID[name])
        if lid <= 0:
            continue
        gb_list.append(gt_boxes[i].to(device=device, dtype=torch.float32))
        gl_list.append(lid)
    if not gb_list:
        return None, None
    return torch.stack(gb_list, dim=0), torch.tensor(gl_list, device=device, dtype=torch.long)


def _valid_gt_indices(gt_labels: List[str]) -> List[int]:
    """VOC 中参与检测攻击的 GT 下标（跳过 background / 未知类）。"""
    out: List[int] = []
    for i, name in enumerate(gt_labels):
        if name not in VOC_CLASS_TO_ID:
            continue
        if int(VOC_CLASS_TO_ID[name]) <= 0:
            continue
        out.append(i)
    return out


def count_metrics(
    gt_boxes: torch.Tensor,
    gt_labels: List[str],
    pred_boxes: torch.Tensor,
    pred_labels: List[int],
    iou_thresh: float,
) -> Tuple[int, int, int, int]:
    tp_here = fn_here = matched_here = 0
    for g_box, g_name in zip(gt_boxes, gt_labels):
        if g_name not in VOC_CLASS_TO_ID:
            continue
        gt_lid = VOC_CLASS_TO_ID[g_name]
        g_box_list = g_box.tolist()
        found = False
        for p_box, p_lid in zip(pred_boxes, pred_labels):
            if int(p_lid) == int(gt_lid) and compute_iou(g_box_list, p_box.tolist()) >= iou_thresh:
                found = True
                break
        if found:
            tp_here += 1
            matched_here += 1
        else:
            fn_here += 1
    fp_here = max(0, len(pred_labels) - matched_here)
    return tp_here, fp_here, fn_here, matched_here


def evaluate_once(model, dataset, args, run_id: int) -> Dict:
    n = min(int(args.num_eval), len(dataset))
    idxs = random.sample(range(len(dataset)), n)
    log_every = max(1, int(args.log_every))
    t0 = time.time()
    print(f"  [ADBA 第{run_id}次] 评估 {n} 张；每 {log_every} 张打印进度。", flush=True)

    c_tp = c_fp = c_fn = 0
    a_tp = a_fp = a_fn = 0
    gt_detected_clean = gt_success_count = 0
    img_detected_clean = img_success_count = 0
    clean_imgs, adv_imgs = [], []

    for step, idx in enumerate(idxs):
        img_tensor, target = dataset[idx]
        img = img_tensor.to(args.device)
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)

        pred_clean = infer(model, img)
        fp_clean = filter_pred(pred_clean, conf_thresh=args.conf)
        fp_ref = filter_pred(pred_clean, conf_thresh=args.ref_conf)
        boxes_c = fp_clean["boxes"].cpu()
        labels_c = fp_clean["labels"].cpu().tolist()

        ref_boxes, ref_labels, ref_scores = build_blackbox_attack_refs(
            fp_ref,
            gt_boxes,
            gt_labels,
            img.device,
            args.ref_topk,
            args.ref_mode,
            hybrid_suppress_iou=float(args.iou),
        )

        oc = float(args.objective_conf) if args.objective_conf is not None else float(args.conf)
        _, H, W = img.shape
        aux_b, aux_lid = _gt_aux_label_tensors(gt_boxes, gt_labels, img.device)
        wfp = float(args.adba_aux_fp)
        wmc = float(args.adba_aux_misclass)
        tk_aux = max(1, int(args.adba_aux_topk))

        qu: Optional[List[int]] = [0] if int(args.max_queries_per_image) > 0 else None
        qmax = int(args.max_queries_per_image)
        adb_ref = max(0, int(args.adb_ray_refine_iters))
        adb_imp = not bool(args.adb_allow_uphill)

        adv = img.detach().clone()
        best_f = float("inf")

        vix = _valid_gt_indices(gt_labels)
        use_sequential_chain = bool(args.adba_sequential_gt) or (
            bool(args.adba_per_gt_local) and len(vix) >= 2
        )

        if use_sequential_chain:
            if len(vix) == 0 or gt_boxes.numel() == 0:
                spatial_mask_sg: Optional[torch.Tensor] = None
                if float(args.gt_focus_expand) > 0.0 and gt_boxes.numel() > 0:
                    spatial_mask_sg = spatial_mask_from_boxes_xyxy(
                        gt_boxes.to(img.device),
                        H,
                        W,
                        img.device,
                        expand_frac=float(args.gt_focus_expand),
                    )
                for _ in range(max(1, int(args.attack_restarts))):
                    cand = adba_attack(
                        model,
                        img,
                        args.eps,
                        args.steps,
                        args.probe_scale,
                        args.alpha,
                        ref_boxes,
                        ref_labels,
                        ref_scores,
                        args.topk,
                        args.iou_match,
                        not bool(args.no_random_start),
                        args.stop_ratio,
                        oc,
                        spatial_mask_sg,
                        None,
                        adb_ray_refine_iters=adb_ref,
                        adb_improve_only=adb_imp,
                        query_used=qu,
                        query_budget=qmax,
                        aux_gt_boxes=aux_b,
                        aux_gt_label_ids=aux_lid,
                        w_aux_fp=wfp,
                        w_aux_misclass=wmc,
                        topk_aux=tk_aux,
                    )
                    fv = _f(
                        model,
                        cand,
                        ref_boxes,
                        ref_labels,
                        ref_scores,
                        args.topk,
                        args.iou_match,
                        oc,
                        aux_gt_boxes=aux_b,
                        aux_gt_label_ids=aux_lid,
                        w_aux_fp=wfp,
                        w_aux_misclass=wmc,
                        topk_aux=tk_aux,
                    )
                    if fv < best_f:
                        best_f = fv
                        adv = cand
            else:
                expand_m = float(args.gt_focus_expand) if float(args.gt_focus_expand) > 0.0 else 0.28
                topk_sg = max(1, min(int(args.topk), 32))
                # per_gt_local：避免「多 GT 串联时每段只用 per_gt_steps」弱于用户为整图设的 --steps
                if bool(args.adba_per_gt_local):
                    per_n = max(4, int(args.per_gt_steps), int(args.steps))
                else:
                    per_n = max(4, int(args.per_gt_steps))
                x_chain = img.detach().clone()
                rs_one = torch.ones((1,), device=img.device, dtype=torch.float32)
                for j, gi in enumerate(vix):
                    rb = gt_boxes[gi : gi + 1].to(device=img.device, dtype=torch.float32)
                    lid = int(VOC_CLASS_TO_ID[gt_labels[gi]])
                    rl = torch.tensor([lid], device=img.device, dtype=torch.long)
                    mask_g = spatial_mask_from_boxes_xyxy(rb, H, W, img.device, expand_frac=expand_m)
                    stage_best = x_chain
                    stage_best_f = float("inf")
                    rs_here = not bool(args.no_random_start) and (j == 0)
                    for _ in range(max(1, int(args.attack_restarts))):
                        cand = adba_attack(
                            model,
                            img,
                            args.eps,
                            per_n,
                            args.probe_scale,
                            args.alpha,
                            rb,
                            rl,
                            rs_one,
                            topk_sg,
                            args.iou_match,
                            rs_here,
                            args.stop_ratio,
                            oc,
                            mask_g,
                            x_chain,
                            adb_ray_refine_iters=adb_ref,
                            adb_improve_only=adb_imp,
                            query_used=qu,
                            query_budget=qmax,
                            aux_gt_boxes=aux_b,
                            aux_gt_label_ids=aux_lid,
                            w_aux_fp=wfp,
                            w_aux_misclass=wmc,
                            topk_aux=tk_aux,
                        )
                        fv = _f(
                            model,
                            cand,
                            rb,
                            rl,
                            rs_one,
                            topk_sg,
                            args.iou_match,
                            oc,
                            aux_gt_boxes=aux_b,
                            aux_gt_label_ids=aux_lid,
                            w_aux_fp=wfp,
                            w_aux_misclass=wmc,
                            topk_aux=tk_aux,
                        )
                        if fv < stage_best_f:
                            stage_best_f = fv
                            stage_best = cand
                    x_chain = project_linf_01(stage_best, img, args.eps)
                adv = x_chain
        elif bool(args.adba_per_gt_local) and len(vix) == 1:
            expand_m = float(args.gt_focus_expand) if float(args.gt_focus_expand) > 0.0 else 0.28
            gi = vix[0]
            rb = gt_boxes[gi : gi + 1].to(device=img.device, dtype=torch.float32)
            mask_g = spatial_mask_from_boxes_xyxy(rb, H, W, img.device, expand_frac=expand_m)
            for _ in range(max(1, int(args.attack_restarts))):
                cand = adba_attack(
                    model,
                    img,
                    args.eps,
                    args.steps,
                    args.probe_scale,
                    args.alpha,
                    ref_boxes,
                    ref_labels,
                    ref_scores,
                    args.topk,
                    args.iou_match,
                    not bool(args.no_random_start),
                    args.stop_ratio,
                    oc,
                    mask_g,
                    None,
                    adb_ray_refine_iters=adb_ref,
                    adb_improve_only=adb_imp,
                    query_used=qu,
                    query_budget=qmax,
                    aux_gt_boxes=aux_b,
                    aux_gt_label_ids=aux_lid,
                    w_aux_fp=wfp,
                    w_aux_misclass=wmc,
                    topk_aux=tk_aux,
                )
                fv = _f(
                    model,
                    cand,
                    ref_boxes,
                    ref_labels,
                    ref_scores,
                    args.topk,
                    args.iou_match,
                    oc,
                    aux_gt_boxes=aux_b,
                    aux_gt_label_ids=aux_lid,
                    w_aux_fp=wfp,
                    w_aux_misclass=wmc,
                    topk_aux=tk_aux,
                )
                if fv < best_f:
                    best_f = fv
                    adv = cand
        else:
            spatial_mask: Optional[torch.Tensor] = None
            if float(args.gt_focus_expand) > 0.0 and gt_boxes.numel() > 0:
                spatial_mask = spatial_mask_from_boxes_xyxy(
                    gt_boxes.to(img.device),
                    H,
                    W,
                    img.device,
                    expand_frac=float(args.gt_focus_expand),
                )

            for _ in range(max(1, int(args.attack_restarts))):
                cand = adba_attack(
                    model,
                    img,
                    args.eps,
                    args.steps,
                    args.probe_scale,
                    args.alpha,
                    ref_boxes,
                    ref_labels,
                    ref_scores,
                    args.topk,
                    args.iou_match,
                    not bool(args.no_random_start),
                    args.stop_ratio,
                    oc,
                    spatial_mask,
                    None,
                    adb_ray_refine_iters=adb_ref,
                    adb_improve_only=adb_imp,
                    query_used=qu,
                    query_budget=qmax,
                    aux_gt_boxes=aux_b,
                    aux_gt_label_ids=aux_lid,
                    w_aux_fp=wfp,
                    w_aux_misclass=wmc,
                    topk_aux=tk_aux,
                )
                fv = _f(
                    model,
                    cand,
                    ref_boxes,
                    ref_labels,
                    ref_scores,
                    args.topk,
                    args.iou_match,
                    oc,
                    aux_gt_boxes=aux_b,
                    aux_gt_label_ids=aux_lid,
                    w_aux_fp=wfp,
                    w_aux_misclass=wmc,
                    topk_aux=tk_aux,
                )
                if fv < best_f:
                    best_f = fv
                    adv = cand
        pred_adv = infer(model, adv)
        fp_adv = filter_pred(pred_adv, conf_thresh=args.conf)
        boxes_a = fp_adv["boxes"].cpu()
        labels_a = fp_adv["labels"].cpu().tolist()

        ctp, cfp, cfn, cm = count_metrics(gt_boxes, gt_labels, boxes_c, labels_c, args.iou)
        atp, afp, afn, am = count_metrics(gt_boxes, gt_labels, boxes_a, labels_a, args.iou)
        c_tp += ctp
        c_fp += cfp
        c_fn += cfn
        a_tp += atp
        a_fp += afp
        a_fn += afn

        for g_box, g_name in zip(gt_boxes, gt_labels):
            if g_name not in VOC_CLASS_TO_ID:
                continue
            gt_lid = VOC_CLASS_TO_ID[g_name]
            g_box_list = g_box.tolist()
            c_hit = any(
                int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= args.iou
                for pb, pl in zip(boxes_c, labels_c)
            )
            if c_hit:
                gt_detected_clean += 1
                a_hit = any(
                    int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= args.iou
                    for pb, pl in zip(boxes_a, labels_a)
                )
                if not a_hit:
                    gt_success_count += 1

        if cm > 0 and gt_boxes.size(0) > 0:
            img_detected_clean += 1
            if am == 0:
                img_success_count += 1

        if len(clean_imgs) < 200:
            clean_imgs.append(img.detach().cpu())
            adv_imgs.append(adv.detach().cpu())

        if (step + 1) % log_every == 0 or (step + 1) == n:
            base = int(gt_detected_clean)
            succ = int(gt_success_count)
            if base <= 0:
                asr_s = "n/a(尚无干净检出GT)"
            else:
                asr_s = f"{(succ / float(base)) * 100.0:.2f}%"
            print(
                f"  [ADBA] {step+1}/{n} | GT-ASR={asr_s} (succ/base={succ}/{base}) | "
                f"{(time.time()-t0)/60:.1f}m",
                flush=True,
            )

    clean_metrics = compute_model_metrics(c_tp, c_fp, c_fn, 0)
    adv_metrics = compute_model_metrics(a_tp, a_fp, a_fn, 0)
    gt_asr = gt_success_count / max(1, gt_detected_clean)
    img_asr = img_success_count / max(1, img_detected_clean)
    pert = compute_perturbation_metrics(clean_imgs, adv_imgs, gt_asr)
    return {
        "clean": clean_metrics,
        "adv": adv_metrics,
        "gt_asr": gt_asr,
        "img_asr": img_asr,
        "gt_succ": gt_success_count,
        "gt_base": gt_detected_clean,
        "img_succ": img_success_count,
        "img_base": img_detected_clean,
        "pert": pert,
    }


def save_results(args, all_r: List[Dict]) -> None:
    asr = np.array([r["gt_asr"] for r in all_r], dtype=np.float64)
    result_file = os.path.join(args.outdir, "adba_results.txt")
    os.makedirs(args.outdir, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("黑盒攻击评估 - ADBA (YOLO 检测适配)\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"eps={args.eps} steps={args.steps} probe_scale={args.probe_scale} alpha={args.alpha} "
            f"restarts={args.attack_restarts} ref_mode={args.ref_mode} ref_conf={args.ref_conf}\n"
        )
        f.write(
            f"adba_sequential_gt={bool(args.adba_sequential_gt)} adba_per_gt_local={bool(args.adba_per_gt_local)} "
            f"per_gt_steps={args.per_gt_steps} "
            f"gt_focus_expand={args.gt_focus_expand} adb_ray_refine={args.adb_ray_refine_iters} "
            f"max_queries={args.max_queries_per_image}\n"
        )
        f.write(
            f"adba_aux_fp={args.adba_aux_fp} adba_aux_misclass={args.adba_aux_misclass} "
            f"adba_aux_topk={args.adba_aux_topk}\n"
        )
        f.write(
            f"num_eval={args.num_eval} conf={args.conf} iou={args.iou} eval_set={args.eval_set}\n\n"
        )
        for i, r in enumerate(all_r, 1):
            f.write(f"Run {i}: GT-ASR={r['gt_asr']*100:.2f}% 图像-ASR={r['img_asr']*100:.2f}%\n")
        f.write(f"\n均值 GT-ASR: {asr.mean()*100:.2f}%  满足>85%: {asr.mean() > 0.85}\n")
    print(f"结果已保存: {result_file}")


def main() -> None:
    p = argparse.ArgumentParser(description="ADBA 黑盒攻击（YOLO VOC）")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=160, help="每图迭代步数（提 GT-ASR 的主要旋钮之一）")
    p.add_argument("--probe_scale", type=float, default=0.33)
    p.add_argument("--alpha", type=float, default=0.019, help="线搜索基准步长，步内会随进度轻微衰减")
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--ref_topk", type=int, default=120)
    p.add_argument(
        "--ref_mode",
        type=str,
        default="hybrid",
        choices=["pred", "gt", "hybrid"],
        help="攻击参考框：hybrid=GT+未与 GT 重叠的高分预测（默认，贴近 GT-ASR）；gt=仅 GT；pred=仅干净预测",
    )
    p.add_argument(
        "--ref_conf",
        type=float,
        default=0.22,
        help="构造参考框时的低阈值（用于 hybrid/pred 分支，可并入更多框参与压制）",
    )
    p.add_argument(
        "--iou_match",
        type=float,
        default=0.5,
        help="目标函数里「预测与参考框」匹配 IoU，默认与评估 --iou 一致以便对齐 GT-ASR",
    )
    p.add_argument(
        "--stop_ratio",
        type=float,
        default=0.0,
        help=">0 时：f 相对初始值低于该比例则提前结束；默认 0 关闭早停。关闭早停时每步少 1 次前向",
    )
    p.add_argument(
        "--attack_restarts",
        type=int,
        default=5,
        help="每张图独立重复攻击次数，取代理目标 f 最小者（查询量约×该倍数）",
    )
    p.add_argument("--no_random_start", action="store_true")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument(
        "--objective_conf",
        type=float,
        default=None,
        help="黑盒目标 margin 对齐的置信度阈值；默认与 --conf 相同，可单独调低以更强压分",
    )
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--num_eval", type=int, default=200)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--load_model", type=str, required=True)
    p.add_argument("--eval_set", type=str, default="test", choices=["train", "val", "trainval", "test"])
    p.add_argument("--outdir", type=str, default="./adv_outputs/adba")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument(
        "--eps_max",
        type=float,
        default=0.05,
        help="L∞ 扰动上限（与 --eps 同时用时取 min）；可设为 0.062745(≈16/255) 等略增预算以换成功率",
    )
    p.add_argument(
        "--gt_focus_expand",
        type=float,
        default=0.0,
        help=">0 时在 GT 框外扩该比例(max(w,h))内采样扰动方向/随机起点，0 关闭（全图）。建议 0.2~0.5 试",
    )
    p.add_argument(
        "--adb_ray_refine_iters",
        type=int,
        default=2,
        help="沿选定 du 在 λ 上的短三分搜索迭代数；0 关闭。对应官方在双成功方向上的 ADB 区间收缩（连续 f 版）",
    )
    p.add_argument(
        "--adb_allow_uphill",
        action="store_true",
        help="允许 _f 不下降仍走线搜索端点（旧行为）；默认关闭，仅当能降 f 才更新",
    )
    p.add_argument(
        "--max_queries_per_image",
        type=int,
        default=0,
        help="每张图在 adba_attack 内 _f 查询上限，0 不限制（对齐官方 budget 思路）",
    )
    p.add_argument(
        "--adba_sequential_gt",
        action="store_true",
        help="按 GT 顺序串联 ADBA：每段只优化该 GT 的 ref + 邻域掩膜，链式 x_init；总查询≈ GT 数×per_gt_steps",
    )
    p.add_argument(
        "--adba_per_gt_local",
        action="store_true",
        help="全图前向 + 局部扰动：≥2 有效 GT 时自动按 GT 串联（同 sequential，每段仅该 GT 外扩邻域）；"
        "单 GT 时仅在该 GT 邻域内用 --steps 与完整 ref 优化。与 --adba_sequential_gt 可同时开（≥2 GT 时等价于开启串联）",
    )
    p.add_argument(
        "--per_gt_steps",
        type=int,
        default=48,
        help="串联分支内每 GT 段迭代步数。仅 --adba_sequential_gt 时 per_n=max(4,本参数)。"
        "若同时 --adba_per_gt_local：per_n=max(4,本参数,--steps)，避免每段弱于整图步数。",
    )
    p.add_argument(
        "--adba_aux_fp",
        type=float,
        default=0.0,
        help="与压制项相加：促假阳性代理（拉高不构成 TP 的检测分数）权重；0=仅原漏检向目标",
    )
    p.add_argument(
        "--adba_aux_misclass",
        type=float,
        default=0.0,
        help="与压制项相加：促 GT 处错类代理（IoU≥iou_match 且类≠GT 的框分数）权重；0=关闭",
    )
    p.add_argument(
        "--adba_aux_topk",
        type=int,
        default=64,
        help="促 FP 项中对伪框候选取前多少个高分参与求和",
    )
    args = p.parse_args()
    em = float(args.eps_max)
    if args.eps > em:
        args.eps = em
        print(f"eps 已截断为 eps_max={em}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("Device:", device)
    print(
        f"ADBA: ref_mode={args.ref_mode} ref_conf={args.ref_conf} steps={args.steps} "
        f"attack_restarts={args.attack_restarts} iou_match={args.iou_match} "
        f"objective_conf={(args.objective_conf if args.objective_conf is not None else args.conf)} "
        f"gt_focus_expand={args.gt_focus_expand} "
        f"sequential_gt={bool(args.adba_sequential_gt)} per_gt_local={bool(args.adba_per_gt_local)} "
        f"per_gt_steps={args.per_gt_steps} "
        f"adb_ray_refine={args.adb_ray_refine_iters} max_q={args.max_queries_per_image} "
        f"aux_fp={args.adba_aux_fp} aux_misclass={args.adba_aux_misclass} aux_topk={args.adba_aux_topk}",
        flush=True,
    )
    os.makedirs(args.outdir, exist_ok=True)
    ds = load_voc2007_dataset(args.eval_set)
    model = build_yolo_voc_model(device, weights=args.load_model)

    all_r = []
    for i in range(1, args.runs + 1):
        r = evaluate_once(model, ds, args, i)
        all_r.append(r)
        print(f"\n[ADBA 第{i}次]")
        print(
            f"  Clean: P={r['clean'].precision*100:.2f}% R={r['clean'].recall*100:.2f}% "
            f"F={r['clean'].f_score:.4f} Acc={r['clean'].accuracy*100:.2f}%"
        )
        print(
            f"  Adv:   P={r['adv'].precision*100:.2f}% R={r['adv'].recall*100:.2f}% "
            f"F={r['adv'].f_score:.4f} Acc={r['adv'].accuracy*100:.2f}%"
        )
        print(f"  攻击成功率(GT/图像): {r['gt_asr']*100:.2f}% / {r['img_asr']*100:.2f}%")
        print(
            f"  扰动: L2={r['pert'].avg_l2_distance:.4f} Linf={r['pert'].avg_linf_distance:.4f} "
            f"MSE={r['pert'].avg_mse:.6f} SSIM={r['pert'].avg_ssim:.4f} PSNR={r['pert'].avg_psnr:.2f}"
        )

    asr = np.array([r["gt_asr"] for r in all_r])
    print("\n" + "=" * 60)
    print("ADBA 汇总（YOLO 检测）")
    print("=" * 60)
    print(f"攻击成功率(GT级)均值: {asr.mean()*100:.2f}%")
    print(f"是否满足 >85%: {asr.mean() > 0.85}")
    save_results(args, all_r)


if __name__ == "__main__":
    main()
