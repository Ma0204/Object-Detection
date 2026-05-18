# -*- coding: utf-8 -*-
"""「真值类别表 vs 预测类别表」评测；黑盒扰动调用上级无梯度优化核（文件名为工程历史遗留）。"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from recognition_backend import ensure_project_root_on_path, load_module_from_parent
from recognition_core import predicted_table_from_scene, tables_match, truth_table_from_voc_target


def _infer_side(args: Any) -> int:
    return int(getattr(args, "infer_imgsz", 640))


def run_table_attack_whitebox(
    model: torch.nn.Module,
    dataset: Any,
    args: Any,
    run_id: int,
    scene_perturber: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
) -> Dict[str, Any]:
    ensure_project_root_on_path()
    from evaluation_metrics import compute_perturbation_metrics

    n = min(int(args.num_eval), len(dataset))
    idxs = random.sample(range(len(dataset)), n)
    log_every = max(1, int(args.log_every))
    t0 = time.time()

    n_scenes = 0
    clean_correct_tables = 0
    restricted_den = 0
    restricted_succ = 0
    clean_imgs: List[torch.Tensor] = []
    adv_imgs: List[torch.Tensor] = []

    side = _infer_side(args)
    conf = float(args.conf)

    for step, idx in enumerate(idxs):
        img, target = dataset[idx]
        img = img.to(args.device)
        truth = truth_table_from_voc_target(target)
        if not truth:
            continue
        n_scenes += 1

        clean_hyp = predicted_table_from_scene(model, img, score_cutoff=conf, image_side=side)
        clean_ok = tables_match(truth, clean_hyp)
        if clean_ok:
            clean_correct_tables += 1

        adv = scene_perturber(model, img)
        adv_hyp = predicted_table_from_scene(model, adv, score_cutoff=conf, image_side=side)
        if clean_ok:
            restricted_den += 1
            if not tables_match(truth, adv_hyp):
                restricted_succ += 1

        if len(clean_imgs) < 200:
            clean_imgs.append(img.detach().cpu())
            adv_imgs.append(adv.detach().cpu())

        if (step + 1) % log_every == 0 or (step + 1) == n:
            rate = restricted_succ / max(1, restricted_den) * 100.0
            print(
                f"  [场景表-白盒 #{run_id}] {step+1}/{n} | 干净表正确 {clean_correct_tables}/{n_scenes} | "
                f"受限表破坏率={rate:.2f}% | {(time.time()-t0)/60:.1f}m",
                flush=True,
            )

    clean_table_accuracy = clean_correct_tables / max(1, n_scenes)
    restricted_table_break_rate = restricted_succ / max(1, restricted_den)
    pert = compute_perturbation_metrics(clean_imgs, adv_imgs, restricted_table_break_rate)
    return {
        "n_scenes": n_scenes,
        "clean_table_accuracy": clean_table_accuracy,
        "restricted_table_break_rate": restricted_table_break_rate,
        "restricted_den": restricted_den,
        "restricted_succ": restricted_succ,
        "pert": pert,
    }


def _bb_imports():
    ensure_project_root_on_path()
    from attack_utils import (
        VOC_CLASSES,
        build_blackbox_attack_refs,
        filter_pred,
        voc_target_to_boxes_and_labels,
    )
    from evaluation_metrics import compute_perturbation_metrics

    return VOC_CLASSES, build_blackbox_attack_refs, filter_pred, voc_target_to_boxes_and_labels, compute_perturbation_metrics


def run_table_attack_blackbox_ttba(model, dataset, args, run_id: int) -> Dict[str, Any]:
    mod = load_module_from_parent("blackbox_ttba.py", "_k_ttba")
    _, build_blackbox_attack_refs, filter_pred, voc_target_to_boxes_and_labels, compute_perturbation_metrics = (
        _bb_imports()
    )

    n = min(int(args.num_eval), len(dataset))
    idxs = random.sample(range(len(dataset)), n)
    log_every = max(1, int(args.log_every))
    t0 = time.time()

    n_scenes = 0
    clean_correct_tables = 0
    restricted_den = 0
    restricted_succ = 0
    clean_imgs: List[torch.Tensor] = []
    adv_imgs: List[torch.Tensor] = []

    imgsz = int(getattr(args, "infer_imgsz", 640))
    oc = float(args.objective_conf) if getattr(args, "objective_conf", None) is not None else float(args.conf)
    side = imgsz
    conf = float(args.conf)

    for step, idx in enumerate(idxs):
        img, target = dataset[idx]
        img = img.to(args.device)
        truth = truth_table_from_voc_target(target)
        if not truth:
            continue
        n_scenes += 1
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)

        clean_hyp = predicted_table_from_scene(model, img, score_cutoff=conf, image_side=side)
        clean_ok = tables_match(truth, clean_hyp)
        if clean_ok:
            clean_correct_tables += 1

        pred_clean = mod.infer(model, img, imgsz=imgsz)
        fp_ref = filter_pred(pred_clean, conf_thresh=float(args.ref_conf))
        ref_boxes, ref_labels, ref_scores = build_blackbox_attack_refs(
            fp_ref,
            gt_boxes,
            gt_labels,
            img.device,
            int(args.ref_topk),
            str(args.ref_mode),
            hybrid_suppress_iou=float(args.iou),
        )

        adv = img.detach().clone()
        best_f = float("inf")
        for _ in range(max(1, int(args.attack_restarts))):
            cand = mod.ttba_attack(
                model,
                img,
                float(args.eps),
                int(args.outer_steps),
                int(args.bridge_iters),
                ref_boxes,
                ref_labels,
                ref_scores,
                int(args.topk),
                float(args.iou_match),
                not bool(args.no_random_start),
                oc,
                imgsz=imgsz,
                bridge_shrink_tol=float(args.bridge_shrink_tol),
            )
            fv = mod._f(
                model,
                cand,
                ref_boxes,
                ref_labels,
                ref_scores,
                int(args.topk),
                float(args.iou_match),
                oc,
                imgsz=imgsz,
            )
            if fv < best_f:
                best_f = fv
                adv = cand

        adv_hyp = predicted_table_from_scene(model, adv, score_cutoff=conf, image_side=side)
        if clean_ok:
            restricted_den += 1
            if not tables_match(truth, adv_hyp):
                restricted_succ += 1

        if len(clean_imgs) < 200:
            clean_imgs.append(img.detach().cpu())
            adv_imgs.append(adv.detach().cpu())

        if (step + 1) % log_every == 0 or (step + 1) == n:
            rate = restricted_succ / max(1, restricted_den) * 100.0
            print(
                f"  [场景表-查询式A #{run_id}] {step+1}/{n} | 干净表正确 {clean_correct_tables}/{n_scenes} | "
                f"受限表破坏率={rate:.2f}% | {(time.time()-t0)/60:.1f}m",
                flush=True,
            )

    clean_table_accuracy = clean_correct_tables / max(1, n_scenes)
    restricted_table_break_rate = restricted_succ / max(1, restricted_den)
    pert = compute_perturbation_metrics(clean_imgs, adv_imgs, restricted_table_break_rate)
    return {
        "n_scenes": n_scenes,
        "clean_table_accuracy": clean_table_accuracy,
        "restricted_table_break_rate": restricted_table_break_rate,
        "restricted_den": restricted_den,
        "restricted_succ": restricted_succ,
        "pert": pert,
    }


def run_table_attack_blackbox_seri(model, dataset, args, run_id: int) -> Dict[str, Any]:
    mod = load_module_from_parent("blackbox_seri.py", "_k_seri")
    _, build_blackbox_attack_refs, filter_pred, voc_target_to_boxes_and_labels, compute_perturbation_metrics = _bb_imports()

    n = min(int(args.num_eval), len(dataset))
    idxs = random.sample(range(len(dataset)), n)
    log_every = max(1, int(args.log_every))
    t0 = time.time()

    n_scenes = 0
    clean_correct_tables = 0
    restricted_den = 0
    restricted_succ = 0
    clean_imgs: List[torch.Tensor] = []
    adv_imgs: List[torch.Tensor] = []

    oc = float(args.objective_conf) if getattr(args, "objective_conf", None) is not None else float(args.conf)
    bilateral = bool(int(getattr(args, "bilateral_sens", 1)))
    side = int(getattr(args, "infer_imgsz", 640))
    conf = float(args.conf)

    for step, idx in enumerate(idxs):
        img, target = dataset[idx]
        img = img.to(args.device)
        truth = truth_table_from_voc_target(target)
        if not truth:
            continue
        n_scenes += 1
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)

        clean_hyp = predicted_table_from_scene(model, img, score_cutoff=conf, image_side=side)
        clean_ok = tables_match(truth, clean_hyp)
        if clean_ok:
            clean_correct_tables += 1

        pred_clean = mod.infer(model, img)
        fp_ref = filter_pred(pred_clean, conf_thresh=float(args.ref_conf))
        ref_boxes, ref_labels, ref_scores = build_blackbox_attack_refs(
            fp_ref,
            gt_boxes,
            gt_labels,
            img.device,
            int(args.ref_topk),
            str(args.ref_mode),
            hybrid_suppress_iou=float(args.iou),
        )

        adv = img.detach().clone()
        best_f = float("inf")
        for _ in range(max(1, int(args.attack_restarts))):
            cand = mod.seri_attack(
                model,
                img,
                float(args.eps),
                int(args.block),
                int(args.top_blocks),
                float(args.sens_delta),
                int(args.refine_steps),
                float(args.refine_alpha),
                ref_boxes,
                ref_labels,
                ref_scores,
                int(args.topk),
                float(args.iou_match),
                oc,
                bilateral_sens=bilateral,
                ref_prior_weight=float(args.ref_prior_weight),
                ref_mask_expand=float(args.ref_mask_expand),
                refine_line_points=int(args.refine_line_points),
                random_init=bool(getattr(args, "attack_random_init", False)),
            )
            fv = mod._f(model, cand, ref_boxes, ref_labels, ref_scores, int(args.topk), float(args.iou_match), oc)
            if fv < best_f:
                best_f = fv
                adv = cand

        adv_hyp = predicted_table_from_scene(model, adv, score_cutoff=conf, image_side=side)
        if clean_ok:
            restricted_den += 1
            if not tables_match(truth, adv_hyp):
                restricted_succ += 1

        if len(clean_imgs) < 200:
            clean_imgs.append(img.detach().cpu())
            adv_imgs.append(adv.detach().cpu())

        if (step + 1) % log_every == 0 or (step + 1) == n:
            rate = restricted_succ / max(1, restricted_den) * 100.0
            print(
                f"  [场景表-查询式B #{run_id}] {step+1}/{n} | 干净表正确 {clean_correct_tables}/{n_scenes} | "
                f"受限表破坏率={rate:.2f}% | {(time.time()-t0)/60:.1f}m",
                flush=True,
            )

    clean_table_accuracy = clean_correct_tables / max(1, n_scenes)
    restricted_table_break_rate = restricted_succ / max(1, restricted_den)
    pert = compute_perturbation_metrics(clean_imgs, adv_imgs, restricted_table_break_rate)
    return {
        "n_scenes": n_scenes,
        "clean_table_accuracy": clean_table_accuracy,
        "restricted_table_break_rate": restricted_table_break_rate,
        "restricted_den": restricted_den,
        "restricted_succ": restricted_succ,
        "pert": pert,
    }


def run_table_attack_blackbox_adba(model, dataset, args, run_id: int) -> Dict[str, Any]:
    mod = load_module_from_parent("blackbox_adba.py", "_k_adba")
    (
        VOC_CLASSES,
        build_blackbox_attack_refs,
        filter_pred,
        voc_target_to_boxes_and_labels,
        compute_perturbation_metrics,
    ) = _bb_imports()
    ensure_project_root_on_path()
    from attack_utils import project_linf_01, spatial_mask_from_boxes_xyxy

    voc_map = {name: idx for idx, name in enumerate(VOC_CLASSES)}

    n = min(int(args.num_eval), len(dataset))
    idxs = random.sample(range(len(dataset)), n)
    log_every = max(1, int(args.log_every))
    t0 = time.time()

    n_scenes = 0
    clean_correct_tables = 0
    restricted_den = 0
    restricted_succ = 0
    clean_imgs: List[torch.Tensor] = []
    adv_imgs: List[torch.Tensor] = []

    oc = float(args.objective_conf) if getattr(args, "objective_conf", None) is not None else float(args.conf)
    side = int(getattr(args, "infer_imgsz", 640))
    conf = float(args.conf)

    for step, idx in enumerate(idxs):
        img, target = dataset[idx]
        img = img.to(args.device)
        truth = truth_table_from_voc_target(target)
        if not truth:
            continue
        n_scenes += 1
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)

        clean_hyp = predicted_table_from_scene(model, img, score_cutoff=conf, image_side=side)
        clean_ok = tables_match(truth, clean_hyp)
        if clean_ok:
            clean_correct_tables += 1

        pred_clean = mod.infer(model, img)
        fp_clean = filter_pred(pred_clean, conf_thresh=float(args.conf))
        fp_ref = filter_pred(pred_clean, conf_thresh=float(args.ref_conf))
        ref_boxes, ref_labels, ref_scores = build_blackbox_attack_refs(
            fp_ref,
            gt_boxes,
            gt_labels,
            img.device,
            int(args.ref_topk),
            str(args.ref_mode),
            hybrid_suppress_iou=float(args.iou),
        )

        _, H, W = img.shape
        aux_b, aux_lid = mod._gt_aux_label_tensors(gt_boxes, gt_labels, img.device)
        wfp = float(getattr(args, "adba_aux_fp", 0.0))
        wmc = float(getattr(args, "adba_aux_misclass", 0.0))
        tk_aux = max(1, int(getattr(args, "adba_aux_topk", 64)))

        qu: Optional[List[int]] = [0] if int(getattr(args, "max_queries_per_image", 0)) > 0 else None
        qmax = int(getattr(args, "max_queries_per_image", 0))
        adb_ref = max(0, int(getattr(args, "adb_ray_refine_iters", 0)))
        adb_imp = not bool(getattr(args, "adb_allow_uphill", False))

        adv = img.detach().clone()
        best_f = float("inf")

        vix = mod._valid_gt_indices(gt_labels)
        use_sequential_chain = bool(getattr(args, "adba_sequential_gt", False)) or (
            bool(getattr(args, "adba_per_gt_local", False)) and len(vix) >= 2
        )

        if use_sequential_chain:
            if len(vix) == 0 or gt_boxes.numel() == 0:
                spatial_mask_sg: Optional[torch.Tensor] = None
                if float(getattr(args, "gt_focus_expand", 0.0)) > 0.0 and gt_boxes.numel() > 0:
                    spatial_mask_sg = spatial_mask_from_boxes_xyxy(
                        gt_boxes.to(img.device),
                        H,
                        W,
                        img.device,
                        expand_frac=float(args.gt_focus_expand),
                    )
                for _ in range(max(1, int(args.attack_restarts))):
                    cand = mod.adba_attack(
                        model,
                        img,
                        float(args.eps),
                        int(args.steps),
                        float(args.probe_scale),
                        float(args.alpha),
                        ref_boxes,
                        ref_labels,
                        ref_scores,
                        int(args.topk),
                        float(args.iou_match),
                        not bool(args.no_random_start),
                        float(args.stop_ratio),
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
                    fv = mod._f(
                        model,
                        cand,
                        ref_boxes,
                        ref_labels,
                        ref_scores,
                        int(args.topk),
                        float(args.iou_match),
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
                expand_m = float(args.gt_focus_expand) if float(getattr(args, "gt_focus_expand", 0.0)) > 0.0 else 0.28
                topk_sg = max(1, min(int(args.topk), 32))
                if bool(getattr(args, "adba_per_gt_local", False)):
                    per_n = max(4, int(args.per_gt_steps), int(args.steps))
                else:
                    per_n = max(4, int(args.per_gt_steps))
                x_chain = img.detach().clone()
                rs_one = torch.ones((1,), device=img.device, dtype=torch.float32)
                for j, gi in enumerate(vix):
                    rb = gt_boxes[gi : gi + 1].to(device=img.device, dtype=torch.float32)
                    lid = int(voc_map[gt_labels[gi]])
                    rl = torch.tensor([lid], device=img.device, dtype=torch.long)
                    mask_g = spatial_mask_from_boxes_xyxy(rb, H, W, img.device, expand_frac=expand_m)
                    stage_best = x_chain
                    stage_best_f = float("inf")
                    rs_here = not bool(args.no_random_start) and (j == 0)
                    for _ in range(max(1, int(args.attack_restarts))):
                        cand = mod.adba_attack(
                            model,
                            img,
                            float(args.eps),
                            per_n,
                            float(args.probe_scale),
                            float(args.alpha),
                            rb,
                            rl,
                            rs_one,
                            topk_sg,
                            float(args.iou_match),
                            rs_here,
                            float(args.stop_ratio),
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
                        fv = mod._f(
                            model,
                            cand,
                            rb,
                            rl,
                            rs_one,
                            topk_sg,
                            float(args.iou_match),
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
                    x_chain = project_linf_01(stage_best, img, float(args.eps))
                adv = x_chain
        elif bool(getattr(args, "adba_per_gt_local", False)) and len(vix) == 1:
            expand_m = float(args.gt_focus_expand) if float(getattr(args, "gt_focus_expand", 0.0)) > 0.0 else 0.28
            gi = vix[0]
            rb = gt_boxes[gi : gi + 1].to(device=img.device, dtype=torch.float32)
            mask_g = spatial_mask_from_boxes_xyxy(rb, H, W, img.device, expand_frac=expand_m)
            for _ in range(max(1, int(args.attack_restarts))):
                cand = mod.adba_attack(
                    model,
                    img,
                    float(args.eps),
                    int(args.steps),
                    float(args.probe_scale),
                    float(args.alpha),
                    ref_boxes,
                    ref_labels,
                    ref_scores,
                    int(args.topk),
                    float(args.iou_match),
                    not bool(args.no_random_start),
                    float(args.stop_ratio),
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
                fv = mod._f(
                    model,
                    cand,
                    ref_boxes,
                    ref_labels,
                    ref_scores,
                    int(args.topk),
                    float(args.iou_match),
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
            if float(getattr(args, "gt_focus_expand", 0.0)) > 0.0 and gt_boxes.numel() > 0:
                spatial_mask = spatial_mask_from_boxes_xyxy(
                    gt_boxes.to(img.device),
                    H,
                    W,
                    img.device,
                    expand_frac=float(args.gt_focus_expand),
                )
            for _ in range(max(1, int(args.attack_restarts))):
                cand = mod.adba_attack(
                    model,
                    img,
                    float(args.eps),
                    int(args.steps),
                    float(args.probe_scale),
                    float(args.alpha),
                    ref_boxes,
                    ref_labels,
                    ref_scores,
                    int(args.topk),
                    float(args.iou_match),
                    not bool(args.no_random_start),
                    float(args.stop_ratio),
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
                fv = mod._f(
                    model,
                    cand,
                    ref_boxes,
                    ref_labels,
                    ref_scores,
                    int(args.topk),
                    float(args.iou_match),
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

        adv_hyp = predicted_table_from_scene(model, adv, score_cutoff=conf, image_side=side)
        if clean_ok:
            restricted_den += 1
            if not tables_match(truth, adv_hyp):
                restricted_succ += 1

        if len(clean_imgs) < 200:
            clean_imgs.append(img.detach().cpu())
            adv_imgs.append(adv.detach().cpu())

        if (step + 1) % log_every == 0 or (step + 1) == n:
            rate = restricted_succ / max(1, restricted_den) * 100.0
            print(
                f"  [场景表-查询式C #{run_id}] {step+1}/{n} | 干净表正确 {clean_correct_tables}/{n_scenes} | "
                f"受限表破坏率={rate:.2f}% | {(time.time()-t0)/60:.1f}m",
                flush=True,
            )

    clean_table_accuracy = clean_correct_tables / max(1, n_scenes)
    restricted_table_break_rate = restricted_succ / max(1, restricted_den)
    pert = compute_perturbation_metrics(clean_imgs, adv_imgs, restricted_table_break_rate)
    return {
        "n_scenes": n_scenes,
        "clean_table_accuracy": clean_table_accuracy,
        "restricted_table_break_rate": restricted_table_break_rate,
        "restricted_den": restricted_den,
        "restricted_succ": restricted_succ,
        "pert": pert,
    }
