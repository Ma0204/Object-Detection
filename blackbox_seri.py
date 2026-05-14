# -*- coding: utf-8 -*-
"""
黑盒攻击：SeRI（Sensitive Region Identification）— YOLO + VOC 检测适配。

论文：Feiyang Wang, Xingquan Zuo, Hai Huang, Gang Chen,
「SeRI: Efficient gradient-free sensitive region identification via boundary-guided search
in decision-based black-box attacks」, ICLR 2026.

与原文对照（写对的部分 / 差异）：
- 一致思想：**先识别敏感区域**（用少量查询比较「局部扰动前后」目标变化），再在敏感区域上
  **集中搜索**扰动，减少无效维度上的查询。
- 原文：决策黑盒 + 分类；敏感区识别后往往结合边界引导的细化（boundary-guided）。
- 论文 / 官方代码（BUPTAIOC/SeRI）：在**基攻击给出的扰动方向**上按子块 L2 选「重要块」，
  再对选中块做 **k1/k2 两尺度**扰动并用 **ADB 二分**贴决策边界（硬标签 oracle）。
- **本检测版**仍用连续标量 `_f`（与 GT-ASR 对齐的压制目标），但借鉴官方思路做了加强：
  **双侧块探测**（± 掩膜扰动）、**参考框空间先验**（与 `spatial_mask_from_boxes_xyxy` 融合掩膜）、
  **沿动量方向的多尺度线搜索**（单步多查询，类比边界搜索的离散化），以缓解「分类 1 维翻转 vs
  检测多框置信度」带来的查询效率差。
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List, Tuple

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
) -> float:
    pred = infer(model, x)
    return float(
        yolo_matched_suppression_objective(
            pred, ref_boxes, ref_labels, ref_scores, topk, iou_match, eval_conf=float(objective_conf)
        ).item()
    )


def _build_block_mask(
    shape: Tuple[int, int, int],
    by: int,
    bx: int,
    block: int,
    device: torch.device,
) -> torch.Tensor:
    _, h, w = shape
    m = torch.zeros((1, h, w), device=device, dtype=torch.float32)
    m[:, by : min(h, by + block), bx : min(w, bx + block)] = 1.0
    return m


def _block_ref_overlap(
    ref_spatial: torch.Tensor, by: int, bx: int, block: int, h: int, w: int
) -> float:
    patch = ref_spatial[:, by : min(h, by + block), bx : min(w, bx + block)]
    if patch.numel() == 0:
        return 0.0
    return float(patch.mean().item())


def seri_attack(
    model,
    x0: torch.Tensor,
    eps: float,
    block: int,
    top_blocks: int,
    sens_delta: float,
    refine_steps: int,
    refine_alpha: float,
    ref_boxes: torch.Tensor,
    ref_labels: torch.Tensor,
    ref_scores: torch.Tensor,
    topk: int,
    iou_match: float,
    objective_conf: float,
    *,
    bilateral_sens: bool = True,
    ref_prior_weight: float = 0.85,
    ref_mask_expand: float = 0.22,
    refine_line_points: int = 5,
    random_init: bool = False,
) -> torch.Tensor:
    """SeRI 检测版：双侧敏感块 + 参考框掩膜融合 + 多尺度线搜索精炼。"""
    device = x0.device
    x = x0.detach().clone()
    if random_init:
        x = project_linf_01(
            x0 + torch.empty_like(x0).uniform_(-float(eps), float(eps)), x0, float(eps)
        )
    _, h, w = x0.shape
    # 敏感区打分始终相对「干净图」f(x0)，与论文/官方从原图探边界的设定一致
    f0 = _f(model, x0, ref_boxes, ref_labels, ref_scores, topk, iou_match, objective_conf)

    ref_sp = spatial_mask_from_boxes_xyxy(
        ref_boxes, h, w, device, expand_frac=float(ref_mask_expand)
    )
    w_prior = max(0.0, float(ref_prior_weight))

    scores: List[Tuple[float, int, int]] = []
    d = float(sens_delta) * float(eps)
    for by in range(0, h, block):
        for bx in range(0, w, block):
            mask = _build_block_mask(x0.shape, by, bx, block, device).expand_as(x0)
            olap = _block_ref_overlap(ref_sp, by, bx, block, h, w)
            if bilateral_sens:
                x_p = project_linf_01(x0 + d * mask, x0, eps)
                x_m = project_linf_01(x0 - d * mask, x0, eps)
                fp = _f(model, x_p, ref_boxes, ref_labels, ref_scores, topk, iou_match, objective_conf)
                fm = _f(model, x_m, ref_boxes, ref_labels, ref_scores, topk, iou_match, objective_conf)
                delta = max(f0 - fp, f0 - fm)
            else:
                x_try = project_linf_01(x0 + d * mask, x0, eps)
                f1 = _f(model, x_try, ref_boxes, ref_labels, ref_scores, topk, iou_match, objective_conf)
                delta = f0 - f1
            bonus = 1.0 + w_prior * olap
            scores.append((delta * bonus, by, bx))
    scores.sort(key=lambda t: t[0], reverse=True)
    keep = scores[: max(1, int(top_blocks))]
    sens_mask = torch.zeros_like(x0)
    for _, by, bx in keep:
        sens_mask += _build_block_mask(x0.shape, by, bx, block, device).expand_as(x0)
    sens_mask = (sens_mask > 0).float()
    perturb_mask = torch.maximum(sens_mask, ref_sp.expand_as(sens_mask))

    vel = torch.zeros_like(x0)
    beta = 0.42
    n_ref = max(1, int(refine_steps))
    n_lp = max(3, min(9, int(refine_line_points)))
    if n_lp % 2 == 0:
        n_lp += 1
    scales = torch.linspace(-1.0, 1.0, steps=n_lp, device=device, dtype=torch.float32)
    for si in range(n_ref):
        prog = float(si) / float(max(n_ref - 1, 1))
        ra = float(refine_alpha) * (1.0 - 0.32 * prog)
        g = torch.sign(torch.randn_like(x0)) * perturb_mask
        if float(g.abs().sum()) < 1e-6:
            g = torch.sign(torch.randn_like(x0)) * perturb_mask
        vel = beta * vel + (1.0 - beta) * g
        u = torch.sign(vel) * perturb_mask
        if float(u.abs().sum()) < 1e-6:
            u = g
        best_x = x
        best_fv = float("inf")
        for s in scales.tolist():
            if abs(float(s)) < 1e-9:
                x_c = x
            else:
                x_c = project_linf_01(x + float(s) * ra * u, x0, eps)
            fv = _f(model, x_c, ref_boxes, ref_labels, ref_scores, topk, iou_match, objective_conf)
            if fv < best_fv:
                best_fv = fv
                best_x = x_c
        x = project_linf_01(best_x, x0, eps)
    return x.detach()


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
    print(f"  [SeRI 第{run_id}次] 评估 {n} 张；每 {log_every} 张打印进度。", flush=True)

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
        adv = img.detach().clone()
        best_f = float("inf")
        for _ in range(max(1, int(args.attack_restarts))):
            cand = seri_attack(
                model,
                img,
                args.eps,
                args.block,
                args.top_blocks,
                args.sens_delta,
                args.refine_steps,
                args.refine_alpha,
                ref_boxes,
                ref_labels,
                ref_scores,
                args.topk,
                args.iou_match,
                oc,
                bilateral_sens=bool(args.bilateral_sens),
                ref_prior_weight=float(args.ref_prior_weight),
                ref_mask_expand=float(args.ref_mask_expand),
                refine_line_points=int(args.refine_line_points),
                random_init=bool(args.attack_random_init),
            )
            fv = _f(model, cand, ref_boxes, ref_labels, ref_scores, args.topk, args.iou_match, oc)
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
            cur_asr = (gt_success_count / max(1, gt_detected_clean)) * 100.0
            print(f"  [SeRI] {step+1}/{n} | GT-ASR={cur_asr:.2f}% | {(time.time()-t0)/60:.1f}m", flush=True)

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
    path = os.path.join(args.outdir, "seri_results.txt")
    os.makedirs(args.outdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("黑盒攻击评估 - SeRI (YOLO 检测适配)\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"eps={args.eps} block={args.block} top_blocks={args.top_blocks} "
            f"refine_steps={args.refine_steps} restarts={args.attack_restarts} "
            f"ref_mode={args.ref_mode} ref_conf={args.ref_conf} "
            f"bilateral_sens={args.bilateral_sens} refine_line_points={args.refine_line_points}\n"
        )
        f.write(f"均值 GT-ASR: {asr.mean()*100:.2f}%\n")
    print(f"结果已保存: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="SeRI 黑盒攻击（YOLO VOC）")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--block", type=int, default=48)
    p.add_argument("--top_blocks", type=int, default=14)
    p.add_argument("--sens_delta", type=float, default=0.42)
    p.add_argument(
        "--bilateral_sens",
        type=int,
        default=1,
        choices=[0, 1],
        help="1：敏感区双侧 ± 探测（推荐，查询×2）；0：仅正向单侧（省查询）",
    )
    p.add_argument(
        "--ref_prior_weight",
        type=float,
        default=0.85,
        help="参考框与块重叠时对敏感度分数的加成权重（0 关闭）",
    )
    p.add_argument(
        "--ref_mask_expand",
        type=float,
        default=0.22,
        help="参考框空间掩膜外扩比例，与精炼扰动掩膜融合",
    )
    p.add_argument(
        "--refine_line_points",
        type=int,
        default=5,
        help="每步沿动量方向在 [-1,1] 上均匀采样的点数（奇数，含 0；越大查询越多）",
    )
    p.add_argument(
        "--attack_random_init",
        action="store_true",
        help="每次 SeRI 从 L∞ 球内随机起点开始精炼（敏感区仍相对干净图打分）",
    )
    p.add_argument("--refine_steps", type=int, default=120)
    p.add_argument("--refine_alpha", type=float, default=0.018)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--ref_topk", type=int, default=120)
    p.add_argument(
        "--ref_mode",
        type=str,
        default="hybrid",
        choices=["pred", "gt", "hybrid"],
        help="攻击参考框来源；默认 hybrid 贴近 GT-ASR",
    )
    p.add_argument("--ref_conf", type=float, default=0.22)
    p.add_argument("--iou_match", type=float, default=0.5)
    p.add_argument(
        "--attack_restarts",
        type=int,
        default=3,
        help="每张图重复 SeRI 全流程次数，取 f 最小者（查询量约×该倍数）",
    )
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument(
        "--objective_conf",
        type=float,
        default=None,
        help="黑盒目标 margin 阈值；默认与 --conf 相同",
    )
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--num_eval", type=int, default=200)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--load_model", type=str, required=True)
    p.add_argument("--eval_set", type=str, default="test", choices=["train", "val", "trainval", "test"])
    p.add_argument("--outdir", type=str, default="./adv_outputs/seri")
    p.add_argument("--log_every", type=int, default=5)
    args = p.parse_args()
    if args.eps > 0.05:
        args.eps = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("Device:", device)
    print(
        f"SeRI: ref_mode={args.ref_mode} ref_conf={args.ref_conf} refine_steps={args.refine_steps} "
        f"attack_restarts={args.attack_restarts} top_blocks={args.top_blocks} iou_match={args.iou_match} "
        f"objective_conf={(args.objective_conf if args.objective_conf is not None else args.conf)} "
        f"bilateral_sens={args.bilateral_sens} refine_line_points={args.refine_line_points} "
        f"ref_prior_weight={args.ref_prior_weight}",
        flush=True,
    )
    os.makedirs(args.outdir, exist_ok=True)
    ds = load_voc2007_dataset(args.eval_set)
    model = build_yolo_voc_model(device, weights=args.load_model)

    all_r = []
    for i in range(1, args.runs + 1):
        r = evaluate_once(model, ds, args, i)
        all_r.append(r)
        print(f"\n[SeRI 第{i}次]")
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
    print("SeRI 汇总（YOLO 检测）")
    print("=" * 60)
    print(f"攻击成功率(GT级)均值: {asr.mean()*100:.2f}%")
    print(f"是否满足 >85%: {asr.mean() > 0.85}")
    save_results(args, all_r)


if __name__ == "__main__":
    main()
