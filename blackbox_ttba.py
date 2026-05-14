# -*- coding: utf-8 -*-
"""
黑盒检测适配脚本（文件名沿用 TtBA 便于与 ADBA/SeRI 同系列对照）。

论文：Feiyang Wang 等, 「TtBA: Two-third bridge approach for decision-based adversarial attack」,
ICML 2025. 官方实现：https://github.com/BUPTAIOC/TtBA

**与 ICML 2025 TtBA 论文是否一致：否。** 论文中的「two-third bridge」指：在决策边界法向量
与当前扰动方向张成的平面内，用权重 k 构造桥方向 d_bridge = k·N̂ + (1−k)·d̂，对 k_bridge
做二分，再用 k = (2/3)·k_bridge 更新搜索方向（见论文图 1 与主算法）。**本文件未实现**
法向量估计、k_bridge 二分或上述方向混合；`ttba_attack` 实现的是：沿随机符号方向 u，
在当前对抗样本 x 附近的射线 `Proj_L∞(x + t·eps·u)` 上做**标量区间三分搜索**以减小连续目标 f
（相对旧版从 x0 出发的射线，更利于在 L∞ 球内累积改进）。要对齐原文请 fork [BUPTAIOC/TtBA](https://github.com/BUPTAIOC/TtBA) 并将决策 oracle 换为本文 `_f()`。

**速度说明：** 每次 `_f` 都会跑一次完整 YOLO `predict`。`outer_steps×bridge_iters` 过大时查询量爆炸。
已做：缓存 f(x)、内层区间足够小时提前结束、`--infer_imgsz` 降分辨率加速；仍建议 `outer_steps` 与
`bridge_iters` 控制在「脚本启动时打印的估计 infer 次数」可接受范围内。
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
def infer(model, img: torch.Tensor, *, imgsz: int) -> Dict[str, torch.Tensor]:
    return infer_yolo(model, img, imgsz=int(imgsz))


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
    imgsz: int,
) -> float:
    pred = infer(model, x, imgsz=int(imgsz))
    return float(
        yolo_matched_suppression_objective(
            pred, ref_boxes, ref_labels, ref_scores, topk, iou_match, eval_conf=float(objective_conf)
        ).item()
    )


def ttba_attack(
    model,
    x0: torch.Tensor,
    eps: float,
    outer_steps: int,
    bridge_iters: int,
    ref_boxes: torch.Tensor,
    ref_labels: torch.Tensor,
    ref_scores: torch.Tensor,
    topk: int,
    iou_match: float,
    random_start: bool,
    objective_conf: float,
    *,
    imgsz: int,
    bridge_shrink_tol: float = 0.04,
) -> torch.Tensor:
    """
    标量三分搜索：每轮随机 u，在当前 iterate x 附近的射线 x+t·eps·u 上缩小含优区间，
    用中点候选更新当前最优 x（仍约束在相对 x0 的 L∞ 球 + [0,1]）。

    性能：相对旧版在每轮外循环去掉重复的 f(x)；区间足够小时提前结束内层三分迭代；
    推理分辨率由 imgsz 控制（略降可显著加速黑盒查询）。
    """
    x = x0.detach().clone()
    if random_start:
        x = x + (torch.rand_like(x) * 2.0 - 1.0) * float(eps)
        x = project_linf_01(x, x0, eps)

    fx = _f(
        model,
        x,
        ref_boxes,
        ref_labels,
        ref_scores,
        topk,
        iou_match,
        objective_conf,
        imgsz=int(imgsz),
    )
    btol = max(1e-6, float(bridge_shrink_tol))

    for _ in range(int(outer_steps)):
        u = torch.sign(torch.randn_like(x0))
        lo, hi = 0.0, 1.0
        for __ in range(int(bridge_iters)):
            if hi - lo <= btol:
                break
            t1 = (2.0 * lo + hi) / 3.0
            t2 = (lo + 2.0 * hi) / 3.0
            x1 = project_linf_01(x + t1 * float(eps) * u, x0, eps)
            x2 = project_linf_01(x + t2 * float(eps) * u, x0, eps)
            f1 = _f(
                model,
                x1,
                ref_boxes,
                ref_labels,
                ref_scores,
                topk,
                iou_match,
                objective_conf,
                imgsz=int(imgsz),
            )
            f2 = _f(
                model,
                x2,
                ref_boxes,
                ref_labels,
                ref_scores,
                topk,
                iou_match,
                objective_conf,
                imgsz=int(imgsz),
            )
            if f1 <= f2:
                hi = t2
            else:
                lo = t1
        t_mid = (lo + hi) / 2.0
        x_cand = project_linf_01(x + t_mid * float(eps) * u, x0, eps)
        fc = _f(
            model,
            x_cand,
            ref_boxes,
            ref_labels,
            ref_scores,
            topk,
            iou_match,
            objective_conf,
            imgsz=int(imgsz),
        )
        if fc < fx:
            x = x_cand
            fx = float(fc)
        x = project_linf_01(x, x0, eps)
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
    print(f"  [TtBA 第{run_id}次] 评估 {n} 张；每 {log_every} 张打印 GT-ASR 汇总。", flush=True)
    print(
        "  说明：每张图会先干净推理 1 次，再跑数千次 YOLO 黑盒查询；"
        "在凑满 log_every 张之前汇总行不会刷新，看起来像「卡住」实为在算第 1 张。",
        flush=True,
    )

    c_tp = c_fp = c_fn = 0
    a_tp = a_fp = a_fn = 0
    gt_detected_clean = gt_success_count = 0
    img_detected_clean = img_success_count = 0
    clean_imgs, adv_imgs = [], []

    for step, idx in enumerate(idxs):
        t_img = time.time()
        print(
            f"  [TtBA] 图 {step + 1}/{n} 开始 (dataset_idx={idx}) …",
            flush=True,
        )
        img_tensor, target = dataset[idx]
        img = img_tensor.to(args.device)
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)

        pred_clean = infer(model, img, imgsz=int(args.infer_imgsz))
        print(f"  [TtBA] 图 {step + 1}/{n} 干净推理完成，黑盒优化中（本张可能需数分钟）…", flush=True)
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
            cand = ttba_attack(
                model,
                img,
                args.eps,
                args.outer_steps,
                args.bridge_iters,
                ref_boxes,
                ref_labels,
                ref_scores,
                args.topk,
                args.iou_match,
                not bool(args.no_random_start),
                oc,
                imgsz=int(args.infer_imgsz),
                bridge_shrink_tol=float(args.bridge_shrink_tol),
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
                imgsz=int(args.infer_imgsz),
            )
            if fv < best_f:
                best_f = fv
                adv = cand
        pred_adv = infer(model, adv, imgsz=int(args.infer_imgsz))
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

        img_sec = time.time() - t_img
        cur_asr_one = (gt_success_count / max(1, gt_detected_clean)) * 100.0
        print(
            f"  [TtBA] 图 {step + 1}/{n} 完成 本张用时 {img_sec/60:.1f}m | 累计 GT-ASR={cur_asr_one:.2f}%",
            flush=True,
        )

        if (step + 1) % log_every == 0 or (step + 1) == n:
            cur_asr = (gt_success_count / max(1, gt_detected_clean)) * 100.0
            print(f"  [TtBA] 小结 {step+1}/{n} | GT-ASR={cur_asr:.2f}% | 总用时 {(time.time()-t0)/60:.1f}m", flush=True)

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
    path = os.path.join(args.outdir, "ttba_results.txt")
    os.makedirs(args.outdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("黑盒攻击评估 - TtBA (YOLO 检测适配)\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"eps={args.eps} outer_steps={args.outer_steps} bridge_iters={args.bridge_iters} "
            f"infer_imgsz={args.infer_imgsz} restarts={args.attack_restarts} ref_mode={args.ref_mode} ref_conf={args.ref_conf}\n"
        )
        f.write(f"均值 GT-ASR: {asr.mean()*100:.2f}%\n")
    print(f"结果已保存: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="TtBA 黑盒攻击（YOLO VOC）")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--outer_steps", type=int, default=48)
    p.add_argument(
        "--bridge_iters",
        type=int,
        default=6,
        help="内层三分迭代上限；区间小于 bridge_shrink_tol 时会提前结束以省查询",
    )
    p.add_argument(
        "--bridge_shrink_tol",
        type=float,
        default=0.04,
        help="[lo,hi] 宽度小于该值时提前结束内层三分搜索（略省查询，略降精细度）",
    )
    p.add_argument(
        "--infer_imgsz",
        type=int,
        default=512,
        help="黑盒每次 infer_yolo 的 imgsz；640 更准但更慢，416/512 常显著加速",
    )
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--ref_topk", type=int, default=120)
    p.add_argument(
        "--ref_mode",
        type=str,
        default="hybrid",
        choices=["pred", "gt", "hybrid"],
        help="攻击参考框来源，与 ADBA/SeRI 一致；默认 hybrid 贴近 GT-ASR",
    )
    p.add_argument("--ref_conf", type=float, default=0.22)
    p.add_argument("--iou_match", type=float, default=0.5)
    p.add_argument(
        "--attack_restarts",
        type=int,
        default=5,
        help="每张图重复标量三分搜索次数，取 f 最小者",
    )
    p.add_argument("--no_random_start", action="store_true")
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
    p.add_argument("--outdir", type=str, default="./adv_outputs/ttba")
    p.add_argument("--log_every", type=int, default=5)
    args = p.parse_args()
    if args.eps > 0.05:
        args.eps = 0.05

    r = max(1, int(args.attack_restarts))
    os_ = max(1, int(args.outer_steps))
    bi = max(1, int(args.bridge_iters))
    worst_q = r * (1 + os_ * (1 + 2 * bi) + 1)
    print(
        f"估计每张最坏 infer 次数 ≈ attack_restarts×(1 + outer_steps×(1+2×bridge_iters)+1) "
        f"= {r}×(1+{os_}×{1+2*bi}+1) = {worst_q}（实际因 bridge_shrink_tol 常更少）",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("Device:", device)
    print(
        f"TtBA: ref_mode={args.ref_mode} ref_conf={args.ref_conf} outer_steps={args.outer_steps} "
        f"bridge_iters={args.bridge_iters} bridge_shrink_tol={args.bridge_shrink_tol} "
        f"attack_restarts={args.attack_restarts} infer_imgsz={args.infer_imgsz} iou_match={args.iou_match} "
        f"objective_conf={(args.objective_conf if args.objective_conf is not None else args.conf)}",
        flush=True,
    )
    os.makedirs(args.outdir, exist_ok=True)
    ds = load_voc2007_dataset(args.eval_set)
    model = build_yolo_voc_model(device, weights=args.load_model)

    all_r = []
    for i in range(1, args.runs + 1):
        r = evaluate_once(model, ds, args, i)
        all_r.append(r)
        print(f"\n[TtBA 第{i}次]")
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
    print("TtBA 汇总（YOLO 检测）")
    print("=" * 60)
    print(f"攻击成功率(GT级)均值: {asr.mean()*100:.2f}%")
    print(f"是否满足 >85%: {asr.mean() > 0.85}")
    save_results(args, all_r)


if __name__ == "__main__":
    main()
