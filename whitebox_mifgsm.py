# -*- coding: utf-8 -*-
"""
白盒攻击（VOC + YOLO）：脚本名仍为 whitebox_mifgsm.py，内核为 **L∞ Adam-PGD**，
针对检测比经典 MI-FGSM（梯度逐像素归一化 + 简单动量）更稳、通常更易拉高 GT-ASR。

- 目标：最小化 `yolo_whitebox_objective`（与 FGSM/PGD 一致的可微 surrogate）。
- 更新：Adam 一阶矩 m 对梯度 EMA，bias-correct 后取 sign，每步 L∞ 投影 + [0,1] 裁剪。
- 约束：eps 默认 0.05，与项目 L∞ 要求一致；评估仍为 infer_yolo + conf + IoU 的 GT-ASR。
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict

import numpy as np
import torch

from attack_utils import (
    build_yolo_voc_model,
    ensure_tensor_01,
    infer_yolo,
    load_voc2007_dataset,
    voc_target_to_boxes_and_labels,
    compute_iou,
    filter_pred,
    yolo_whitebox_objective,
    input_diversity,
    ti_smooth_grad,
)
from evaluation_metrics import compute_model_metrics, compute_perturbation_metrics

VOC_CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}


def build_voc_model(device: torch.device, weights: str) -> torch.nn.Module:
    return build_yolo_voc_model(device, weights=weights)


@torch.no_grad()
def infer(model: torch.nn.Module, img: torch.Tensor) -> Dict[str, torch.Tensor]:
    return infer_yolo(model, img)


def det_adam_pgd_attack(
    model: torch.nn.Module,
    img_01: torch.Tensor,
    eps: float,
    steps: int,
    alpha: float,
    beta1: float,
    random_start: bool,
    target_conf: float,
    min_steps: int,
    stop_loss: float,
    di_prob: float,
    di_scale_min: float,
    ti_kernel: int,
    ti_sigma: float,
    restarts: int,
    topk: int = 400,
) -> torch.Tensor:
    """L∞ Adam-PGD：最小化 yolo_whitebox_objective，多 restart 取 surrogate 最优。"""
    x0 = img_01.detach()
    best_x = x0.clone()
    best_score = float("inf")
    b1 = float(beta1)
    b1 = min(0.999, max(0.5, b1))

    n_restarts = max(1, int(restarts))
    for _ in range(n_restarts):
        x = (
            ensure_tensor_01(x0 + torch.empty_like(x0).uniform_(-eps, eps))
            if random_start
            else x0.clone()
        )
        m = torch.zeros_like(x0)
        last_loss: float | None = None

        for si in range(int(steps)):
            x = x.detach().clone().requires_grad_(True)
            x_in = input_diversity(x, prob=float(di_prob), scale_min=float(di_scale_min))
            loss = yolo_whitebox_objective(
                model, x_in, topk=int(topk), target_conf=float(target_conf)
            )
            loss.backward()
            grad = ti_smooth_grad(
                x.grad.detach(), kernel_size=int(ti_kernel), sigma=float(ti_sigma)
            )
            m = b1 * m + (1.0 - b1) * grad
            t = float(si + 1)
            m_hat = m / (1.0 - b1**t)
            with torch.no_grad():
                x = x - float(alpha) * m_hat.sign()
                x = torch.max(torch.min(x, x0 + float(eps)), x0 - float(eps))
                x = ensure_tensor_01(x)
            last_loss = float(loss.detach().item())
            if (
                float(stop_loss) >= 0.0
                and (si + 1) >= int(min_steps)
                and last_loss <= float(stop_loss)
            ):
                break

        with torch.no_grad():
            score = float(
                yolo_whitebox_objective(
                    model, x.detach(), topk=int(topk), target_conf=float(target_conf)
                ).item()
            )
        if score < best_score:
            best_score = score
            best_x = x.detach().clone()

    return best_x.detach()


def evaluate_once(model, dataset, args, run_id: int):
    n = min(int(args.num_eval), len(dataset))
    idxs = random.sample(range(len(dataset)), n)
    t0 = time.time()

    c_tp = c_fp = c_fn = 0
    a_tp = a_fp = a_fn = 0
    gt_detected_clean = gt_success_count = 0
    img_detected_clean = img_success_count = 0
    clean_imgs, adv_imgs = [], []

    def count_metrics(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh):
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

    for step, idx in enumerate(idxs):
        img_tensor, target = dataset[idx]
        img = img_tensor.to(args.device)
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)

        pred_clean = infer(model, img)
        fp_clean = filter_pred(pred_clean, conf_thresh=args.conf)
        boxes_c = fp_clean["boxes"].cpu()
        labels_c = fp_clean["labels"].cpu().tolist()

        adv = det_adam_pgd_attack(
            model,
            img,
            eps=args.eps,
            steps=args.steps,
            alpha=args.alpha,
            beta1=args.beta1,
            random_start=args.random_start,
            target_conf=args.target_conf,
            min_steps=args.min_steps,
            stop_loss=args.stop_loss,
            di_prob=args.di_prob,
            di_scale_min=args.di_scale_min,
            ti_kernel=args.ti_kernel,
            ti_sigma=args.ti_sigma,
            restarts=args.restarts,
            topk=int(args.topk),
        )

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

        if (step + 1) % max(1, int(args.log_every)) == 0 or (step + 1) == n:
            done = step + 1
            elapsed = time.time() - t0
            speed = done / max(elapsed, 1e-6)
            eta = (n - done) / max(speed, 1e-6)
            cur_gt_asr = (gt_success_count / max(1, gt_detected_clean)) * 100.0
            print(
                f"  [AdamPGD检测白盒 第{run_id}次] 进度 {done}/{n} ({done/max(1,n)*100:.1f}%) | "
                f"当前GT-ASR={cur_gt_asr:.2f}% | 用时={elapsed/60:.1f}m | ETA={eta/60:.1f}m"
            )

    clean_metrics = compute_model_metrics(c_tp, c_fp, c_fn, 0)
    adv_metrics = compute_model_metrics(a_tp, a_fp, a_fn, 0)
    gt_asr = gt_success_count / max(1, gt_detected_clean)
    img_asr = img_success_count / max(1, img_detected_clean)
    pert = compute_perturbation_metrics(clean_imgs, adv_imgs, gt_asr)
    return {"clean": clean_metrics, "adv": adv_metrics, "gt_asr": gt_asr, "img_asr": img_asr, "pert": pert}


def main():
    p = argparse.ArgumentParser(
        description="VOC YOLO 白盒（脚本名 mifgsm；内核为 L∞ Adam-PGD + yolo_whitebox_objective）"
    )
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.0010)
    p.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam 一阶矩衰减系数（常用 0.9）",
    )
    p.add_argument(
        "--mu",
        type=float,
        default=None,
        help="已弃用：若指定则覆盖 --beta1，兼容旧命令（原 MI-FGSM 动量参数）",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=400,
        help="yolo_whitebox_objective 中参与 top-k 的锚点数",
    )
    p.add_argument(
        "--target_conf",
        type=float,
        default=None,
        help="白盒目标阈值项；默认与 --conf 一致",
    )
    p.add_argument("--min_steps", type=int, default=40)
    p.add_argument(
        "--stop_loss",
        type=float,
        default=-1.0,
        help="早停阈值；<0 关闭",
    )
    p.add_argument(
        "--di_prob",
        type=float,
        default=0.35,
        help="输入多样化概率（检测上不宜过大）",
    )
    p.add_argument("--di_scale_min", type=float, default=0.95)
    p.add_argument("--ti_kernel", type=int, default=5)
    p.add_argument("--ti_sigma", type=float, default=1.0)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--random_start", action="store_true", default=True)
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--eval_set", type=str, default="test", choices=["train", "val", "trainval", "test"])
    p.add_argument("--num_eval", type=int, default=500)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--load_model", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./adv_outputs/mifgsm")
    p.add_argument("--log_every", type=int, default=20)
    args = p.parse_args()
    if getattr(args, "mu", None) is not None:
        args.beta1 = float(args.mu)
    if args.target_conf is None:
        args.target_conf = float(args.conf)

    if args.eps > 0.05:
        print(f"eps={args.eps} 超过 5%，已截断为 0.05")
        args.eps = 0.05

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", args.device)
    print(
        f"AdamPGD(检测): beta1={args.beta1} alpha={args.alpha} di_prob={args.di_prob} "
        f"di_scale_min={args.di_scale_min} topk={args.topk} restarts={args.restarts} eps={args.eps}",
        flush=True,
    )
    if args.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    os.makedirs(args.outdir, exist_ok=True)
    ds = load_voc2007_dataset(args.eval_set)
    model = build_voc_model(args.device, args.load_model)

    all_r = []
    for i in range(1, args.runs + 1):
        r = evaluate_once(model, ds, args, i)
        all_r.append(r)
        print(f"\n[AdamPGD检测白盒 第{i}次]")
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
    print("AdamPGD 检测白盒汇总（原 whitebox_mifgsm 入口）")
    print("=" * 60)
    print(f"攻击成功率(GT级)均值: {asr.mean()*100:.2f}%")
    print(f"是否满足 >85%: {asr.mean() > 0.85}")


if __name__ == "__main__":
    main()
