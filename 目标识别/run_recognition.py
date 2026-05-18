# -*- coding: utf-8 -*-
"""
「场景类别表」实验统一入口：白盒 / 黑盒 / 噪声。

用法示例：
  python run_recognition.py attack --strategy fgsm --weights yolov8n.pt
  python run_recognition.py attack --strategy ttba --weights yolov8n.pt --num_eval 30
  python run_recognition.py noise --weights yolov8n.pt
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from attack_utils import build_yolo_voc_model, load_voc2007_dataset
from recognition_benchmark import (
    run_table_attack_blackbox_adba,
    run_table_attack_blackbox_seri,
    run_table_attack_blackbox_ttba,
    run_table_attack_whitebox,
)
from recognition_perturb_whitebox import perturb_scene_whitebox


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--weights", type=str, required=True, help="与主工程相同的 YOLO 权重路径")
    p.add_argument("--eval_set", type=str, default="test", choices=["train", "val", "trainval", "test"])
    p.add_argument("--num_eval", type=int, default=200)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--log_every", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.2, help="预测表聚合时的分数门限")
    p.add_argument("--infer_imgsz", type=int, default=640, help="前向边长（内部用）")
    p.add_argument("--iou", type=float, default=0.5, help="黑盒构造参考时 hybrid 与 GT 的 IoU 阈值")


def _add_whitebox_hparams(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--alpha", type=float, default=0.0012)
    p.add_argument("--target_conf", type=float, default=None)
    p.add_argument("--min_steps", type=int, default=35)
    p.add_argument("--stop_loss", type=float, default=-1.0)
    p.add_argument("--di_prob", type=float, default=0.5)
    p.add_argument("--di_scale_min", type=float, default=0.92)
    p.add_argument("--ti_kernel", type=int, default=5)
    p.add_argument("--ti_sigma", type=float, default=1.0)
    p.add_argument("--random_start", action="store_true", default=False)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--topk_surrogate", type=int, default=300)
    p.add_argument("--beta1", type=float, default=0.9)


def cmd_attack(ns: argparse.Namespace) -> None:
    random.seed(ns.seed)
    np.random.seed(ns.seed)
    torch.manual_seed(ns.seed)
    if ns.target_conf is None:
        ns.target_conf = float(ns.conf)
    if ns.eps > 0.05:
        ns.eps = 0.05
    ns.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ns.outdir, exist_ok=True)
    ds = load_voc2007_dataset(ns.eval_set)
    model = build_yolo_voc_model(ns.device, weights=ns.weights)

    strat = str(ns.strategy).lower().strip()

    scene_perturber = None
    if strat in ("fgsm", "pgd", "adam"):

        def scene_perturber(m: torch.nn.Module, x: torch.Tensor, _ns=ns, _st=strat) -> torch.Tensor:
            if _st == "adam":
                return perturb_scene_whitebox(
                    m,
                    x,
                    "adam",
                    eps=_ns.eps,
                    steps=_ns.steps,
                    alpha=_ns.alpha,
                    random_start=_ns.random_start,
                    target_conf=_ns.target_conf,
                    min_steps=_ns.min_steps,
                    stop_loss=_ns.stop_loss,
                    di_prob=_ns.di_prob,
                    di_scale_min=_ns.di_scale_min,
                    ti_kernel=_ns.ti_kernel,
                    ti_sigma=_ns.ti_sigma,
                    restarts=int(_ns.restarts),
                    topk_surrogate=int(_ns.topk_surrogate),
                    beta1=float(_ns.beta1),
                )
            return perturb_scene_whitebox(
                m,
                x,
                "pgd" if _st == "pgd" else "fgsm",
                eps=_ns.eps,
                steps=_ns.steps,
                alpha=_ns.alpha,
                random_start=_ns.random_start,
                target_conf=_ns.target_conf,
                min_steps=_ns.min_steps,
                stop_loss=_ns.stop_loss,
                di_prob=_ns.di_prob,
                di_scale_min=_ns.di_scale_min,
                ti_kernel=_ns.ti_kernel,
                ti_sigma=_ns.ti_sigma,
            )

    rows = []
    for run in range(1, ns.runs + 1):
        if strat in ("fgsm", "pgd", "adam"):
            assert scene_perturber is not None
            r = run_table_attack_whitebox(model, ds, ns, run, scene_perturber)
        elif strat == "ttba":
            r = run_table_attack_blackbox_ttba(model, ds, ns, run)
        elif strat == "seri":
            r = run_table_attack_blackbox_seri(model, ds, ns, run)
        elif strat == "adba":
            r = run_table_attack_blackbox_adba(model, ds, ns, run)
        else:
            raise SystemExit(f"未知 --strategy: {ns.strategy}")

        rows.append(r)
        print(
            f"\n[run {run}] 有效场景={r['n_scenes']} | 干净表准确率={r['clean_table_accuracy']*100:.2f}% | "
            f"受限表破坏率={r['restricted_table_break_rate']*100:.2f}%"
        )
        p = r["pert"]
        print(f"  像素侧: L2={p.avg_l2_distance:.4f} Linf={p.avg_linf_distance:.4f} MSE={p.avg_mse:.6f}")

    mean_break = float(np.mean([x["restricted_table_break_rate"] for x in rows]) * 100.0)
    print("\n=== 汇总：受限表破坏率（均值）===")
    print(f"{mean_break:.2f}%")


def cmd_noise(ns: argparse.Namespace) -> None:
    from recognition_core import predicted_table_from_scene, tables_match, truth_table_from_voc_target

    random.seed(ns.seed)
    np.random.seed(ns.seed)
    torch.manual_seed(ns.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_yolo_voc_model(device, weights=ns.weights)
    ds = load_voc2007_dataset(ns.eval_set)
    n = min(int(ns.num_eval), len(ds))
    idxs = random.sample(range(len(ds)), n)

    def cap(clean: torch.Tensor, noisy: torch.Tensor, eps: float) -> torch.Tensor:
        if eps <= 0:
            return noisy
        d = (noisy - clean).clamp(-eps, eps)
        return (clean + d).clamp(0.0, 1.0)

    ok0 = ok1 = used = 0
    for i in idxs:
        img, target = ds[i]
        img = img.to(device)
        truth = truth_table_from_voc_target(target)
        if not truth:
            continue
        used += 1
        h0 = predicted_table_from_scene(model, img, score_cutoff=float(ns.conf), image_side=int(ns.infer_imgsz))
        if tables_match(truth, h0):
            ok0 += 1
        noise = torch.randn_like(img) * float(ns.gauss_eps)
        img2 = cap(img, img + noise, float(ns.gauss_eps))
        h1 = predicted_table_from_scene(model, img2, score_cutoff=float(ns.conf), image_side=int(ns.infer_imgsz))
        if tables_match(truth, h1):
            ok1 += 1
    print(f"有效场景 {used}")
    print(f"干净表匹配: {ok0/max(1,used)*100:.2f}%")
    print(f"加噪后表匹配: {ok1/max(1,used)*100:.2f}%")


def main() -> None:
    root = argparse.ArgumentParser(description="场景类别表（目标识别简化版）")
    sub = root.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("attack", help="对场景图做扰动，再比较类别表")
    _add_shared(pa)
    pa.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["fgsm", "pgd", "adam", "ttba", "seri", "adba"],
        help="白盒：像素梯度多步；黑盒：查询式 A/B/C（实现核与主工程共享）",
    )
    pa.add_argument("--outdir", type=str, default="./outputs_recognition_attack")
    _add_whitebox_hparams(pa)

    pa.add_argument("--outer_steps", type=int, default=48)
    pa.add_argument("--bridge_iters", type=int, default=6)
    pa.add_argument("--bridge_shrink_tol", type=float, default=0.04)
    pa.add_argument("--no_random_start", action="store_true")
    pa.add_argument("--topk", type=int, default=256)
    pa.add_argument("--ref_topk", type=int, default=120)
    pa.add_argument("--ref_mode", type=str, default="hybrid", choices=["pred", "gt", "hybrid"])
    pa.add_argument("--ref_conf", type=float, default=0.22)
    pa.add_argument("--iou_match", type=float, default=0.5)
    pa.add_argument("--attack_restarts", type=int, default=5)
    pa.add_argument("--objective_conf", type=float, default=None)

    pa.add_argument("--block", type=int, default=48)
    pa.add_argument("--top_blocks", type=int, default=14)
    pa.add_argument("--sens_delta", type=float, default=0.42)
    pa.add_argument("--bilateral_sens", type=int, default=1, choices=[0, 1])
    pa.add_argument("--ref_prior_weight", type=float, default=0.85)
    pa.add_argument("--ref_mask_expand", type=float, default=0.22)
    pa.add_argument("--refine_line_points", type=int, default=5)
    pa.add_argument("--attack_random_init", action="store_true")
    pa.add_argument("--refine_steps", type=int, default=120)
    pa.add_argument("--refine_alpha", type=float, default=0.018)

    pa.add_argument("--steps_adba", type=int, default=160)
    pa.add_argument("--probe_scale", type=float, default=0.33)
    pa.add_argument("--alpha_adba", type=float, default=0.019)
    pa.add_argument("--stop_ratio", type=float, default=0.0)
    pa.add_argument("--eps_max", type=float, default=0.05)
    pa.add_argument("--gt_focus_expand", type=float, default=0.0)
    pa.add_argument("--adb_ray_refine_iters", type=int, default=2)
    pa.add_argument("--adb_allow_uphill", action="store_true")
    pa.add_argument("--max_queries_per_image", type=int, default=0)
    pa.add_argument("--adba_sequential_gt", action="store_true")
    pa.add_argument("--adba_per_gt_local", action="store_true")
    pa.add_argument("--per_gt_steps", type=int, default=48)
    pa.add_argument("--adba_aux_fp", type=float, default=0.0)
    pa.add_argument("--adba_aux_misclass", type=float, default=0.0)
    pa.add_argument("--adba_aux_topk", type=int, default=64)

    pn = sub.add_parser("noise", help="仅加噪，看类别表是否仍与真值一致")
    _add_shared(pn)
    pn.add_argument("--gauss_eps", type=float, default=0.08)

    ns = root.parse_args()
    if ns.cmd == "attack":
        if str(ns.strategy) == "adba":
            em = float(ns.eps_max)
            if ns.eps > em:
                ns.eps = em
            ns.steps = int(ns.steps_adba)
            ns.alpha = float(ns.alpha_adba)
        cmd_attack(ns)
    elif ns.cmd == "noise":
        cmd_noise(ns)


if __name__ == "__main__":
    main()
