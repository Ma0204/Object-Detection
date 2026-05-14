# -*- coding: utf-8 -*-
"""
VOC2007 对抗攻击评估

用法:
  python3 adversarial_attack_evaluation.py --load_model ./checkpoints/voc_model.pth --method pgd --num_eval 500
"""

import os
import argparse
import random
from typing import List, Dict, Tuple

import torch
import numpy as np
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from attack_utils import (
    load_voc2007_trainval_dataset,
    make_pseudo_targets_from_pred,
    ensure_tensor_01,
    voc_target_to_boxes_and_labels,
    compute_iou,
    filter_pred,
)
from evaluation_metrics import (
    compute_model_metrics,
    compute_perturbation_metrics,
    print_model_metrics,
    print_perturbation_metrics,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

VOC_CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]
VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}


def build_voc_model(device: torch.device) -> torch.nn.Module:
    """COCO backbone + VOC 检测头（与 object_detection_voc_base.py 完全一致）"""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=len(VOC_CLASSES))
    model.to(device)
    return model


# ========== 攻击方法 ==========

@torch.no_grad()
def infer(model: torch.nn.Module, img: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    return model([img])[0]


def fgsm_attack(model, img_01, pseudo_targets, eps):
    x = img_01.detach().clone().requires_grad_(True)
    model.train()
    losses = model([x], [pseudo_targets])
    loss = sum(v for v in losses.values())
    loss.backward()
    adv = ensure_tensor_01(x + eps * x.grad.sign())
    return adv.detach()


def pgd_attack(model, img_01, pseudo_targets, eps, steps=10, alpha=0.005):
    x0 = img_01.detach()
    x = x0.clone()
    for _ in range(int(steps)):
        x = x.detach().clone().requires_grad_(True)
        model.train()
        losses = model([x], [pseudo_targets])
        loss = sum(v for v in losses.values())
        loss.backward()
        with torch.no_grad():
            x = x + alpha * x.grad.sign()
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)
            x = ensure_tensor_01(x)
    return x.detach()


def mifgsm_attack(model, img_01, pseudo_targets, eps, steps=10, alpha=0.005, mu=1.0):
    x0 = img_01.detach()
    x = x0.clone()
    g = torch.zeros_like(x0)
    for _ in range(int(steps)):
        x = x.detach().clone().requires_grad_(True)
        model.train()
        losses = model([x], [pseudo_targets])
        loss = sum(v for v in losses.values())
        loss.backward()
        grad = x.grad.detach()
        grad_norm = grad.abs().mean(dim=(0, 1, 2), keepdim=True).clamp_min(1e-12)
        grad = grad / grad_norm
        g = mu * g + grad
        with torch.no_grad():
            x = x + alpha * g.sign()
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)
            x = ensure_tensor_01(x)
    return x.detach()


# ========== 评估函数 ==========

def evaluate_adversarial_attack(
    model: torch.nn.Module,
    dataset,
    attack_fn,
    attack_name: str,
    num_eval: int = 500,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.5,
) -> Tuple[Dict, Dict]:
    """评估对抗攻击效果（VOC 类别匹配，随机采样）"""

    clean_tp = clean_fp = clean_fn = 0
    adv_tp = adv_fp = adv_fn = 0
    attack_success_count = total_attackable = 0
    clean_imgs_list: List[torch.Tensor] = []
    adv_imgs_list: List[torch.Tensor] = []

    total_samples = len(dataset)
    n = min(int(num_eval), total_samples)
    sample_indices = random.sample(range(total_samples), n)
    print(f"\n随机评估 {n} 张图像 ({attack_name})...")

    for step, idx in enumerate(sample_indices):
        img_tensor, target = dataset[idx]
        img_tensor = img_tensor.to(device)
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)

        with torch.no_grad():
            pred_clean = infer(model, img_tensor)

        fp_clean = filter_pred(pred_clean, conf_thresh=conf_thresh)
        pred_boxes_c = fp_clean["boxes"].cpu()
        pred_labels_c = fp_clean["labels"].cpu().tolist()

        clean_matched = 0
        for g_box, g_name in zip(gt_boxes, gt_labels):
            if g_name not in VOC_CLASS_TO_ID:
                continue
            gt_lid = VOC_CLASS_TO_ID[g_name]
            g_box_list = g_box.tolist()
            matched = any(
                int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= iou_thresh
                for pb, pl in zip(pred_boxes_c, pred_labels_c)
            )
            if matched:
                clean_tp += 1
                clean_matched += 1
            else:
                clean_fn += 1
        clean_fp += max(0, len(pred_labels_c) - clean_matched)

        pseudo = make_pseudo_targets_from_pred(pred_clean, conf_thresh=conf_thresh, topk=30)
        adv_img = attack_fn(img_tensor, pseudo)
        adv_img = ensure_tensor_01(adv_img).to(device)

        with torch.no_grad():
            pred_adv = infer(model, adv_img)

        fp_adv = filter_pred(pred_adv, conf_thresh=conf_thresh)
        pred_boxes_a = fp_adv["boxes"].cpu()
        pred_labels_a = fp_adv["labels"].cpu().tolist()

        adv_matched = 0
        for g_box, g_name in zip(gt_boxes, gt_labels):
            if g_name not in VOC_CLASS_TO_ID:
                continue
            gt_lid = VOC_CLASS_TO_ID[g_name]
            g_box_list = g_box.tolist()
            matched = any(
                int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= iou_thresh
                for pb, pl in zip(pred_boxes_a, pred_labels_a)
            )
            if matched:
                adv_tp += 1
                adv_matched += 1
            else:
                adv_fn += 1
        adv_fp += max(0, len(pred_labels_a) - adv_matched)

        if clean_matched > 0:
            total_attackable += 1
            if adv_matched < clean_matched:
                attack_success_count += 1

        clean_imgs_list.append(img_tensor.detach().cpu())
        adv_imgs_list.append(adv_img.detach().cpu())

        if (step + 1) % 50 == 0:
            print(f"  已处理 {step + 1}/{n} 张图像")

    clean_metrics = compute_model_metrics(clean_tp, clean_fp, clean_fn, 0)
    adv_metrics = compute_model_metrics(adv_tp, adv_fp, adv_fn, 0)
    attack_success_rate = attack_success_count / max(1, total_attackable)
    pert_metrics = compute_perturbation_metrics(clean_imgs_list, adv_imgs_list, attack_success_rate)

    return (
        {"model_metrics": clean_metrics, "attack_success_count": attack_success_count, "total_attackable": total_attackable},
        {"model_metrics": adv_metrics, "perturbation_metrics": pert_metrics},
    )


def main():
    parser = argparse.ArgumentParser(description="对抗攻击评估")
    parser.add_argument("--method", type=str, default="pgd", choices=["fgsm", "pgd", "mifgsm"])
    parser.add_argument("--eps", type=float, default=0.05, help="扰动幅度")
    parser.add_argument("--steps", type=int, default=10, help="迭代步数")
    parser.add_argument("--alpha", type=float, default=0.005, help="步长")
    parser.add_argument("--num_eval", type=int, default=500, help="评估样本数")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--outdir", type=str, default="./adv_eval_results", help="输出目录")
    parser.add_argument("--load_model", type=str, default=None,
                        help="加载已训练模型路径（推荐用 object_detection_voc_base.py 训练好的）")
    args = parser.parse_args()

    if args.eps > 0.05:
        print(f"警告：eps {args.eps} 超过 0.05，已调整为 0.05")
        args.eps = 0.05

    print("加载数据集...")
    ds = load_voc2007_trainval_dataset()

    print("构建模型（COCO backbone + VOC 检测头）...")
    model = build_voc_model(device)

    if args.load_model is not None:
        print(f"加载已有模型: {args.load_model}")
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        model.eval()
    else:
        print("警告：未指定 --load_model，建议先用 object_detection_voc_base.py 训练并保存模型")
        model.eval()

    if args.method == "fgsm":
        def attack_fn(img, pseudo): return fgsm_attack(model, img, pseudo, eps=args.eps)
    elif args.method == "pgd":
        def attack_fn(img, pseudo): return pgd_attack(model, img, pseudo, eps=args.eps, steps=args.steps, alpha=args.alpha)
    else:
        def attack_fn(img, pseudo): return mifgsm_attack(model, img, pseudo, eps=args.eps, steps=args.steps, alpha=args.alpha)

    clean_results, adv_results = evaluate_adversarial_attack(
        model=model, dataset=ds, attack_fn=attack_fn,
        attack_name=args.method.upper(), num_eval=args.num_eval, conf_thresh=args.conf,
    )

    print("\n" + "="*60)
    print(f"对抗攻击评估结果 - {args.method.upper()}")
    print("="*60)
    print(f"扰动幅度 (eps): {args.eps:.4f} ({args.eps*100:.2f}%)")
    print(f"迭代步数: {args.steps}  步长: {args.alpha}")
    print_model_metrics(clean_results["model_metrics"], "（原始图像）")
    print_model_metrics(adv_results["model_metrics"], "（对抗图像）")
    print_perturbation_metrics(adv_results["perturbation_metrics"], f"（{args.method.upper()}）")

    asr = adv_results["perturbation_metrics"].attack_success_rate
    print("\n" + "="*60)
    print("项目要求检查")
    print("="*60)
    eps_ok = args.eps <= 0.05
    asr_ok = asr > 0.85
    print(f"扰动幅度 <= 5%: {eps_ok}  ({args.eps*100:.2f}%)")
    print(f"攻击成功率 > 85%: {asr_ok}  ({asr*100:.2f}%)")
    if eps_ok and asr_ok:
        print("满足所有项目要求！")
    else:
        print("未完全满足要求，可调整 --eps 或 --steps")

    os.makedirs(args.outdir, exist_ok=True)
    result_file = os.path.join(args.outdir, f"{args.method}_results.txt")
    cm = clean_results["model_metrics"]
    am = adv_results["model_metrics"]
    pm = adv_results["perturbation_metrics"]
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"对抗攻击评估结果 - {args.method.upper()}\n")
        f.write("="*60 + "\n")
        f.write(f"eps: {args.eps}  steps: {args.steps}  alpha: {args.alpha}\n\n")
        f.write(f"[原始] 精确率:{cm.precision*100:.2f}%  召回率:{cm.recall*100:.2f}%  "
                f"F-score:{cm.f_score:.4f}  准确率:{cm.accuracy*100:.2f}%\n")
        f.write(f"[对抗] 精确率:{am.precision*100:.2f}%  召回率:{am.recall*100:.2f}%  "
                f"F-score:{am.f_score:.4f}  准确率:{am.accuracy*100:.2f}%\n")
        f.write(f"攻击成功率: {pm.attack_success_rate*100:.2f}%\n")
        f.write(f"平均L2: {pm.avg_l2_distance:.6f}  Linf: {pm.avg_linf_distance:.6f}\n")
        f.write(f"MSE: {pm.avg_mse:.6f}  SSIM: {pm.avg_ssim:.4f}  PSNR: {pm.avg_psnr:.2f}dB\n")
    print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
