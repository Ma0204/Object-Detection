# -*- coding: utf-8 -*-
"""
VOC2007 噪声鲁棒性实验（YOLO 版本）

加噪后默认将扰动限制在 L∞ 半径内（与对抗攻击 eps 语义一致），可用 --linf_eps 调整；≤0 关闭。
"""

import os
import argparse
import random
import math
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from PIL import ImageDraw
from torchvision import transforms

from attack_utils import (
    VOC_CLASSES,
    load_voc2007_dataset,
    build_yolo_voc_model,
    infer_yolo,
    voc_target_to_boxes_and_labels,
    compute_iou,
    filter_pred,
)
from evaluation_metrics import compute_model_metrics

# 默认写入本脚本所在工程目录下的 voc_outputs_noise（与 cwd 无关）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOC_NOISE_OUTDIR = os.path.join(_SCRIPT_DIR, "voc_outputs_noise")

VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def _safe_noise_dir_name(noise_name: str) -> str:
    s = noise_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    return s[:180] if len(s) > 180 else s


def _preview_dir(output_dir: str, noise_name: str) -> str:
    d = os.path.join(output_dir, "noise_previews", _safe_noise_dir_name(noise_name))
    os.makedirs(d, exist_ok=True)
    return d


def _save_noisy_preview(path: str, img_chw_01: torch.Tensor) -> None:
    t = img_chw_01.detach().float().cpu().clamp(0.0, 1.0)
    transforms.ToPILImage()(t).save(path)


def apply_linf_noise_cap(
    img_clean: torch.Tensor, img_noisy: torch.Tensor, linf_eps: float
) -> torch.Tensor:
    """|x' - x|_∞ ≤ linf_eps，再 [0,1] 裁剪；与项目对抗扰动 L∞ 球一致。"""
    if linf_eps <= 0.0:
        return img_noisy
    d = img_noisy - img_clean
    d = d.clamp(-float(linf_eps), float(linf_eps))
    return (img_clean + d).clamp(0.0, 1.0)


class AddGaussianNoise:
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.sigma
        return torch.clamp(img + noise, 0.0, 1.0)


class AddSaltPepperNoise:
    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape
        rand = torch.rand((h, w), device=img.device)
        noisy = img.clone()
        salt = rand < (self.p / 2.0)
        pepper = (rand >= (self.p / 2.0)) & (rand < self.p)
        for i in range(c):
            noisy[i][salt] = 1.0
            noisy[i][pepper] = 0.0
        return noisy


class AddRainNoise:
    def __init__(
        self,
        drop_count: int = 1200,
        length: int = 18,
        thickness: int = 1,
        angle_deg: float = -15.0,
        intensity: float = 0.35,
        seed: Optional[int] = None,
    ):
        self.drop_count = drop_count
        self.length = length
        self.thickness = thickness
        self.angle_deg = angle_deg
        self.intensity = intensity
        self.seed = seed

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape
        img_pil = transforms.ToPILImage()(img.cpu())
        draw = ImageDraw.Draw(img_pil, mode="RGBA")
        rng = np.random.default_rng(self.seed)
        theta = np.deg2rad(self.angle_deg)
        dx = int(round(self.length * np.cos(theta)))
        dy = int(round(self.length * np.sin(theta)))
        alpha = int(max(0, min(255, round(255 * self.intensity))))
        color = (220, 220, 220, alpha)
        for _ in range(self.drop_count):
            x = int(rng.integers(0, max(1, w)))
            y = int(rng.integers(0, max(1, h)))
            x2 = int(max(0, min(w - 1, x + dx)))
            y2 = int(max(0, min(h - 1, y + dy)))
            draw.line([(x, y), (x2, y2)], fill=color, width=self.thickness)
        noisy = transforms.ToTensor()(img_pil)
        return torch.clamp(noisy, 0.0, 1.0)


def compute_perturbation_metrics_from_pairs(clean_imgs: List[torch.Tensor], adv_imgs: List[torch.Tensor]) -> Dict[str, float]:
    if not clean_imgs:
        return {"l2": 0.0, "l2_full": 0.0, "linf": 0.0, "mse": 0.0, "ssim": 0.0, "psnr": 0.0}

    l2_rms_list, l2_full_list, linf_list, mse_list, ssim_list, psnr_list = [], [], [], [], [], []
    for c, a in zip(clean_imgs, adv_imgs):
        diff = (a - c).float()
        n = int(diff.numel())
        l2_full = float(diff.norm(p=2).item())
        l2_rms = l2_full / math.sqrt(max(1, n))
        linf = float(diff.abs().max().item())
        mse = float((diff ** 2).mean().item())
        psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10))

        c_np = c.cpu().numpy()
        a_np = a.cpu().numpy()
        ssim_val = 0.0
        for ch in range(c_np.shape[0]):
            mu1 = c_np[ch].mean()
            mu2 = a_np[ch].mean()
            sigma1 = c_np[ch].std()
            sigma2 = a_np[ch].std()
            sigma12 = ((c_np[ch] - mu1) * (a_np[ch] - mu2)).mean()
            k1, k2 = 0.01, 0.03
            c1, c2 = (k1 ** 2), (k2 ** 2)
            ssim_ch = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
                (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2)
            )
            ssim_val += ssim_ch
        ssim_val /= c_np.shape[0]

        l2_rms_list.append(l2_rms)
        l2_full_list.append(l2_full)
        linf_list.append(linf)
        mse_list.append(mse)
        ssim_list.append(float(ssim_val))
        psnr_list.append(float(psnr))

    return {
        "l2": float(np.mean(l2_rms_list)),
        "l2_full": float(np.mean(l2_full_list)),
        "linf": float(np.mean(linf_list)),
        "mse": float(np.mean(mse_list)),
        "ssim": float(np.mean(ssim_list)),
        "psnr": float(np.mean(psnr_list)),
    }


def evaluate_noise_once(
    model,
    dataset,
    noise_transform,
    num_eval: int = 1000,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    imgsz: int = 960,
    tta: bool = False,
    log_every: int = 200,
    run_id: int = 1,
    noise_name: str = "noise",
    output_dir: Optional[str] = None,
    save_preview_n: int = 5,
    linf_eps: float = 0.05,
) -> Dict:
    total_samples = len(dataset)
    num_eval = min(int(num_eval), total_samples)
    sample_indices = random.sample(range(total_samples), num_eval)

    c_tp = c_fp = c_fn = 0
    n_tp = n_fp = n_fn = 0
    gt_detected_clean = gt_success_count = 0
    img_detected_clean = img_success_count = 0
    collect_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    max_pairs = 200
    preview_dir_path: Optional[str] = None

    def count_metrics(gt_boxes, gt_labels, pred_boxes_t, pred_labels_t):
        tp_here = fn_here = matched_here = 0
        for g_box, g_name in zip(gt_boxes, gt_labels):
            if g_name not in VOC_CLASS_TO_ID:
                continue
            gt_lid = VOC_CLASS_TO_ID[g_name]
            g_box_list = g_box.tolist()
            found = False
            for p_box, p_lid in zip(pred_boxes_t, pred_labels_t):
                if int(p_lid) == int(gt_lid) and compute_iou(g_box_list, p_box.tolist()) >= iou_thresh:
                    tp_here += 1
                    matched_here += 1
                    found = True
                    break
            if not found:
                fn_here += 1
        fp_here = max(0, len(pred_labels_t) - matched_here)
        return tp_here, fp_here, fn_here, matched_here

    for step, idx in enumerate(sample_indices):
        img_tensor, target = dataset[idx]
        img_clean = img_tensor.to(device)
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)

        pred_clean = infer_yolo(model, img_clean, imgsz=imgsz, use_tta=tta)
        fp_c = filter_pred(pred_clean, conf_thresh=conf_thresh)
        pred_labels_c = fp_c["labels"].cpu().tolist()
        pred_boxes_c = fp_c["boxes"].cpu()

        img_noisy = noise_transform(img_tensor).to(device)
        img_noisy = apply_linf_noise_cap(img_clean, img_noisy, float(linf_eps))
        if (
            int(save_preview_n) > 0
            and output_dir
            and int(run_id) == 1
            and step < int(save_preview_n)
        ):
            if preview_dir_path is None:
                preview_dir_path = _preview_dir(output_dir, noise_name)
            png_path = os.path.join(
                preview_dir_path, f"noisy_{step + 1:02d}_dsidx{idx:05d}.png"
            )
            _save_noisy_preview(png_path, img_noisy)
        pred_noisy = infer_yolo(model, img_noisy, imgsz=imgsz, use_tta=tta)
        fp_n = filter_pred(pred_noisy, conf_thresh=conf_thresh)
        pred_labels_n = fp_n["labels"].cpu().tolist()
        pred_boxes_n = fp_n["boxes"].cpu()

        c_tp_h, c_fp_h, c_fn_h, c_m = count_metrics(gt_boxes, gt_labels, pred_boxes_c, pred_labels_c)
        n_tp_h, n_fp_h, n_fn_h, n_m = count_metrics(gt_boxes, gt_labels, pred_boxes_n, pred_labels_n)
        c_tp += c_tp_h
        c_fp += c_fp_h
        c_fn += c_fn_h
        n_tp += n_tp_h
        n_fp += n_fp_h
        n_fn += n_fn_h

        for g_box, g_name in zip(gt_boxes, gt_labels):
            if g_name not in VOC_CLASS_TO_ID:
                continue
            gt_lid = VOC_CLASS_TO_ID[g_name]
            g_box_list = g_box.tolist()
            c_hit = any(
                int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= iou_thresh
                for pb, pl in zip(pred_boxes_c, pred_labels_c)
            )
            if c_hit:
                gt_detected_clean += 1
                n_hit = any(
                    int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= iou_thresh
                    for pb, pl in zip(pred_boxes_n, pred_labels_n)
                )
                if not n_hit:
                    gt_success_count += 1

        if c_m > 0 and gt_boxes.size(0) > 0:
            img_detected_clean += 1
            if n_m == 0:
                img_success_count += 1

        if len(collect_pairs) < max_pairs:
            collect_pairs.append((img_clean.cpu(), img_noisy.cpu()))

        if (step + 1) % max(1, int(log_every)) == 0:
            print(f"  [{noise_name} 第{run_id}次] 已处理 {step + 1}/{num_eval}")

    if preview_dir_path is not None:
        print(
            f"  加噪预览图已保存（本轮抽样中前 {int(save_preview_n)} 张）: {os.path.abspath(preview_dir_path)}",
            flush=True,
        )

    c_metrics = compute_model_metrics(c_tp, c_fp, c_fn, 0)
    n_metrics = compute_model_metrics(n_tp, n_fp, n_fn, 0)
    gt_attack_rate = gt_success_count / max(1, gt_detected_clean)
    img_attack_rate = img_success_count / max(1, img_detected_clean)
    pert = compute_perturbation_metrics_from_pairs([p[0] for p in collect_pairs], [p[1] for p in collect_pairs])

    return {
        "run_id": run_id,
        "clean": c_metrics,
        "noisy": n_metrics,
        "gt_attack_rate": gt_attack_rate,
        "img_attack_rate": img_attack_rate,
        "gt_success_count": gt_success_count,
        "gt_detected_clean": gt_detected_clean,
        "img_success_count": img_success_count,
        "img_detected_clean": img_detected_clean,
        "pert": pert,
    }


def run_noise_experiment(
    model,
    dataset,
    noise_transform,
    noise_name: str,
    num_runs: int,
    num_eval: int,
    conf_thresh: float,
    iou_thresh: float,
    imgsz: int,
    tta: bool,
    log_every: int,
    output_dir: str,
    save_preview_n: int = 5,
    linf_eps: float = 0.05,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"噪声类型：{noise_name}（YOLO）")
    print(f"{'#' * 60}")
    if float(linf_eps) > 0.0:
        print(f"  L∞ 扰动上限 |x'-x|_∞ ≤ {float(linf_eps)}（与对抗 eps 语义一致）", flush=True)
    else:
        print("  L∞ 扰动上限：无（--linf_eps≤0，与旧版无约束一致）", flush=True)
    if int(save_preview_n) > 0:
        sub = _safe_noise_dir_name(noise_name)
        prev_abs = os.path.join(os.path.abspath(output_dir), "noise_previews", sub)
        print(
            f"  预览图将写入(绝对路径): {prev_abs}/ "
            f"(noisy_01_*.png … 共前 {int(save_preview_n)} 张，仅 run=1)",
            flush=True,
        )

    results = []
    for run_id in range(1, num_runs + 1):
        r = evaluate_noise_once(
            model=model,
            dataset=dataset,
            noise_transform=noise_transform,
            num_eval=num_eval,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            imgsz=imgsz,
            tta=tta,
            log_every=log_every,
            run_id=run_id,
            noise_name=noise_name,
            output_dir=output_dir,
            save_preview_n=save_preview_n,
            linf_eps=float(linf_eps),
        )
        results.append(r)
        print(f"\n第 {run_id} 次结果 [{noise_name}]：")
        print(f"  [干净] P={r['clean'].precision*100:.2f}% R={r['clean'].recall*100:.2f}% F={r['clean'].f_score:.4f} Acc={r['clean'].accuracy*100:.2f}%")
        print(f"  [噪声] P={r['noisy'].precision*100:.2f}% R={r['noisy'].recall*100:.2f}% F={r['noisy'].f_score:.4f} Acc={r['noisy'].accuracy*100:.2f}%")
        print(f"  ASR(GT/图像): {r['gt_attack_rate']*100:.2f}% / {r['img_attack_rate']*100:.2f}%")
        print(
            f"  L2_rms={r['pert']['l2']:.4f} (sqrt(均方差),[0,1]尺度)  L2_full={r['pert']['l2_full']:.2f} (全图Frobenius) "
            f"Linf={r['pert']['linf']:.4f} MSE={r['pert']['mse']:.6f} SSIM={r['pert']['ssim']:.4f} PSNR={r['pert']['psnr']:.2f}"
        )

    safe_name = noise_name.replace("/", "_").replace(" ", "_")
    report_file = os.path.join(output_dir, f"report_yolo_{safe_name}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"YOLO 噪声实验报告：{noise_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"runs={num_runs}, num_eval={num_eval}, conf={conf_thresh}, iou={iou_thresh}, linf_eps={linf_eps}\n")
        if int(save_preview_n) > 0:
            prev = os.path.join(output_dir, "noise_previews", _safe_noise_dir_name(noise_name))
            f.write(f"noise_preview_png: 前{int(save_preview_n)}张 -> {prev}/\n")
        f.write("\n")
        for r in results:
            f.write(
                f"run={r['run_id']} clean(P/R/F/Acc)=({r['clean'].precision*100:.2f}%/{r['clean'].recall*100:.2f}%/{r['clean'].f_score:.4f}/{r['clean'].accuracy*100:.2f}%) "
                f"noisy(P/R/F/Acc)=({r['noisy'].precision*100:.2f}%/{r['noisy'].recall*100:.2f}%/{r['noisy'].f_score:.4f}/{r['noisy'].accuracy*100:.2f}%) "
                f"ASR(GT/IMG)=({r['gt_attack_rate']*100:.2f}%/{r['img_attack_rate']*100:.2f}%) "
                f"L2_rms={r['pert']['l2']:.4f} L2_full={r['pert']['l2_full']:.2f} "
                f"Linf={r['pert']['linf']:.4f} MSE={r['pert']['mse']:.6f} SSIM={r['pert']['ssim']:.4f} PSNR={r['pert']['psnr']:.2f}\n"
            )
    print(f"报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="VOC2007 噪声鲁棒性实验（YOLO）")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_VOC_NOISE_OUTDIR,
        help=f"结果与预览图输出目录，默认为本脚本旁 voc_outputs_noise（当前即 {DEFAULT_VOC_NOISE_OUTDIR}）",
    )
    parser.add_argument("--eval_set", type=str, default="test", choices=["train", "val", "trainval", "test"], help="评估集划分")
    parser.add_argument(
        "--load_model",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "checkpoints", "yolo_voc_best_e20.pt"),
        help="YOLO 权重路径；默认本工程 checkpoints/yolo_voc_best_e20.pt，不存在时请显式传入",
    )
    parser.add_argument("--log_every", type=int, default=200, help="测试阶段每多少张打印一次进度")
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=0.10,
        help="高斯噪声标准差（与 ToTensor 后 [0,1] 同量纲）；略增大可明显提高漏检型 ASR",
    )
    parser.add_argument(
        "--salt_pepper_p",
        type=float,
        default=0.05,
        help="椒盐噪声像素比例（越大越强，易超 ε 语义但作鲁棒性对比常用）",
    )
    parser.add_argument("--rain_drop_count", type=int, default=1200, help="雨丝条数")
    parser.add_argument("--rain_intensity", type=float, default=0.35, help="雨丝 alpha 强度 0~1")
    parser.add_argument(
        "--linf_eps",
        type=float,
        default=0.05,
        help="加噪后扰动 L∞ 上限 |x'-x|_∞≤该值，再裁剪到[0,1]；与对抗攻击 eps 对齐。≤0 关闭限制（旧行为）",
    )
    parser.add_argument(
        "--noise_preview_n",
        type=int,
        default=5,
        help="每种噪声保存前 N 张加噪图 PNG 到 outdir/noise_previews/...；0 关闭",
    )
    args = parser.parse_args()
    if float(args.linf_eps) > 0.05:
        print(f"linf_eps={args.linf_eps} 超过 0.05，已截断为 0.05（与项目对抗上限一致）", flush=True)
        args.linf_eps = 0.05
    out_abs = os.path.abspath(args.outdir)
    print(f"当前工作目录 cwd: {os.getcwd()}", flush=True)
    print(f"--outdir 绝对路径: {out_abs}", flush=True)
    print(
        f"加噪预览 PNG 根目录: {os.path.join(out_abs, 'noise_previews')} "
        f"(其下每种噪声一个子文件夹；需 --noise_preview_n>0 且仅第 1 轮 run 写入)",
        flush=True,
    )
    le = float(args.linf_eps)
    if le > 0:
        print(f"L∞ 扰动上限 linf_eps={le}（加噪后再将 |x'-x|_∞ 压到此半径内）", flush=True)
    else:
        print("L∞ 扰动上限：关闭（--linf_eps≤0）", flush=True)

    ds = load_voc2007_dataset(args.eval_set)
    print(f"VOC2007 {args.eval_set} 样本数: {len(ds)}")
    print(f"加载 YOLO 模型: {args.load_model}")
    model = build_yolo_voc_model(device, weights=args.load_model)

    gs = float(args.gaussian_sigma)
    sp = float(args.salt_pepper_p)
    rd = int(args.rain_drop_count)
    ri = float(args.rain_intensity)
    noise_configs = [
        (f"高斯噪声(sigma={gs})", AddGaussianNoise(sigma=gs)),
        (f"椒盐噪声(p={sp})", AddSaltPepperNoise(p=sp)),
        (f"雨丝噪声(count={rd},intensity={ri})", AddRainNoise(drop_count=rd, length=18, intensity=ri)),
    ]

    for noise_name, noise_fn in noise_configs:
        run_noise_experiment(
            model=model,
            dataset=ds,
            noise_transform=noise_fn,
            noise_name=noise_name,
            num_runs=args.runs,
            num_eval=args.num_eval,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            imgsz=args.imgsz,
            tta=args.tta,
            log_every=args.log_every,
            output_dir=args.outdir,
            save_preview_n=int(args.noise_preview_n),
            linf_eps=float(args.linf_eps),
        )

    print("\n所有 YOLO 噪声实验完成。")


if __name__ == "__main__":
    main()
