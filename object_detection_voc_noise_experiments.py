# -*- coding: utf-8 -*-
"""
VOC2007 噪声鲁棒性实验（YOLO 版本）

加噪后默认将扰动限制在 L∞ 半径内（与对抗 eps 一致），可用 --linf_eps 调整；≤0 关闭。
高斯/椒盐在 linf_eps>0 时在球内原生生成；椒盐 B×B 块按边缘显著性抽块，并对块权重做低通调制（类高斯 lowfreq 成簇）；块内 RGB 可独立 ±ε。雨丝单独更高 L∞（--rain_cap_max / --rain_linf_mult）、更密主 pass + 第二遍短丝（--rain_second_pass）、更长线段（--rain_length）；随机噪声 ASR 仍随模型与抽样波动，不保证固定 40%。
"""

import os
import argparse
import random
import math
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
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


def _luminance_sobel_mag_hw(img_chw: torch.Tensor) -> torch.Tensor:
    """灰度 Sobel 幅值 [H,W]，与 img 同 device/dtype。"""
    gray = (0.2989 * img_chw[0] + 0.5870 * img_chw[1] + 0.1140 * img_chw[2]).unsqueeze(0).unsqueeze(0)
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=img_chw.device,
        dtype=img_chw.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=img_chw.device,
        dtype=img_chw.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8).squeeze(0).squeeze(0)


class AddGaussianNoise:
    """
    高斯加性噪声。linf_eps>0：δ~N(0,σ²)，可选低分辨率再上采样（结构化），再截断到 ±linf_eps 后 x+δ。
    linf_eps=0：旧行为 clamp(x+n,0,1)。
    """

    def __init__(self, sigma: float = 0.1, linf_eps: float = 0.0, lowfreq_downscale: int = 1):
        self.sigma = float(sigma)
        self.linf_eps = float(linf_eps)
        self.lowfreq_downscale = max(1, int(lowfreq_downscale))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.linf_eps > 0.0:
            e = self.linf_eps
            c_, h_, w_ = img.shape
            sf = self.lowfreq_downscale
            if sf > 1:
                sh = max(1, h_ // sf)
                sw = max(1, w_ // sf)
                delta = torch.randn((c_, sh, sw), device=img.device, dtype=img.dtype) * self.sigma
                delta = F.interpolate(delta.unsqueeze(0), size=(h_, w_), mode="bilinear", align_corners=False).squeeze(0)
            else:
                delta = torch.randn_like(img) * self.sigma
            delta = delta.clamp(-e, e)
            return (img + delta).clamp(0.0, 1.0)
        noise = torch.randn_like(img) * self.sigma
        return torch.clamp(img + noise, 0.0, 1.0)


class AddSaltPepperNoise:
    """
    椒盐噪声。
    - linf_eps=0：0/1 硬翻转（空间掩膜三通道共用）。
    - linf_eps>0 且 block_size=1：逐位置 ±linf_eps；per_channel 时 RGB 独立掩膜。
    - linf_eps>0 且 block_size>1：B×B 块为单位 ±linf_eps；默认按梯度显著性加权抽块，并对块权重乘低分辨率随机场上采样（成簇，更类高斯 lowfreq）。
      块内 RGB 默认独立随机符号（色度扰动更强）。块数 k≈p·总块数。
    """

    def __init__(
        self,
        p: float = 0.05,
        linf_eps: float = 0.0,
        per_channel: bool = True,
        block_size: int = 1,
        salience_weighted_blocks: bool = True,
        salience_power: float = 1.35,
        block_mask_lowfreq: int = 4,
        independent_channel_signs: bool = True,
        block_count_gain: float = 1.07,
    ):
        self.p = float(p)
        self.linf_eps = float(linf_eps)
        self.per_channel = bool(per_channel)
        self.block_size = max(1, int(block_size))
        self.salience_weighted_blocks = bool(salience_weighted_blocks)
        self.salience_power = float(salience_power)
        self.block_mask_lowfreq = max(1, int(block_mask_lowfreq))
        self.independent_channel_signs = bool(independent_channel_signs)
        self.block_count_gain = max(0.5, min(1.5, float(block_count_gain)))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape
        noisy = img.clone()
        e = float(self.linf_eps)
        p = self.p
        B = self.block_size

        if e > 0.0 and B > 1:
            pad_h = (B - h % B) % B
            pad_w = (B - w % B) % B
            img_p = F.pad(img.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
            _, H, W = img_p.shape
            nbh, nbw = H // B, W // B
            nblocks = nbh * nbw

            if self.salience_weighted_blocks and nblocks > 0:
                mag = _luminance_sobel_mag_hw(img_p)
                mag = mag.view(nbh, B, nbw, B).mean(dim=(1, 3))
                wblk = torch.clamp(mag, min=1e-6) ** self.salience_power
                sf = self.block_mask_lowfreq
                if sf > 1 and nbh >= 2 and nbw >= 2:
                    sh = max(1, nbh // sf)
                    sw = max(1, nbw // sf)
                    aux = torch.rand((1, 1, sh, sw), device=img.device, dtype=img.dtype) * 0.45 + 0.55
                    aux_up = F.interpolate(aux, size=(nbh, nbw), mode="bilinear", align_corners=False)
                    wblk = wblk * aux_up.squeeze(0).squeeze(0)
                flat = wblk.reshape(-1)
                k = int(round(p * float(nblocks) * self.block_count_gain))
                k = max(0, min(nblocks, k))
                if k == 0:
                    return img.clone()
                pr = flat / flat.sum()
                sel = torch.multinomial(pr, num_samples=k, replacement=False)
                perm = torch.randperm(k, device=img.device)
                n_salt = k // 2
                salt_sel = sel[perm[:n_salt]]
                pep_sel = sel[perm[n_salt:]]
                d = torch.zeros_like(img_p)
                if self.independent_channel_signs:
                    for t in salt_sel:
                        by = int(t.item()) // nbw
                        bx = int(t.item()) % nbw
                        y0, x0 = by * B, bx * B
                        sgn = torch.randint(0, 2, (c, 1, 1), device=img.device, dtype=img.dtype).float() * 2.0 - 1.0
                        d[:, y0 : y0 + B, x0 : x0 + B] = sgn * e
                    for t in pep_sel:
                        by = int(t.item()) // nbw
                        bx = int(t.item()) % nbw
                        y0, x0 = by * B, bx * B
                        sgn = torch.randint(0, 2, (c, 1, 1), device=img.device, dtype=img.dtype).float() * 2.0 - 1.0
                        d[:, y0 : y0 + B, x0 : x0 + B] = sgn * e
                else:
                    salt_hw = torch.zeros((H, W), dtype=torch.bool, device=img.device)
                    pep_hw = torch.zeros((H, W), dtype=torch.bool, device=img.device)
                    by_s = salt_sel // nbw
                    bx_s = salt_sel % nbw
                    y0s = by_s * B
                    x0s = bx_s * B
                    by_p = pep_sel // nbw
                    bx_p = pep_sel % nbw
                    y0p = by_p * B
                    x0p = bx_p * B
                    for dy in range(B):
                        for dx in range(B):
                            salt_hw[y0s + dy, x0s + dx] = True
                            pep_hw[y0p + dy, x0p + dx] = True
                    for i in range(c):
                        d[i][salt_hw] = e
                        d[i][pep_hw] = -e
                out = (img_p + d).clamp(0.0, 1.0)
                return out[:, :h, :w]

            rand = torch.rand((nbh, nbw), device=img.device)
            salt_nb = (rand < (p / 2.0)).float().unsqueeze(0).unsqueeze(0)
            pep_nb = (((rand >= (p / 2.0)) & (rand < p))).float().unsqueeze(0).unsqueeze(0)
            salt_hw = (F.interpolate(salt_nb, size=(H, W), mode="nearest").squeeze(0).squeeze(0) > 0.5)
            pep_hw = (F.interpolate(pep_nb, size=(H, W), mode="nearest").squeeze(0).squeeze(0) > 0.5)
            d = torch.zeros_like(img_p)
            for i in range(c):
                d[i][salt_hw] = e
                d[i][pep_hw] = -e
            out = (img_p + d).clamp(0.0, 1.0)
            return out[:, :h, :w]

        if e > 0.0:
            if self.per_channel:
                rand = torch.rand((c, h, w), device=img.device)
            else:
                rand = torch.rand((h, w), device=img.device)
            salt = rand < (p / 2.0)
            pepper = (rand >= (p / 2.0)) & (rand < p)
            d = torch.zeros_like(img)
            d[salt] = e
            d[pepper] = -e
            return (img + d).clamp(0.0, 1.0)

        rand = torch.rand((h, w), device=img.device)
        salt = rand < (p / 2.0)
        pepper = (rand >= (p / 2.0)) & (rand < p)
        for i in range(c):
            noisy[i][salt] = 1.0
            noisy[i][pepper] = 0.0
        return noisy


class AddRainNoise:
    """
    雨丝：梯度加权主笔画 + 交叉角 + 第二遍随机短丝（提高全图遮挡）；条数×drop_scale×extra；
    L∞ 裁剪前强 alpha；再裁剪。cap 可与椒盐不同（通常更大）。
    """

    def __init__(
        self,
        drop_count: int = 1200,
        length: int = 18,
        thickness: int = 1,
        angle_deg: float = -15.0,
        intensity: float = 0.35,
        seed: Optional[int] = None,
        linf_boost: float = 1.0,
        grad_weighted_starts: bool = True,
        grad_start_power: float = 1.34,
        drop_scale_with_grad: float = 1.05,
        angle_jitter_deg: float = 16.0,
        length_jitter: float = 0.22,
        extra_drop_l2_scale: float = 1.0,
        crosshatch_frac: float = 0.72,
        crosshatch_deg: float = 78.0,
        length_boost: float = 1.2,
        dark_line_frac: float = 0.45,
        second_pass_frac: float = 0.58,
        second_length_scale: float = 0.52,
        second_angle_extra_deg: float = 32.0,
        linf_fill: float = 0.0,
        rain_max_primary_drops: int = 0,
    ):
        self.drop_count = int(drop_count)
        self.length = int(length)
        self.thickness = thickness
        self.angle_deg = float(angle_deg)
        self.intensity = float(intensity)
        self.seed = seed
        self.linf_boost = max(0.0, float(linf_boost))
        self.grad_weighted_starts = bool(grad_weighted_starts)
        self.grad_start_power = float(grad_start_power)
        self.drop_scale_with_grad = float(drop_scale_with_grad)
        self.angle_jitter_deg = float(angle_jitter_deg)
        self.length_jitter = float(length_jitter)
        self.extra_drop_l2_scale = max(0.0, float(extra_drop_l2_scale))
        self.crosshatch_frac = float(min(1.0, max(0.0, crosshatch_frac)))
        self.crosshatch_deg = float(crosshatch_deg)
        self.length_boost = max(0.5, float(length_boost))
        self.dark_line_frac = float(min(1.0, max(0.0, dark_line_frac)))
        self.second_pass_frac = float(min(1.0, max(0.0, second_pass_frac)))
        self.second_length_scale = max(0.15, float(second_length_scale))
        self.second_angle_extra_deg = float(second_angle_extra_deg)
        self.linf_fill = max(0.0, float(linf_fill))
        self.rain_max_primary_drops = max(0, int(rain_max_primary_drops))
        self._rng = np.random.default_rng(seed)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape
        img_pil = transforms.ToPILImage()(img.cpu())
        draw = ImageDraw.Draw(img_pil, mode="RGBA")
        rng = self._rng
        eff = min(1.0, float(self.intensity) * self.linf_boost)
        alpha = int(max(0, min(255, round(255.0 * eff))))
        color_light = (220, 220, 220, alpha)
        color_dim = (165, 170, 175, max(24, alpha - 35))

        n_drops = max(1, int(round(float(self.drop_count) * self.extra_drop_l2_scale)))
        flat_idx: Optional[np.ndarray] = None
        if self.grad_weighted_starts and h > 2 and w > 2:
            n_drops = max(1, int(round(float(self.drop_count) * self.drop_scale_with_grad * self.extra_drop_l2_scale)))
            with torch.no_grad():
                mag = _luminance_sobel_mag_hw(img).detach().float().cpu().numpy()
            mag = np.clip(mag, 1e-6, None) ** self.grad_start_power
            pr = mag.reshape(-1).astype(np.float64)
            pr /= pr.sum()
            flat_idx = rng.choice(h * w, size=n_drops, replace=True, p=pr)

        if self.rain_max_primary_drops > 0:
            n_drops = min(n_drops, self.rain_max_primary_drops)
        n_cross = int(round(float(n_drops) * self.crosshatch_frac)) if n_drops > 0 else 0
        n_dark = int(round(float(n_drops) * self.dark_line_frac)) if n_drops > 0 else 0

        for j in range(n_drops):
            if flat_idx is not None:
                t = int(flat_idx[j])
                x = t % w
                y = t // w
            else:
                x = int(rng.integers(0, max(1, w)))
                y = int(rng.integers(0, max(1, h)))
            base_ang = self.angle_deg + (self.crosshatch_deg if j < n_cross else 0.0)
            ang = base_ang + float(rng.uniform(-self.angle_jitter_deg, self.angle_jitter_deg))
            ln = (
                float(self.length)
                * self.length_boost
                * float(rng.uniform(max(0.35, 1.0 - self.length_jitter), 1.0 + self.length_jitter))
            )
            theta = np.deg2rad(ang)
            dx = int(round(ln * np.cos(theta)))
            dy = int(round(ln * np.sin(theta)))
            x2 = int(max(0, min(w - 1, x + dx)))
            y2 = int(max(0, min(h - 1, y + dy)))
            if j < n_dark:
                col = color_dim
            else:
                col = color_light
            draw.line([(x, y), (x2, y2)], fill=col, width=self.thickness)

        n2 = int(round(float(n_drops) * self.second_pass_frac))
        for j in range(n2):
            x = int(rng.integers(0, max(1, w)))
            y = int(rng.integers(0, max(1, h)))
            ang = (
                self.angle_deg
                + float(rng.uniform(-self.second_angle_extra_deg, self.second_angle_extra_deg))
                + (self.crosshatch_deg * 0.5 if rng.random() < 0.35 else 0.0)
            )
            ln = (
                float(self.length)
                * self.length_boost
                * self.second_length_scale
                * float(rng.uniform(0.55, 1.05))
            )
            theta = np.deg2rad(ang)
            dx = int(round(ln * np.cos(theta)))
            dy = int(round(ln * np.sin(theta)))
            x2 = int(max(0, min(w - 1, x + dx)))
            y2 = int(max(0, min(h - 1, y + dy)))
            col = color_dim if rng.random() < 0.42 else color_light
            wline = max(1, int(self.thickness) - 1)
            draw.line([(x, y), (x2, y2)], fill=col, width=wline)
        noisy = transforms.ToTensor()(img_pil).to(dtype=torch.float32)
        img_f = img.detach().float()
        if noisy.device != img_f.device:
            noisy = noisy.to(img_f.device)
        if self.linf_fill > 0.0:
            d = noisy - img_f
            mx = float(d.abs().max().item())
            e = float(self.linf_fill)
            if mx > 1e-8 and mx < e:
                d = d * (e / mx)
                d = d.clamp(-e, e)
                noisy = (img_f + d).clamp(0.0, 1.0)
        return noisy.to(dtype=img.dtype).clamp(0.0, 1.0)


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
    linf_cap: float = 0.05,
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
        img_noisy = apply_linf_noise_cap(img_clean, img_noisy, float(linf_cap))
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
    linf_base: float = 0.05,
    linf_cap: float = 0.05,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"噪声类型：{noise_name}（YOLO）")
    print(f"{'#' * 60}")
    if float(linf_cap) > 0.0:
        if abs(float(linf_cap) - float(linf_base)) > 1e-8:
            print(
                f"  L∞ 裁剪 |x'-x|_∞≤{float(linf_cap):.4f}（基线 linf_eps={float(linf_base):.4f}；高斯/椒盐原生；椒盐块+显著性；雨丝后裁剪）",
                flush=True,
            )
        else:
            print(
                f"  L∞ |x'-x|_∞≤{float(linf_cap):.4f}（高斯/椒盐原生；椒盐块+显著性；雨丝后裁剪）",
                flush=True,
            )
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
            linf_cap=float(linf_cap),
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
        f.write(f"runs={num_runs}, num_eval={num_eval}, conf={conf_thresh}, iou={iou_thresh}, linf_base={linf_base}, linf_cap={linf_cap}\n")
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
        help="高斯噪声：无 L∞ 时为加性标准差；有 --linf_eps 时为 N(0,σ²) 再截断到 ±linf_eps 前的 σ（σ 越大越易贴满 ±ε）",
    )
    parser.add_argument(
        "--salt_pepper_p",
        type=float,
        default=0.05,
        help="椒盐比例：无 L∞ 时为 0/1 翻转；有 --linf_eps 时为 ±linf_eps（逐像素或块，见 --salt_block_size）",
    )
    parser.add_argument(
        "--salt_block_size",
        type=int,
        default=1,
        help="L∞ 椒盐：>1 时按 B×B 整块 ±ε（对 CNN 比孤立像素强），建议 8~16 配合 --salt_pepper_p",
    )
    parser.add_argument(
        "--salt_block_lowfreq",
        type=int,
        default=4,
        help="椒盐块显著性权重上：>1 时在粗网格随机场上采样再乘权重，使块成簇（类高斯 lowfreq）；1 关闭",
    )
    parser.add_argument(
        "--salt_block_gain",
        type=float,
        default=1.07,
        help="椒盐显著性模式下块数 k 再乘该系数（略增覆盖以提高 ASR）",
    )
    parser.add_argument("--rain_drop_count", type=int, default=1200, help="雨丝条数（主 pass；另有第二遍短丝）")
    parser.add_argument("--rain_intensity", type=float, default=0.35, help="雨丝 alpha 强度 0~1")
    parser.add_argument(
        "--rain_length",
        type=int,
        default=26,
        help="雨丝线段像素长度（主 pass），略大更易跨物体",
    )
    parser.add_argument(
        "--rain_drop_scale",
        type=float,
        default=1.06,
        help="雨丝梯度模式下相对 --rain_drop_count 的抽样条数比例",
    )
    parser.add_argument(
        "--rain_crosshatch_frac",
        type=float,
        default=0.72,
        help="主 pass 前若干条加 cross 大角度",
    )
    parser.add_argument(
        "--rain_dark_frac",
        type=float,
        default=0.48,
        help="主 pass 前若干条用略暗 RGB",
    )
    parser.add_argument(
        "--gaussian_lowfreq",
        type=int,
        default=1,
        help="L∞ 高斯：>1 时先在低分辨率生成噪声再上采样，结构性更强（试 8~12 提高 ASR）",
    )
    parser.add_argument(
        "--rain_linf_boost",
        type=float,
        default=1.0,
        help="雨丝：等效强度=min(1, intensity×boost)，L∞ 裁剪前画更亮（试 2~3）",
    )
    parser.add_argument(
        "--rain_thickness",
        type=int,
        default=1,
        help="雨丝线宽（像素），L∞ 下建议 4~8 提高被裁剪后仍覆盖的笔画面积",
    )
    parser.add_argument(
        "--salt_cap_max",
        type=float,
        default=0.09,
        help="椒盐 L∞ 裁剪绝对上限（与 linf_eps×salt_linf_mult 取 min）",
    )
    parser.add_argument(
        "--rain_cap_max",
        type=float,
        default=0.12,
        help="雨丝 L∞ 目标/裁剪上限（与 linf_eps×rain_linf_mult 取 min）；PIL 半透明时实际 L∞ 常远小于 cap，见 --rain_linf_fill",
    )
    parser.add_argument(
        "--linf_eps",
        type=float,
        default=0.05,
        help="基线 L∞ 上限（高斯截断/报告对齐）；椒盐原生 ± 与雨丝后裁剪可单独高于该值，见 --salt_linf_mult / --rain_linf_mult。≤0 关闭",
    )
    parser.add_argument(
        "--salt_linf_mult",
        type=float,
        default=1.32,
        help="仅椒盐：L∞ 裁剪=min(--salt_cap_max, linf_eps×该倍数)；与 salt_l2_exp/salt_p_gain 联调",
    )
    parser.add_argument(
        "--rain_linf_mult",
        type=float,
        default=1.92,
        help="仅雨丝：rain_cap=min(--rain_cap_max, linf_eps×该倍数)。若日志里 LinfCap≈0.055 说明未同步本脚本（旧版 mult≈1.1）",
    )
    parser.add_argument(
        "--salt_l2_exp",
        type=float,
        default=2.32,
        help="椒盐 p_eff 中 (linf_eps/salt_cap) 的指数；>2 相对平方更压 per-pixel L2，常配合 --salt_p_gain",
    )
    parser.add_argument(
        "--salt_p_gain",
        type=float,
        default=1.26,
        help="椒盐在 L2 缩放后再乘该增益以提高 ASR（随机噪声不保证固定百分比）",
    )
    parser.add_argument(
        "--rain_l2_exp",
        type=float,
        default=2.02,
        help="雨丝条数缩放中 (linf_eps/rain_cap) 的指数；略接近 2 在 rain_cap 变大时保留更多条数",
    )
    parser.add_argument(
        "--rain_drop_gain",
        type=float,
        default=1.55,
        help="雨丝条数再乘该增益（配合更高 L∞ cap）",
    )
    parser.add_argument(
        "--rain_second_pass",
        type=float,
        default=0.45,
        help="雨丝第二遍短丝条数 = 主 pass 条数×该比例（全图均匀）；略降以减耗时",
    )
    parser.add_argument(
        "--rain_max_primary_drops",
        type=int,
        default=7500,
        help="主 pass 条数上限（0 不限制）；count 很大时可明显加速",
    )
    parser.add_argument(
        "--rain_linf_fill",
        type=float,
        default=-1.0,
        help="雨丝画完后将 |x'-x|_∞ 拉至该值（≤0 则用雨丝 LinfCap）。解决 PIL 半透明导致报告 Linf 远小于 cap",
    )
    parser.add_argument(
        "--noise_preview_n",
        type=int,
        default=5,
        help="每种噪声保存前 N 张加噪图 PNG 到 outdir/noise_previews/...；0 关闭",
    )
    args = parser.parse_args()
    print(f"噪声实验脚本: {os.path.abspath(__file__)}", flush=True)
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
    salt_mult = max(1.0, float(args.salt_linf_mult))
    rain_mult = max(1.0, float(args.rain_linf_mult))
    sp = float(args.salt_pepper_p)
    if le > 0:
        salt_cap_max = max(le, min(0.12, float(args.salt_cap_max)))
        rain_cap_max = max(le, min(0.15, float(args.rain_cap_max)))
        salt_le = min(salt_cap_max, le * salt_mult)
        rain_le = min(rain_cap_max, le * rain_mult)
        salt_l2_exp = max(1.5, min(3.5, float(args.salt_l2_exp)))
        rain_l2_exp = max(1.5, min(3.5, float(args.rain_l2_exp)))
        p_salt = min(1.0, sp * (le / max(salt_le, 1e-12)) ** salt_l2_exp * float(args.salt_p_gain))
        rain_drop_l2 = (le / max(rain_le, 1e-12)) ** rain_l2_exp * float(args.rain_drop_gain)
    else:
        salt_le = 0.0
        rain_le = 0.0
        p_salt = sp
        rain_drop_l2 = 1.0

    rain_fill = 0.0
    if le > 0:
        rain_fill = float(rain_le)
        if float(args.rain_linf_fill) > 0.0:
            rain_fill = min(float(rain_le), float(args.rain_linf_fill))

    if le > 0:
        print(
            f"L∞ 基线 linf_eps={le}；椒盐 cap={salt_le:.4f}（×{salt_mult:.3f}），p_eff≈{p_salt:.4f}（exp={salt_l2_exp:.2f}×gain={float(args.salt_p_gain):.2f}）；"
            f"雨丝 cap={rain_le:.4f}（×{rain_mult:.3f}），LinfFill={rain_fill:.4f}，条数×{float(args.rain_drop_scale) * rain_drop_l2:.3f}（exp={rain_l2_exp:.2f}×gain={float(args.rain_drop_gain):.2f}）；"
            f"高斯仍用基线 ±{le}。",
            flush=True,
        )
    else:
        print("L∞ 扰动上限：关闭（--linf_eps≤0）", flush=True)

    ds = load_voc2007_dataset(args.eval_set)
    print(f"VOC2007 {args.eval_set} 样本数: {len(ds)}")
    print(f"加载 YOLO 模型: {args.load_model}")
    model = build_yolo_voc_model(device, weights=args.load_model)

    gs = float(args.gaussian_sigma)
    rd = int(args.rain_drop_count)
    ri = float(args.rain_intensity)
    lf = max(1, int(args.gaussian_lowfreq))
    rb = float(args.rain_linf_boost)
    sb = max(1, int(args.salt_block_size))
    sblf = max(1, int(args.salt_block_lowfreq))
    rds = float(args.rain_drop_scale)
    rcf = float(min(1.0, max(0.0, args.rain_crosshatch_frac)))
    rt = max(1, int(args.rain_thickness))
    sbg = float(min(1.35, max(0.85, args.salt_block_gain)))
    rdf = float(min(1.0, max(0.0, args.rain_dark_frac)))
    rmn = max(0, int(args.rain_max_primary_drops))
    rln = max(8, int(args.rain_length))
    rsp = float(min(1.0, max(0.0, args.rain_second_pass)))
    noise_configs = [
        (f"高斯噪声(sigma={gs},lowfreq={lf})", AddGaussianNoise(sigma=gs, linf_eps=le, lowfreq_downscale=lf), le),
        (
            f"椒盐噪声(p_cmd={sp},p_eff={p_salt:.4f},block={sb},lfBlk={sblf},blkGain={sbg:.2f},chSign,LinfCap={salt_le:.4f},salience)",
            AddSaltPepperNoise(
                p=p_salt,
                linf_eps=salt_le,
                block_size=sb,
                salience_weighted_blocks=True,
                block_mask_lowfreq=sblf,
                block_count_gain=sbg,
            ),
            salt_le if le > 0 else 0.0,
        ),
        (
            f"雨丝噪声(count={rd},len={rln},intensity={ri},rainBoost={rb},thick={rt},LinfCap={rain_le:.4f},LinfFill={rain_fill:.4f},maxPrim={rmn},gradDrops×{rds * rain_drop_l2:.3f},cross={rcf:.2f},2nd={rsp:.2f},dark={rdf:.2f})",
            AddRainNoise(
                drop_count=rd,
                length=rln,
                thickness=rt,
                intensity=ri,
                linf_boost=rb,
                grad_weighted_starts=True,
                drop_scale_with_grad=rds,
                extra_drop_l2_scale=rain_drop_l2,
                crosshatch_frac=rcf,
                dark_line_frac=rdf,
                second_pass_frac=rsp,
                linf_fill=rain_fill,
                rain_max_primary_drops=rmn,
            ),
            rain_le if le > 0 else 0.0,
        ),
    ]

    for noise_name, noise_fn, linf_cap in noise_configs:
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
            linf_base=le,
            linf_cap=float(linf_cap),
        )

    print("\n所有 YOLO 噪声实验完成。")


if __name__ == "__main__":
    main()
