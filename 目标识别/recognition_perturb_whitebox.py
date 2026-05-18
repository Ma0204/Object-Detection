# -*- coding: utf-8 -*-
"""在 L∞ 约束下扰动「场景图」，使内部可微目标下降（不在这里讨论框，只操作像素）。"""
from __future__ import annotations

from typing import Literal

import torch

from recognition_backend import ensure_project_root_on_path

Variant = Literal["fgsm", "pgd", "adam"]


def _whitebox_step_loop(
    model: torch.nn.Module,
    scene01: torch.Tensor,
    *,
    eps: float,
    steps: int,
    alpha: float,
    random_start: bool,
    target_conf: float,
    min_steps: int,
    stop_loss: float,
    di_prob: float,
    di_scale_min: float,
    ti_kernel: int,
    ti_sigma: float,
    topk_surrogate: int,
) -> torch.Tensor:
    ensure_project_root_on_path()
    from attack_utils import (
        ensure_tensor_01,
        input_diversity,
        ti_smooth_grad,
        yolo_whitebox_objective,
    )

    x0 = scene01.detach()
    x = ensure_tensor_01(x0 + torch.empty_like(x0).uniform_(-eps, eps)) if random_start else x0.clone()
    for si in range(int(steps)):
        x = x.detach().clone().requires_grad_(True)
        x_in = input_diversity(x, prob=float(di_prob), scale_min=float(di_scale_min))
        loss = yolo_whitebox_objective(model, x_in, topk=int(topk_surrogate), target_conf=float(target_conf))
        loss.backward()
        grad = ti_smooth_grad(x.grad.detach(), kernel_size=int(ti_kernel), sigma=float(ti_sigma))
        with torch.no_grad():
            x = x - float(alpha) * grad.sign()
            x = torch.max(torch.min(x, x0 + float(eps)), x0 - float(eps))
            x = ensure_tensor_01(x)
        if float(stop_loss) >= 0.0 and (si + 1) >= int(min_steps) and float(loss.detach().item()) <= float(stop_loss):
            break
    return x.detach()


def _adam_style_loop(
    model: torch.nn.Module,
    scene01: torch.Tensor,
    *,
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
    topk_surrogate: int,
) -> torch.Tensor:
    ensure_project_root_on_path()
    from attack_utils import (
        ensure_tensor_01,
        input_diversity,
        ti_smooth_grad,
        yolo_whitebox_objective,
    )

    x0 = scene01.detach()
    best_x = x0.clone()
    best_score = float("inf")
    b1 = min(0.999, max(0.5, float(beta1)))
    for _ in range(max(1, int(restarts))):
        x = (
            ensure_tensor_01(x0 + torch.empty_like(x0).uniform_(-eps, eps))
            if random_start
            else x0.clone()
        )
        m = torch.zeros_like(x0)
        for si in range(int(steps)):
            x = x.detach().clone().requires_grad_(True)
            x_in = input_diversity(x, prob=float(di_prob), scale_min=float(di_scale_min))
            loss = yolo_whitebox_objective(
                model, x_in, topk=int(topk_surrogate), target_conf=float(target_conf)
            )
            loss.backward()
            grad = ti_smooth_grad(x.grad.detach(), kernel_size=int(ti_kernel), sigma=float(ti_sigma))
            m = b1 * m + (1.0 - b1) * grad
            t = float(si + 1)
            m_hat = m / (1.0 - b1**t)
            with torch.no_grad():
                x = x - float(alpha) * m_hat.sign()
                x = torch.max(torch.min(x, x0 + float(eps)), x0 - float(eps))
                x = ensure_tensor_01(x)
            if float(stop_loss) >= 0.0 and (si + 1) >= int(min_steps) and float(loss.detach().item()) <= float(
                stop_loss
            ):
                break
        with torch.no_grad():
            score = float(
                yolo_whitebox_objective(
                    model, x.detach(), topk=int(topk_surrogate), target_conf=float(target_conf)
                ).item()
            )
        if score < best_score:
            best_score = score
            best_x = x.detach().clone()
    return best_x


def perturb_scene_whitebox(
    model: torch.nn.Module,
    scene01: torch.Tensor,
    variant: Variant,
    *,
    eps: float,
    steps: int,
    alpha: float,
    random_start: bool,
    target_conf: float,
    min_steps: int,
    stop_loss: float,
    di_prob: float,
    di_scale_min: float,
    ti_kernel: int,
    ti_sigma: float,
    restarts: int = 1,
    topk_surrogate: int = 300,
    beta1: float = 0.9,
) -> torch.Tensor:
    """根据 variant 选择更新形式；fgsm 与 pgd 在本实现中为同一符号梯度多步循环。"""
    if variant in ("fgsm", "pgd"):
        return _whitebox_step_loop(
            model,
            scene01,
            eps=eps,
            steps=steps,
            alpha=alpha,
            random_start=random_start,
            target_conf=target_conf,
            min_steps=min_steps,
            stop_loss=stop_loss,
            di_prob=di_prob,
            di_scale_min=di_scale_min,
            ti_kernel=ti_kernel,
            ti_sigma=ti_sigma,
            topk_surrogate=topk_surrogate,
        )
    if variant == "adam":
        return _adam_style_loop(
            model,
            scene01,
            eps=eps,
            steps=steps,
            alpha=alpha,
            beta1=beta1,
            random_start=random_start,
            target_conf=target_conf,
            min_steps=min_steps,
            stop_loss=stop_loss,
            di_prob=di_prob,
            di_scale_min=di_scale_min,
            ti_kernel=ti_kernel,
            ti_sigma=ti_sigma,
            restarts=restarts,
            topk_surrogate=topk_surrogate,
        )
    raise ValueError(f"unknown variant: {variant}")
