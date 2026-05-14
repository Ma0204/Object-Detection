# -*- coding: utf-8 -*-
"""
完整的评估指标计算模块

包含：
- 模型性能指标：精确率、召回率、F-score、准确率
- 攻击/干扰指标：平均扰动距离、平均失真度、SSIM、PSNR
"""

import torch
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetrics:
    """模型性能指标"""
    precision: float  # 精确率
    recall: float     # 召回率
    f_score: float    # F-score
    accuracy: float   # 准确率


@dataclass(frozen=True)
class PerturbationMetrics:
    """扰动/攻击指标"""
    attack_success_rate: float  # 攻击成功率 (%)
    avg_l2_distance: float      # 平均 L2 扰动距离
    avg_linf_distance: float    # 平均 L∞ 扰动距离
    avg_mse: float              # 平均失真度 (MSE)
    avg_ssim: float             # 平均结构相似度 (SSIM)
    avg_psnr: float             # 平均峰值信噪比 (PSNR)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    计算两张图像的结构相似度 (SSIM)
    img1, img2: [C, H, W] 张量，值域 [0, 1]
    """
    if img1.shape != img2.shape:
        return 0.0
    
    img1 = img1.float()
    img2 = img2.float()
    
    # 计算均值
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    # 计算方差和协方差
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = (img1 ** 2).mean() - mu1_sq
    sigma2_sq = (img2 ** 2).mean() - mu2_sq
    sigma12 = (img1 * img2).mean() - mu1_mu2
    
    # SSIM 公式
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(ssim.clamp(min=-1, max=1))


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    计算峰值信噪比 (PSNR)
    img1, img2: [C, H, W] 张量，值域 [0, 1]
    """
    if img1.shape != img2.shape:
        return 0.0
    
    img1 = img1.float()
    img2 = img2.float()
    
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return 100.0  # 完全相同
    
    psnr = 20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse))
    return float(psnr)


def compute_model_metrics(
    tp: int,  # True Positive
    fp: int,  # False Positive
    fn: int,  # False Negative
    tn: int,  # True Negative
) -> ModelMetrics:
    """
    计算模型性能指标
    
    tp: 真正例（正确检测到的目标）
    fp: 假正例（错误检测的目标）
    fn: 假负例（漏检的目标）
    tn: 真负例（正确未检测的背景）
    """
    # 精确率 = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 召回率 = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F-score = 2 * (precision * recall) / (precision + recall)
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 准确率 = (TP + TN) / (TP + FP + FN + TN)
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    return ModelMetrics(
        precision=precision,
        recall=recall,
        f_score=f_score,
        accuracy=accuracy,
    )


def compute_perturbation_metrics(
    clean_imgs: list,      # 原始图像列表 [C,H,W]
    perturbed_imgs: list,  # 扰动后图像列表 [C,H,W]
    attack_success_rate: float,  # 攻击成功率 (0-1)
) -> PerturbationMetrics:
    """
    计算扰动/攻击指标
    """
    if len(clean_imgs) == 0:
        return PerturbationMetrics(
            attack_success_rate=0.0,
            avg_l2_distance=0.0,
            avg_linf_distance=0.0,
            avg_mse=0.0,
            avg_ssim=0.0,
            avg_psnr=0.0,
        )
    
    l2_distances = []
    linf_distances = []
    mses = []
    ssims = []
    psnrs = []
    
    for clean_img, pert_img in zip(clean_imgs, perturbed_imgs):
        clean_img = clean_img.float()
        pert_img = pert_img.float()
        
        # L2 距离
        l2_dist = torch.norm(clean_img - pert_img, p=2).item()
        l2_distances.append(l2_dist)
        
        # L∞ 距离
        linf_dist = torch.norm(clean_img - pert_img, p=float('inf')).item()
        linf_distances.append(linf_dist)
        
        # MSE (失真度)
        mse = ((clean_img - pert_img) ** 2).mean().item()
        mses.append(mse)
        
        # SSIM
        ssim = compute_ssim(clean_img, pert_img, data_range=1.0)
        ssims.append(ssim)
        
        # PSNR
        psnr = compute_psnr(clean_img, pert_img, data_range=1.0)
        psnrs.append(psnr)
    
    return PerturbationMetrics(
        attack_success_rate=attack_success_rate * 100,  # 转换为百分比
        avg_l2_distance=float(np.mean(l2_distances)),
        avg_linf_distance=float(np.mean(linf_distances)),
        avg_mse=float(np.mean(mses)),
        avg_ssim=float(np.mean(ssims)),
        avg_psnr=float(np.mean(psnrs)),
    )


def print_model_metrics(metrics: ModelMetrics, name: str = ""):
    """打印模型性能指标"""
    print(f"\n{'='*50}")
    print(f"模型性能指标 {name}")
    print(f"{'='*50}")
    print(f"精确率 (Precision): {metrics.precision*100:.2f}%")
    print(f"召回率 (Recall):   {metrics.recall*100:.2f}%")
    print(f"F-score:          {metrics.f_score:.4f}")
    print(f"准确率 (Accuracy): {metrics.accuracy*100:.2f}%")


def print_perturbation_metrics(metrics: PerturbationMetrics, name: str = ""):
    """打印扰动/攻击指标"""
    print(f"\n{'='*50}")
    print(f"扰动/攻击指标 {name}")
    print(f"{'='*50}")
    print(f"攻击成功率:        {metrics.attack_success_rate:.2f}%")
    print(f"平均 L2 距离:      {metrics.avg_l2_distance:.6f}")
    print(f"平均 L∞ 距离:      {metrics.avg_linf_distance:.6f}")
    print(f"平均失真度 (MSE):  {metrics.avg_mse:.6f}")
    print(f"平均 SSIM:         {metrics.avg_ssim:.4f}")
    print(f"平均 PSNR:         {metrics.avg_psnr:.2f} dB")
