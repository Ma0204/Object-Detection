# -*- coding: utf-8 -*-  # UTF-8 编码，支持中文注释/路径
# ========== 文件说明 ==========  # 分区注释
# 白盒攻击：FGSM（VOC 目标检测）  # 脚本主题
# 注意：这里实现的是“迭代增强版 FGSM”（I-FGSM 风格），每轮都会重新计算梯度  # 重要说明
# 约束：eps<=0.05，并且像素裁剪到 [0,1]  # 项目约束
# 输出：P/R/F/Acc + ASR(GT级/图像级) + L2/Linf/MSE/SSIM/PSNR  # 输出指标

from __future__ import annotations  # 延迟解析注解，兼容 | 等语法

import argparse  # 命令行参数解析
import os  # 路径与目录
import random  # 随机抽样
import time  # 计时与进度显示
from typing import Dict  # 类型提示

import numpy as np  # 统计（均值等）
import torch  # 张量与自动求导

from attack_utils import (  # 项目工具
    build_yolo_voc_model,  # 构建 YOLO VOC 检测模型
    ensure_tensor_01,  # 裁剪到 [0,1]
    infer_yolo,  # YOLO 推理接口
    load_voc2007_dataset,  # 加载 VOC 数据集
    voc_target_to_boxes_and_labels,  # 解析 VOC 标注
    compute_iou,  # IoU
    filter_pred,  # 过滤预测
    yolo_whitebox_objective,  # YOLO 白盒目标函数
    input_diversity,  # DI
    ti_smooth_grad,  # TI
)  # 工具导入结束
from evaluation_metrics import compute_model_metrics, compute_perturbation_metrics  # 指标计算

VOC_CLASSES = [  # VOC 类别列表（含 background）
    "background",  # 背景
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",  # 1-9
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",  # 10-17
    "sofa", "train", "tvmonitor",  # 18-20
]  # 列表结束
VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}  # 名称到 id 映射


def build_voc_model(device: torch.device, weights: str) -> torch.nn.Module:  # 构建 YOLO(VOC映射) 模型
    return build_yolo_voc_model(device, weights=weights)  # 返回 YOLO 模型


@torch.no_grad()  # 推理阶段不建图，节省显存
def infer(model: torch.nn.Module, img: torch.Tensor) -> Dict[str, torch.Tensor]:  # 单张图推理，返回 boxes/scores/labels
    return infer_yolo(model, img)  # 统一走 YOLO 适配层


def fgsm_attack(  # 迭代 FGSM（I-FGSM）：用伪标注 loss 的输入梯度生成对抗样本
    model: torch.nn.Module,  # 检测模型（Faster R-CNN）
    img_01: torch.Tensor,  # 输入图像（已归一化到 [0,1]）
    eps: float,  # 最大 L∞ 扰动预算（每像素不超过 eps）
    steps: int,  # 迭代轮数（越大越强但更慢）
    alpha: float,  # 每步更新步长
    random_start: bool,  # 是否从 eps 邻域随机起点开始
    target_conf: float,  # 目标压制阈值（越小攻击越激进）
    min_steps: int,  # 允许早停前最少步数
    stop_loss: float,  # 早停阈值
    di_prob: float,  # 输入多样化触发概率
    di_scale_min: float,  # 输入多样化最小缩放
    ti_kernel: int,  # 梯度平滑核大小
    ti_sigma: float,  # 梯度平滑sigma
) -> torch.Tensor:  # 返回对抗图（仍在 [0,1] 且满足 L∞ 约束）
    # 增强版 FGSM（迭代符号梯度）：每一步都用梯度符号更新，再投影回 eps 约束  # 核心思路
    x0 = img_01.detach()  # 保存原图（不参与梯度）
    x = ensure_tensor_01(x0 + torch.empty_like(x0).uniform_(-eps, eps)) if random_start else x0.clone()  # 可选随机起点
    for si in range(int(steps)):  # 迭代 steps 轮（支持提前早停）
        x = x.detach().clone().requires_grad_(True)  # 让当前 x 可求导以计算 d(loss)/d(x)
        x_in = input_diversity(x, prob=float(di_prob), scale_min=float(di_scale_min))
        loss = yolo_whitebox_objective(model, x_in, topk=300, target_conf=float(target_conf))  # 以 YOLO 目标强度为可微目标
        loss.backward()  # 反向传播，得到 x.grad（输入梯度）
        grad = ti_smooth_grad(x.grad.detach(), kernel_size=int(ti_kernel), sigma=float(ti_sigma))
        with torch.no_grad():  # 更新对抗样本不需要梯度图
            x = x - alpha * grad.sign()  # 沿梯度下降方向更新（使检测目标减小，提升攻击性）
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)  # 投影到 L∞ 球（保证每像素扰动不超 eps）
            x = ensure_tensor_01(x)  # 像素裁剪到 [0,1]
        # 自适应早停：达到目标强度后提前结束，缩短平均耗时
        if float(stop_loss) >= 0.0 and (si + 1) >= int(min_steps) and float(loss.detach().item()) <= float(stop_loss):
            break
    return x.detach()  # 返回对抗样本（脱离计算图）


def evaluate_once(model, dataset, args, run_id: int):  # 单次运行：随机抽样 num_eval 张并统计指标
    n = min(int(args.num_eval), len(dataset))  # 本轮评估张数（不超过数据集大小）
    idxs = random.sample(range(len(dataset)), n)  # 无放回随机抽样索引
    t0 = time.time()  # 本轮起始时间

    c_tp = c_fp = c_fn = 0  # clean 侧 TP/FP/FN
    a_tp = a_fp = a_fn = 0  # adv 侧 TP/FP/FN
    gt_detected_clean = gt_success_count = 0  # GT 级 ASR 计数（分母/分子）
    img_detected_clean = img_success_count = 0  # 图像级 ASR 计数（分母/分子）
    clean_imgs, adv_imgs = [], []  # 记录部分 clean/adv 图像对用于计算扰动指标

    def count_metrics(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh):  # 单图统计 TP/FP/FN（类别一致 + IoU>=阈值）
        tp_here = fn_here = matched_here = 0  # 本图局部 TP/FN/匹配计数
        for g_box, g_name in zip(gt_boxes, gt_labels):  # 遍历每个 GT
            if g_name not in VOC_CLASS_TO_ID:  # 过滤异常类别名
                continue  # 跳过该 GT
            gt_lid = VOC_CLASS_TO_ID[g_name]  # GT 类别 id
            g_box_list = g_box.tolist()  # GT 框转 list 供 IoU 计算
            found = False  # 是否找到匹配预测
            for p_box, p_lid in zip(pred_boxes, pred_labels):  # 遍历预测框
                if int(p_lid) == int(gt_lid) and compute_iou(g_box_list, p_box.tolist()) >= iou_thresh:  # 同类且 IoU 达标
                    found = True  # 标记命中
                    break  # 一个 GT 只计一次
            if found:  # 命中则 TP
                tp_here += 1  # TP+1
                matched_here += 1  # 记录匹配数（用于估计 FP）
            else:  # 未命中则 FN
                fn_here += 1  # FN+1
        fp_here = max(0, len(pred_labels) - matched_here)  # 简化：未匹配到 GT 的预测计为 FP
        return tp_here, fp_here, fn_here, matched_here  # 返回统计量

    for step, idx in enumerate(idxs):  # 逐张图评估
        img_tensor, target = dataset[idx]  # 读取图像与标注
        img = img_tensor.to(args.device)  # 图像放到设备
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)  # 解析 GT 框与类别名

        pred_clean = infer(model, img)  # 干净图推理
        fp_clean = filter_pred(pred_clean, conf_thresh=args.conf)  # 按置信度过滤 clean 预测
        boxes_c = fp_clean["boxes"].cpu()  # clean 预测框（CPU）
        labels_c = fp_clean["labels"].cpu().tolist()  # clean 预测类别列表（CPU）

        adv = fgsm_attack(
            model,
            img,
            eps=args.eps,
            steps=args.steps,
            alpha=args.alpha,
            random_start=args.random_start,
            target_conf=args.target_conf,
            min_steps=args.min_steps,
            stop_loss=args.stop_loss,
            di_prob=args.di_prob,
            di_scale_min=args.di_scale_min,
            ti_kernel=args.ti_kernel,
            ti_sigma=args.ti_sigma,
        )  # 生成对抗图

        pred_adv = infer(model, adv)  # 对抗图推理
        fp_adv = filter_pred(pred_adv, conf_thresh=args.conf)  # 按置信度过滤 adv 预测
        boxes_a = fp_adv["boxes"].cpu()  # adv 预测框（CPU）
        labels_a = fp_adv["labels"].cpu().tolist()  # adv 预测类别列表（CPU）

        ctp, cfp, cfn, cm = count_metrics(gt_boxes, gt_labels, boxes_c, labels_c, args.iou)  # clean TP/FP/FN + 匹配数
        atp, afp, afn, am = count_metrics(gt_boxes, gt_labels, boxes_a, labels_a, args.iou)  # adv TP/FP/FN + 匹配数

        c_tp += ctp; c_fp += cfp; c_fn += cfn  # 累加 clean 统计量
        a_tp += atp; a_fp += afp; a_fn += afn  # 累加 adv 统计量

        for g_box, g_name in zip(gt_boxes, gt_labels):  # 统计 GT 级攻击成功
            if g_name not in VOC_CLASS_TO_ID:  # 跳过异常类别
                continue  # 下一个 GT
            gt_lid = VOC_CLASS_TO_ID[g_name]  # GT 类别 id
            g_box_list = g_box.tolist()  # GT 框
            c_hit = any(int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= args.iou for pb, pl in zip(boxes_c, labels_c))  # clean 是否命中
            if c_hit:  # 只统计 clean 能命中的 GT
                gt_detected_clean += 1  # GT 级分母 +1
                a_hit = any(int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= args.iou for pb, pl in zip(boxes_a, labels_a))  # adv 是否仍命中
                if not a_hit:  # adv 漏检则攻击成功
                    gt_success_count += 1  # GT 级分子 +1

        if cm > 0 and gt_boxes.size(0) > 0:  # clean 图像至少命中 1 个 GT
            img_detected_clean += 1  # 图像级分母 +1
            if am == 0:  # adv 图像 0 命中（全漏检）
                img_success_count += 1  # 图像级攻击成功 +1

        if len(clean_imgs) < 200:  # 控制用于扰动指标的样本对数量
            clean_imgs.append(img.detach().cpu())  # 保存 clean 图像
            adv_imgs.append(adv.detach().cpu())  # 保存 adv 图像

        if (step + 1) % max(1, int(args.log_every)) == 0 or (step + 1) == n:  # 按 log_every 打印实时进度
            done = step + 1
            elapsed = time.time() - t0
            speed = done / max(elapsed, 1e-6)
            eta = (n - done) / max(speed, 1e-6)
            cur_gt_asr = (gt_success_count / max(1, gt_detected_clean)) * 100.0
            print(
                f"  [FGSM 第{run_id}次] 进度 {done}/{n} ({done/max(1,n)*100:.1f}%) | "
                f"当前GT-ASR={cur_gt_asr:.2f}% | 用时={elapsed/60:.1f}m | ETA={eta/60:.1f}m"
            )

    clean_metrics = compute_model_metrics(c_tp, c_fp, c_fn, 0)  # 汇总 clean 有效性指标
    adv_metrics = compute_model_metrics(a_tp, a_fp, a_fn, 0)  # 汇总 adv 有效性指标
    gt_asr = gt_success_count / max(1, gt_detected_clean)  # GT 级攻击成功率
    img_asr = img_success_count / max(1, img_detected_clean)  # 图像级攻击成功率
    pert = compute_perturbation_metrics(clean_imgs, adv_imgs, gt_asr)  # 扰动距离/失真/感知指标

    return {  # 返回本轮统计结果
        "clean": clean_metrics,  # clean 指标对象
        "adv": adv_metrics,  # adv 指标对象
        "gt_asr": gt_asr,  # GT 级 ASR
        "img_asr": img_asr,  # 图像级 ASR
        "gt_succ": gt_success_count,  # GT 级成功次数
        "gt_base": gt_detected_clean,  # GT 级基数
        "img_succ": img_success_count,  # 图像级成功次数
        "img_base": img_detected_clean,  # 图像级基数
        "pert": pert,  # 扰动指标对象
    }  # 字典结束


def main():  # 脚本入口：解析参数、加载模型/数据集、重复 runs 次评估并汇总
    p = argparse.ArgumentParser()  # 参数解析器
    p.add_argument("--eps", type=float, default=0.05)  # 扰动上限 eps
    p.add_argument("--steps", type=int, default=80)  # 迭代步数（默认提速）
    p.add_argument("--alpha", type=float, default=0.0012)  # 每步步长（与步数配套）
    p.add_argument(
        "--target_conf",
        type=float,
        default=None,
        help="白盒目标中 ReLU(conf-·) 的阈值；默认与 --conf 一致以便与评估对齐",
    )
    p.add_argument("--min_steps", type=int, default=35, help="允许早停前至少执行的步数")
    p.add_argument(
        "--stop_loss",
        type=float,
        default=-1.0,
        help="早停阈值（loss<=该值则停）；<0 关闭早停，利于冲高 ASR",
    )
    p.add_argument("--di_prob", type=float, default=0.5, help="输入多样化触发概率")
    p.add_argument("--di_scale_min", type=float, default=0.92, help="输入多样化最小缩放比例")
    p.add_argument("--ti_kernel", type=int, default=5, help="梯度平滑核大小(奇数)")
    p.add_argument("--ti_sigma", type=float, default=1.0, help="梯度平滑高斯sigma")
    p.add_argument("--random_start", action="store_true", default=True)  # 是否随机起点
    p.add_argument("--conf", type=float, default=0.2)  # 预测置信度阈值（适配攻击评估）
    p.add_argument("--iou", type=float, default=0.5)  # 匹配 IoU 阈值
    p.add_argument("--eval_set", type=str, default="test", choices=["train", "val", "trainval", "test"])  # 评估集划分
    p.add_argument("--num_eval", type=int, default=500)  # 评估图像数量
    p.add_argument("--runs", type=int, default=5)  # 重复运行次数
    p.add_argument("--load_model", type=str, required=True)  # 模型权重路径
    p.add_argument("--outdir", type=str, default="./adv_outputs/fgsm")  # 输出目录
    p.add_argument("--log_every", type=int, default=20)  # 每多少张打印一次进度
    args = p.parse_args()  # 解析参数
    if args.target_conf is None:
        args.target_conf = float(args.conf)

    if args.eps > 0.05:  # eps 超过项目约束则截断
        print(f"eps={args.eps} 超过 5%，已截断为 0.05")  # 打印截断提示
        args.eps = 0.05  # 将 eps 截断到 0.05

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
    print("Device:", args.device)  # 打印设备
    if args.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    os.makedirs(args.outdir, exist_ok=True)  # 创建输出目录
    ds = load_voc2007_dataset(args.eval_set)  # 加载数据集
    model = build_voc_model(args.device, args.load_model)  # 构建 YOLO 模型

    all_r = []  # 收集每次 run 的结果
    for i in range(1, args.runs + 1):  # 循环 runs 次
        r = evaluate_once(model, ds, args, i)  # 跑一次评估
        all_r.append(r)  # 保存结果
        print(f"\n[FGSM 第{i}次]")  # 打印轮次
        print(f"  Clean: P={r['clean'].precision*100:.2f}% R={r['clean'].recall*100:.2f}% F={r['clean'].f_score:.4f} Acc={r['clean'].accuracy*100:.2f}%")  # clean 指标
        print(f"  Adv:   P={r['adv'].precision*100:.2f}% R={r['adv'].recall*100:.2f}% F={r['adv'].f_score:.4f} Acc={r['adv'].accuracy*100:.2f}%")  # adv 指标
        print(f"  攻击成功率(GT/图像): {r['gt_asr']*100:.2f}% / {r['img_asr']*100:.2f}%")  # 两种 ASR
        print(f"  扰动: L2={r['pert'].avg_l2_distance:.4f} Linf={r['pert'].avg_linf_distance:.4f} MSE={r['pert'].avg_mse:.6f} SSIM={r['pert'].avg_ssim:.4f} PSNR={r['pert'].avg_psnr:.2f}")  # 扰动质量

    asr = np.array([r["gt_asr"] for r in all_r])  # 提取每次 run 的 GT ASR
    print("\n" + "=" * 60)  # 分隔线
    print("FGSM 汇总")  # 汇总标题
    print("=" * 60)  # 分隔线
    print(f"攻击成功率(GT级)均值: {asr.mean()*100:.2f}%")  # 平均 ASR
    print(f"是否满足 >85%: {asr.mean() > 0.85}")  # 达标检查


if __name__ == "__main__":  # 作为脚本直接运行时执行
    main()  # 进入主函数
