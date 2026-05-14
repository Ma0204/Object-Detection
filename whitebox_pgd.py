# -*- coding: utf-8 -*-  # UTF-8 编码，支持中文注释
# ========== 文件说明 ==========  # 分区注释
# 白盒攻击：PGD（VOC 目标检测）  # 脚本主题
# 说明：PGD=多步迭代 + 每步投影回 eps 约束；可选 random_start 提升攻击强度  # 方法说明

from __future__ import annotations  # 延迟解析类型注解，兼容新语法

import argparse  # 命令行参数解析
import os  # 目录与路径操作
import random  # 随机抽样评估图片
import time  # 计时与进度显示
from typing import Dict  # 类型提示（infer 输出）

import numpy as np  # runs 汇总统计
import torch  # 张量与自动求导

from attack_utils import (  # 项目工具函数
    build_yolo_voc_model,  # 构建 YOLO VOC 模型
    ensure_tensor_01,  # 像素裁剪到 [0,1]
    infer_yolo,  # YOLO 推理
    load_voc2007_dataset,  # 加载 VOC 数据
    voc_target_to_boxes_and_labels,  # 解析 VOC 标注
    compute_iou,  # IoU 计算
    filter_pred,  # 预测过滤
    yolo_whitebox_objective,  # YOLO 白盒目标
    input_diversity,  # DI
    ti_smooth_grad,  # TI
)
from evaluation_metrics import compute_model_metrics, compute_perturbation_metrics  # 指标计算

VOC_CLASSES = [  # VOC 类别列表（含 background）
    "background",  # 背景
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",  # 1-9
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",  # 10-17
    "sofa", "train", "tvmonitor",  # 18-20
]  # 列表结束
VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}  # 名称到 id 映射


def build_voc_model(device: torch.device, weights: str) -> torch.nn.Module:  # 构建 YOLO(VOC映射) 模型
    return build_yolo_voc_model(device, weights=weights)  # 返回模型


@torch.no_grad()  # 推理不建图，节省显存
def infer(model: torch.nn.Module, img: torch.Tensor) -> Dict[str, torch.Tensor]:  # 单图推理接口
    return infer_yolo(model, img)  # 调用 YOLO 适配推理


def pgd_attack(
    model: torch.nn.Module,
    img_01: torch.Tensor,
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
) -> torch.Tensor:  # PGD 攻击主函数
    x0 = img_01.detach()  # 保存原图（不参与梯度）
    x = ensure_tensor_01(x0 + torch.empty_like(x0).uniform_(-eps, eps)) if random_start else x0.clone()  # 可选随机起点
    for si in range(int(steps)):  # 迭代 steps 次（支持提前早停）
        x = x.detach().clone().requires_grad_(True)  # 使 x 可求导
        x_in = input_diversity(x, prob=float(di_prob), scale_min=float(di_scale_min))
        loss = yolo_whitebox_objective(model, x_in, topk=300, target_conf=float(target_conf))  # YOLO 可微目标
        loss.backward()  # 反向传播得到 x.grad
        grad = ti_smooth_grad(x.grad.detach(), kernel_size=int(ti_kernel), sigma=float(ti_sigma))
        with torch.no_grad():  # 更新不需要梯度图
            x = x - alpha * grad.sign()  # 沿梯度下降方向更新（减小检测目标，提升攻击性）
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)  # 投影到 eps 约束
            x = ensure_tensor_01(x)  # 像素裁剪到 [0,1]
        # 自适应早停：达到目标强度后提前结束，缩短平均耗时
        if float(stop_loss) >= 0.0 and (si + 1) >= int(min_steps) and float(loss.detach().item()) <= float(stop_loss):
            break
    return x.detach()  # 返回对抗样本


def evaluate_once(model, dataset, args, run_id: int):  # 单次评估：抽样 n 张图并统计全部指标
    n = min(int(args.num_eval), len(dataset))  # 本轮实际评估张数
    idxs = random.sample(range(len(dataset)), n)  # 随机抽样索引
    t0 = time.time()  # 本轮起始时间

    c_tp = c_fp = c_fn = 0  # clean 侧 TP/FP/FN
    a_tp = a_fp = a_fn = 0  # adv 侧 TP/FP/FN
    gt_detected_clean = gt_success_count = 0  # GT 级 ASR 计数（分母/分子）
    img_detected_clean = img_success_count = 0  # 图像级 ASR 计数（分母/分子）
    clean_imgs, adv_imgs = [], []  # 用于计算扰动指标的图像对（最多 200 对）

    def count_metrics(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh):  # 单图 TP/FP/FN 统计
        tp_here = fn_here = matched_here = 0  # 本图局部 TP/FN/匹配数
        for g_box, g_name in zip(gt_boxes, gt_labels):  # 遍历每个 GT 目标
            if g_name not in VOC_CLASS_TO_ID:  # 跳过未知类别
                continue  # 处理下一个 GT
            gt_lid = VOC_CLASS_TO_ID[g_name]  # GT 类别 id
            g_box_list = g_box.tolist()  # GT 框转 list
            found = False  # 当前 GT 是否匹配到预测
            for p_box, p_lid in zip(pred_boxes, pred_labels):  # 遍历预测框
                if int(p_lid) == int(gt_lid) and compute_iou(g_box_list, p_box.tolist()) >= iou_thresh:  # 类别一致且 IoU 达标
                    found = True  # 标记匹配成功
                    break  # 单个 GT 只记一次匹配
            if found:  # 命中
                tp_here += 1; matched_here += 1  # TP 与匹配计数加一
            else:  # 未命中
                fn_here += 1  # FN 加一
        fp_here = max(0, len(pred_labels) - matched_here)  # 简化：未匹配到 GT 的预测计为 FP
        return tp_here, fp_here, fn_here, matched_here  # 返回本图统计

    for step, idx in enumerate(idxs):
        img_tensor, target = dataset[idx]  # 取图像与标注
        img = img_tensor.to(args.device)  # 放到设备
        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)  # 解析 GT 框与类别名

        pred_clean = infer(model, img)  # clean 推理
        fp_clean = filter_pred(pred_clean, conf_thresh=args.conf)  # 过滤低置信度预测
        boxes_c = fp_clean["boxes"].cpu()  # clean 预测框
        labels_c = fp_clean["labels"].cpu().tolist()  # clean 预测类别列表

        adv = pgd_attack(
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
        )  # PGD 生成对抗图

        pred_adv = infer(model, adv)  # adv 推理
        fp_adv = filter_pred(pred_adv, conf_thresh=args.conf)  # 同阈值过滤
        boxes_a = fp_adv["boxes"].cpu()  # adv 预测框
        labels_a = fp_adv["labels"].cpu().tolist()  # adv 预测类别

        ctp, cfp, cfn, cm = count_metrics(gt_boxes, gt_labels, boxes_c, labels_c, args.iou)  # clean TP/FP/FN
        atp, afp, afn, am = count_metrics(gt_boxes, gt_labels, boxes_a, labels_a, args.iou)  # adv TP/FP/FN
        c_tp += ctp; c_fp += cfp; c_fn += cfn  # 累加 clean
        a_tp += atp; a_fp += afp; a_fn += afn  # 累加 adv

        for g_box, g_name in zip(gt_boxes, gt_labels):  # 统计 GT 级 ASR
            if g_name not in VOC_CLASS_TO_ID:  # 跳过未知类别
                continue  # 下一个 GT
            gt_lid = VOC_CLASS_TO_ID[g_name]  # GT 类别 id
            g_box_list = g_box.tolist()  # GT 框
            c_hit = any(int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= args.iou for pb, pl in zip(boxes_c, labels_c))  # clean 是否命中
            if c_hit:  # 仅对 clean 命中的 GT 计入分母
                gt_detected_clean += 1  # GT 级分母 +1
                a_hit = any(int(pl) == int(gt_lid) and compute_iou(g_box_list, pb.tolist()) >= args.iou for pb, pl in zip(boxes_a, labels_a))  # adv 是否仍命中
                if not a_hit:  # adv 未命中则视作攻击成功
                    gt_success_count += 1  # GT 级分子 +1

        if cm > 0 and gt_boxes.size(0) > 0:  # clean 图像至少命中 1 个 GT
            img_detected_clean += 1  # 图像级分母 +1
            if am == 0:  # adv 图像 0 命中
                img_success_count += 1  # 图像级攻击成功 +1

        if len(clean_imgs) < 200:  # 控制计算扰动指标的样本对数量
            clean_imgs.append(img.detach().cpu())  # 保存 clean 图像
            adv_imgs.append(adv.detach().cpu())  # 保存 adv 图像

        if (step + 1) % max(1, int(args.log_every)) == 0 or (step + 1) == n:  # 按 log_every 打印实时进度
            done = step + 1
            elapsed = time.time() - t0
            speed = done / max(elapsed, 1e-6)
            eta = (n - done) / max(speed, 1e-6)
            cur_gt_asr = (gt_success_count / max(1, gt_detected_clean)) * 100.0
            print(
                f"  [PGD 第{run_id}次] 进度 {done}/{n} ({done/max(1,n)*100:.1f}%) | "
                f"当前GT-ASR={cur_gt_asr:.2f}% | 用时={elapsed/60:.1f}m | ETA={eta/60:.1f}m"
            )

    clean_metrics = compute_model_metrics(c_tp, c_fp, c_fn, 0)  # 汇总 clean 指标
    adv_metrics = compute_model_metrics(a_tp, a_fp, a_fn, 0)  # 汇总 adv 指标
    gt_asr = gt_success_count / max(1, gt_detected_clean)  # GT 级 ASR
    img_asr = img_success_count / max(1, img_detected_clean)  # 图像级 ASR
    pert = compute_perturbation_metrics(clean_imgs, adv_imgs, gt_asr)  # 扰动质量指标
    return {"clean": clean_metrics, "adv": adv_metrics, "gt_asr": gt_asr, "img_asr": img_asr, "pert": pert}  # 返回结果字典


def main():  # 主流程：解析参数、加载模型、运行评估并汇总
    p = argparse.ArgumentParser()  # 参数解析器
    p.add_argument("--eps", type=float, default=0.05)  # 扰动上限
    p.add_argument("--steps", type=int, default=100)  # 迭代步数（默认提速）
    p.add_argument("--alpha", type=float, default=0.0010)  # 每步步长
    p.add_argument(
        "--target_conf",
        type=float,
        default=None,
        help="白盒目标阈值项；默认与 --conf 一致以便与评估对齐",
    )
    p.add_argument("--min_steps", type=int, default=40, help="允许早停前至少执行的步数")
    p.add_argument(
        "--stop_loss",
        type=float,
        default=-1.0,
        help="早停阈值；<0 关闭早停",
    )
    p.add_argument("--di_prob", type=float, default=0.5, help="输入多样化触发概率")
    p.add_argument("--di_scale_min", type=float, default=0.92, help="输入多样化最小缩放比例")
    p.add_argument("--ti_kernel", type=int, default=5, help="梯度平滑核大小(奇数)")
    p.add_argument("--ti_sigma", type=float, default=1.0, help="梯度平滑高斯sigma")
    p.add_argument("--random_start", action="store_true", default=True)  # 是否随机起点
    p.add_argument("--conf", type=float, default=0.2)  # 置信度阈值（适配攻击评估）
    p.add_argument("--iou", type=float, default=0.5)  # IoU 匹配阈值
    p.add_argument("--eval_set", type=str, default="test", choices=["train", "val", "trainval", "test"])  # 评估集划分
    p.add_argument("--num_eval", type=int, default=500)  # 评估图像数
    p.add_argument("--runs", type=int, default=5)  # 重复运行次数
    p.add_argument("--load_model", type=str, required=True)  # 模型权重路径
    p.add_argument("--outdir", type=str, default="./adv_outputs/pgd")  # 输出目录
    p.add_argument("--log_every", type=int, default=20)  # 每多少张打印一次进度
    args = p.parse_args()  # 解析参数
    if args.target_conf is None:
        args.target_conf = float(args.conf)

    if args.eps > 0.05:  # eps 超过项目约束则截断
        print(f"eps={args.eps} 超过 5%，已截断为 0.05")  # 提示截断
        args.eps = 0.05  # 强制满足约束

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
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

    all_r = []  # 存储每次 run 的结果
    for i in range(1, args.runs + 1):  # 多次重复评估
        r = evaluate_once(model, ds, args, i)  # 执行一次评估
        all_r.append(r)  # 保存结果
        print(f"\n[PGD 第{i}次]")  # 轮次标题
        print(f"  Clean: P={r['clean'].precision*100:.2f}% R={r['clean'].recall*100:.2f}% F={r['clean'].f_score:.4f} Acc={r['clean'].accuracy*100:.2f}%")  # clean 指标
        print(f"  Adv:   P={r['adv'].precision*100:.2f}% R={r['adv'].recall*100:.2f}% F={r['adv'].f_score:.4f} Acc={r['adv'].accuracy*100:.2f}%")  # adv 指标
        print(f"  攻击成功率(GT/图像): {r['gt_asr']*100:.2f}% / {r['img_asr']*100:.2f}%")  # ASR 指标
        print(f"  扰动: L2={r['pert'].avg_l2_distance:.4f} Linf={r['pert'].avg_linf_distance:.4f} MSE={r['pert'].avg_mse:.6f} SSIM={r['pert'].avg_ssim:.4f} PSNR={r['pert'].avg_psnr:.2f}")  # 扰动指标

    asr = np.array([r["gt_asr"] for r in all_r])  # 汇总 GT 级 ASR
    print("\n" + "=" * 60)  # 分隔线
    print("PGD 汇总")  # 标题
    print("=" * 60)  # 分隔线
    print(f"攻击成功率(GT级)均值: {asr.mean()*100:.2f}%")  # 平均 ASR
    print(f"是否满足 >85%: {asr.mean() > 0.85}")  # 达标检查


if __name__ == "__main__":  # 脚本直跑入口
    main()  # 执行主函数
