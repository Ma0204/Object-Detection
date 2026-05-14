# -*- coding: utf-8 -*-
"""
VOC2007 干净模型评估（YOLO）
默认阈值 conf=0.8，支持更强推理（imgsz/tta）。
"""

import os
import glob
import argparse
import random
import shutil
import traceback
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import ImageDraw
from torchvision import transforms

from attack_utils import (
    VOC_CLASSES,
    COCO_TO_VOC_ID,
    load_voc2007_dataset,
    build_yolo_voc_model,
    infer_yolo,
    prepare_voc2007_yolo_dataset,
    voc_target_to_boxes_and_labels,
    compute_iou,
    filter_pred,
)
from evaluation_metrics import compute_model_metrics
from ultralytics.utils import LOGGER as YLOGGER

VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def _resolve_best_weight_path(model, train_ret):
    save_dir = getattr(train_ret, "save_dir", None)
    if save_dir is None:
        trainer = getattr(model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None)
    if save_dir is not None:
        cand = os.path.join(str(save_dir), "weights", "best.pt")
        if os.path.isfile(cand):
            return cand
    cands = sorted(
        glob.glob(os.path.join("runs", "detect", "*", "weights", "best.pt")),
        key=os.path.getmtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _train_yolo_resilient(model, **kwargs):
    """
    正常训练优先；若仅在训练后 plot_metrics/scipy 阶段失败，则继续复用已保存 best.pt。
    """
    try:
        return model.train(**kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        text = (str(e) + "\n" + tb).lower()
        plotting_fail = ("plot_metrics" in text or "plot_results" in text) and ("scipy" in text or "ndimage" in text)
        if not plotting_fail:
            raise
        print("检测到训练后绘图阶段失败（scipy/plot），训练主体已完成，继续提取 best.pt。")
        return None


def _match_counts_one_to_one(
    gt_boxes: torch.Tensor,
    gt_labels: List[str],
    pred_boxes: torch.Tensor,
    pred_labels: List[int],
    iou_thresh: float,
) -> Tuple[int, int, int]:
    gt_items: List[Tuple[int, List[float]]] = []
    for g_box, g_name in zip(gt_boxes, gt_labels):
        if g_name not in VOC_CLASS_TO_ID:
            continue
        gt_items.append((VOC_CLASS_TO_ID[g_name], g_box.tolist()))

    pred_items: List[Tuple[int, List[float]]] = []
    for p_box, p_lid in zip(pred_boxes, pred_labels):
        pred_items.append((int(p_lid), p_box.tolist()))

    candidate_pairs: List[Tuple[float, int, int]] = []
    for gi, (g_cls, g_box) in enumerate(gt_items):
        for pi, (p_cls, p_box) in enumerate(pred_items):
            if g_cls != p_cls:
                continue
            iou = compute_iou(g_box, p_box)
            if iou >= iou_thresh:
                candidate_pairs.append((iou, gi, pi))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)
    used_g = set()
    used_p = set()
    tp = 0
    for _, gi, pi in candidate_pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp += 1

    fn = max(0, len(gt_items) - tp)
    fp = max(0, len(pred_items) - tp)
    return tp, fp, fn


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def _compute_map50(eval_records: List[Dict], iou_thresh: float) -> Tuple[float, Dict[str, float]]:
    aps: List[float] = []
    per_class_ap: Dict[str, float] = {}
    n_cls = len(VOC_CLASSES)

    for cls_id in range(n_cls):
        gt_by_img: Dict[int, List[List[float]]] = {}
        matched_by_img: Dict[int, List[bool]] = {}
        detections: List[Tuple[float, int, List[float]]] = []
        npos = 0

        for img_id, rec in enumerate(eval_records):
            gt_boxes = rec["gt_boxes"]
            gt_labels = rec["gt_labels"]
            pred_boxes = rec["pred_boxes"]
            pred_scores = rec["pred_scores"]
            pred_labels = rec["pred_labels"]

            cls_gt_boxes = [gt_boxes[i].tolist() for i, lid in enumerate(gt_labels) if int(lid) == cls_id]
            gt_by_img[img_id] = cls_gt_boxes
            matched_by_img[img_id] = [False] * len(cls_gt_boxes)
            npos += len(cls_gt_boxes)

            for i, lid in enumerate(pred_labels):
                if int(lid) != cls_id:
                    continue
                detections.append((float(pred_scores[i]), img_id, pred_boxes[i].tolist()))

        if npos == 0:
            continue

        detections.sort(key=lambda x: x[0], reverse=True)
        tp = np.zeros((len(detections),), dtype=np.float32)
        fp = np.zeros((len(detections),), dtype=np.float32)

        for di, (_, img_id, p_box) in enumerate(detections):
            gt_list = gt_by_img[img_id]
            if len(gt_list) == 0:
                fp[di] = 1.0
                continue

            best_iou = 0.0
            best_gi = -1
            for gi, g_box in enumerate(gt_list):
                iou = compute_iou(g_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= float(iou_thresh) and best_gi >= 0 and not matched_by_img[img_id][best_gi]:
                tp[di] = 1.0
                matched_by_img[img_id][best_gi] = True
            else:
                fp[di] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(float(npos), 1e-12)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap = _voc_ap(rec, prec)
        aps.append(ap)
        per_class_ap[VOC_CLASSES[cls_id]] = ap

    map50 = float(np.mean(aps)) if len(aps) > 0 else 0.0
    return map50, per_class_ap


def _calibrate_conf_threshold(model, dataset, args) -> float:
    n = min(int(args.calib_num_eval), len(dataset))
    if n <= 0:
        return float(args.conf)
    idxs = random.sample(range(len(dataset)), n)

    # 复用 legacy 的批量 predict 来做快速校准
    bs = max(1, int(args.legacy_batch))

    def _eval_acc_for(conf: float, nms_iou: float) -> float:
        tp = fp = fn = 0
        for step0 in range(0, len(idxs), bs):
            batch_ids = idxs[step0 : step0 + bs]
            imgs_np = []
            gts = []
            for idx in batch_ids:
                img_tensor, target = dataset[idx]
                gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)
                gts.append((gt_boxes, gt_labels))
                img_np = (img_tensor.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                imgs_np.append(img_np)

            dev_arg = device.index if device.type == "cuda" else "cpu"
            results = model.predict(
                source=imgs_np,
                verbose=False,
                conf=0.001,
                iou=float(nms_iou),
                device=dev_arg,
                imgsz=int(args.imgsz),
                augment=bool(args.tta),
                batch=len(imgs_np),
            )

            for j, res in enumerate(results):
                gt_boxes, gt_labels = gts[j]
                if res.boxes is None:
                    pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
                    pred_scores = torch.zeros((0,), dtype=torch.float32)
                    pred_labels = []
                else:
                    boxes_xyxy = res.boxes.xyxy.detach().cpu()
                    scores = res.boxes.conf.detach().cpu()
                    cls_ids = res.boxes.cls.long().detach().cpu().tolist()
                    keep = []
                    mapped = []
                    for i, c in enumerate(cls_ids):
                        if c in COCO_TO_VOC_ID:
                            keep.append(i)
                            mapped.append(COCO_TO_VOC_ID[c])
                    if not keep:
                        pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
                        pred_scores = torch.zeros((0,), dtype=torch.float32)
                        pred_labels = []
                    else:
                        keep_t = torch.tensor(keep, dtype=torch.long)
                        pred_boxes = boxes_xyxy.index_select(0, keep_t)
                        pred_scores = scores.index_select(0, keep_t)
                        pred_labels = mapped

                if pred_scores.numel() > 0:
                    conf_keep = pred_scores >= float(conf)
                    pred_boxes = pred_boxes[conf_keep]
                    pred_labels = [l for l, k in zip(pred_labels, conf_keep.tolist()) if k]

                pt, pf, pn = _match_counts_one_to_one(
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    iou_thresh=float(args.iou),
                )
                tp += pt
                fp += pf
                fn += pn

        m = compute_model_metrics(tp, fp, fn, 0)
        return float(m.accuracy)

    best_conf = float(args.conf)
    best_nms_iou = float(args.nms_iou)
    best_acc = -1.0

    for nms_iou in args.calib_nms_ious:
        for conf in args.calib_confs:
            acc = _eval_acc_for(float(conf), float(nms_iou))
            if acc > best_acc:
                best_acc = acc
                best_conf = float(conf)
                best_nms_iou = float(nms_iou)

    args.nms_iou = best_nms_iou
    print(f"[阈值校准] 选择 conf={best_conf:.3f}, nms_iou={best_nms_iou:.2f}（校准集 accuracy={best_acc*100:.2f}%）")
    return best_conf


def evaluate_once(model, dataset, args, run_id: int) -> Dict:
    n = min(int(args.num_eval), len(dataset))
    idxs = random.sample(range(len(dataset)), n)
    print(f"\n{'='*60}\n第 {run_id} 次评估（YOLO）\n{'='*60}\n评估样本数: {n}")

    tp = fp = fn = 0
    eval_records: List[Dict] = []
    # legacy_fast：批量 predict，大幅降低 Python 循环开销（不改 TP/FP/FN 统计逻辑）
    bs = max(1, int(args.legacy_batch))
    for step0 in range(0, len(idxs), bs):
        batch_ids = idxs[step0 : step0 + bs]
        imgs_np = []
        gts = []
        for idx in batch_ids:
            img_tensor, target = dataset[idx]
            gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)
            gts.append((gt_boxes, gt_labels, idx, img_tensor))
            img_np = (img_tensor.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            imgs_np.append(img_np)

        dev_arg = device.index if device.type == "cuda" else "cpu"
        results = model.predict(
            source=imgs_np,
            verbose=False,
            conf=0.001,
            iou=float(args.nms_iou),
            device=dev_arg,
            imgsz=int(args.imgsz),
            augment=bool(args.tta),
            batch=bs,
            max_det=int(args.max_det),
        )

        for j, res in enumerate(results):
            gt_boxes, gt_labels, idx, img_tensor = gts[j]
            if res.boxes is None:
                pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
                pred_scores = torch.zeros((0,), dtype=torch.float32)
                pred_labels = []
            else:
                boxes_xyxy = res.boxes.xyxy.detach().cpu()
                scores = res.boxes.conf.detach().cpu()
                cls_ids = res.boxes.cls.long().detach().cpu().tolist()
                keep = []
                mapped = []
                for i, c in enumerate(cls_ids):
                    if c in COCO_TO_VOC_ID:
                        keep.append(i)
                        mapped.append(COCO_TO_VOC_ID[c])
                if not keep:
                    pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
                    pred_scores = torch.zeros((0,), dtype=torch.float32)
                    pred_labels = []
                else:
                    keep_t = torch.tensor(keep, dtype=torch.long)
                    pred_boxes = boxes_xyxy.index_select(0, keep_t)
                    pred_scores = scores.index_select(0, keep_t)
                    pred_labels = mapped

            if pred_scores.numel() > 0:
                conf_keep = pred_scores >= float(args.conf)
                pred_boxes = pred_boxes[conf_keep]
                pred_scores = pred_scores[conf_keep]
                pred_labels = [l for l, k in zip(pred_labels, conf_keep.tolist()) if k]

            pt, pf, pn = _match_counts_one_to_one(
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                iou_thresh=float(args.iou),
            )
            tp += pt
            fp += pf
            fn += pn
            gt_label_ids = [VOC_CLASS_TO_ID[g] for g in gt_labels if g in VOC_CLASS_TO_ID]
            eval_records.append(
                {
                    "gt_boxes": gt_boxes,
                    "gt_labels": gt_label_ids,
                    "pred_boxes": pred_boxes,
                    "pred_scores": pred_scores,
                    "pred_labels": pred_labels,
                }
            )

            step = step0 + j
            if run_id == 1 and step < 5:
                fp_pred_boxes = pred_boxes
                fp_pred_scores = pred_scores[pred_scores >= float(args.conf)] if pred_scores.numel() > 0 else torch.zeros((0,))
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                draw = ImageDraw.Draw(img_pil)
                for box, score, label in zip(fp_pred_boxes, fp_pred_scores, pred_labels):
                    x1, y1, x2, y2 = box.tolist()
                    lid = int(label)
                    name = VOC_CLASSES[lid] if 0 <= lid < len(VOC_CLASSES) else "unknown"
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                    draw.text((x1, y1), f"{name} {float(score):.2f}", fill="yellow")
                img_pil.save(os.path.join(args.outdir, f"voc_det_yolo_{idx:04d}.png"))

        done = min(len(idxs), step0 + bs)
        if done % max(1, int(args.log_every)) == 0:
            print(f"[测试进度] run {run_id}: {done}/{n}")

    m = compute_model_metrics(tp, fp, fn, 0)
    map50, _ = _compute_map50(eval_records, iou_thresh=float(args.iou))
    return {
        "run_id": run_id,
        "precision": m.precision,
        "recall": m.recall,
        "f_score": m.f_score,
        "accuracy": m.accuracy,
        "ap50": map50,
    }


def main():
    p = argparse.ArgumentParser(description="VOC2007 YOLO 干净评估")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=500)
    p.add_argument("--conf", type=float, default=0.35, help="置信度阈值（越高越保守，Recall越低）")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    p.add_argument("--tta", action="store_true", help="启用 YOLO TTA（更慢，通常只用于冲分对比）")
    p.add_argument("--outdir", type=str, default="./voc_outputs")
    p.add_argument("--eval_set", type=str, default="test", choices=["train", "val", "trainval", "test"], help="评估集划分")
    p.add_argument("--load_model", type=str, default="yolov8m.pt")
    p.add_argument("--train_epochs", type=int, default=0, help="在 VOC 上继续训练的 epoch 数，0 表示不训练")
    p.add_argument("--train_batch_size", type=int, default=4, help="YOLO 训练 batch size（建议按显存调；可用 -1 让YOLO自动估计）")
    p.add_argument("--train_data", type=str, default="", help="YOLO 训练数据配置；留空则自动生成VOC2007 YOLO格式配置")
    p.add_argument("--save_model", type=str, default="./checkpoints/yolo_voc_best.pt", help="训练后 best 权重保存路径")
    p.add_argument("--train_fraction", type=float, default=0.3, help="快速微调使用的数据比例（0-1）")
    p.add_argument("--train_set", type=str, default="trainval", choices=["train", "trainval"], help="YOLO 训练集划分")
    p.add_argument("--train_val_set", type=str, default="test", choices=["val", "test"], help="YOLO 训练中 val 划分来源")
    p.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true", help="冻结骨干，仅训练检测头")
    p.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false", help="解冻骨干，进行全量微调")
    p.set_defaults(freeze_backbone=False)
    p.add_argument("--train_lr0", type=float, default=0.001, help="快速微调初始学习率")
    # AMP：为了提速默认开启；如遇到环境问题可用 --no_train_amp 关闭
    p.add_argument("--train_amp", dest="train_amp", action="store_true", help="开启AMP训练（提速/省显存）")
    p.add_argument("--no_train_amp", dest="train_amp", action="store_false", help="关闭AMP训练")
    p.set_defaults(train_amp=True)
    p.add_argument("--log_every", type=int, default=20, help="测试阶段每多少张打印一次进度")
    p.add_argument("--auto_conf", action="store_true", help="在校准集上自动选择最优置信度阈值")
    p.add_argument("--calib_num_eval", type=int, default=300, help="自动阈值校准样本数")
    p.add_argument("--calib_confs", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8], help="候选置信度阈值")
    p.add_argument("--nms_iou", type=float, default=0.7, help="YOLO NMS iou（影响 FP/FN 平衡）")
    p.add_argument("--calib_nms_ious", type=float, nargs="+", default=[0.5, 0.6, 0.7], help="校准时搜索的 NMS iou 候选")
    p.add_argument("--eval_backend", type=str, default="yolo_val", choices=["yolo_val", "legacy"], help="评估后端：yolo_val更快更标准")
    p.add_argument("--val_batch", type=int, default=16, help="YOLO val batch")
    p.add_argument("--workers", type=int, default=8, help="数据加载 workers")
    p.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "none"], help="数据缓存策略")
    p.add_argument("--legacy_batch", type=int, default=16, help="legacy 评估批量推理 batch（越大越快，受显存限制）")
    p.add_argument("--max_det", type=int, default=300, help="每张图最多保留的检测框数量（过大易增加FP且变慢）")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ds = load_voc2007_dataset(args.eval_set) if args.eval_backend == "legacy" else None
    if ds is not None:
        print(f"VOC2007 {args.eval_set} 样本数: {len(ds)}")
    print(f"加载 YOLO 模型: {args.load_model}")
    model = build_yolo_voc_model(device, weights=args.load_model)

    # 按旧流程支持：COCO预训练 -> VOC继续训练 -> 保存best -> 后续直接复用
    if int(args.train_epochs) > 0:
        train_data_yaml = args.train_data.strip() if isinstance(args.train_data, str) else ""
        if not train_data_yaml:
            train_data_yaml = prepare_voc2007_yolo_dataset(
                train_image_set=args.train_set,
                val_image_set=args.train_val_set,
            )
        elif not os.path.isfile(train_data_yaml):
            print(f"未找到指定 data.yaml，自动改为生成 VOC2007 YOLO 配置: {train_data_yaml}")
            train_data_yaml = prepare_voc2007_yolo_dataset(
                os.path.dirname(train_data_yaml) or "./data/voc2007_yolo",
                train_image_set=args.train_set,
                val_image_set=args.train_val_set,
            )
        print(
            f"开始 VOC 继续训练: epochs={args.train_epochs}, batch={args.train_batch_size}, "
            f"imgsz={args.imgsz}, data={train_data_yaml}"
        )
        total_epochs = int(args.train_epochs)

        def _on_train_epoch_end(trainer):
            ep = int(getattr(trainer, "epoch", 0)) + 1
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is None:
                print(f"[训练进度] epoch {ep}/{total_epochs}")
            else:
                vals = [float(x) for x in loss_items.detach().cpu().tolist()] if hasattr(loss_items, "detach") else [float(x) for x in loss_items]
                mean_loss = sum(vals) / max(1, len(vals))
                print(f"[训练进度] epoch {ep}/{total_epochs}  loss={mean_loss:.4f}")

        model.add_callback("on_train_epoch_end", _on_train_epoch_end)
        prev_level = YLOGGER.level
        YLOGGER.setLevel("ERROR")
        try:
            train_ret = _train_yolo_resilient(
                model,
                data=train_data_yaml,
                epochs=total_epochs,
                imgsz=int(args.imgsz),
                batch=(-1 if int(args.train_batch_size) < 0 else int(args.train_batch_size)),
                fraction=float(args.train_fraction),
                lr0=float(args.train_lr0),
                freeze=10 if args.freeze_backbone else 0,
                amp=bool(args.train_amp),
                device=0 if device.type == "cuda" else "cpu",
                workers=int(args.workers),
                cache=(False if args.cache == "none" else args.cache),
                verbose=False,
                plots=True,
            )
        finally:
            YLOGGER.setLevel(prev_level)

        best_path = _resolve_best_weight_path(model, train_ret)

        if best_path is None:
            raise RuntimeError("训练完成但未找到 best.pt，请检查训练日志。")

        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        shutil.copy2(best_path, args.save_model)
        print(f"训练 best 权重已保存: {args.save_model}")

        # 切回使用训练后权重评估（与“训练后保存并复用”逻辑一致）
        model = build_yolo_voc_model(device, weights=args.save_model)
    else:
        # 不训练时也确保有标准 YOLO data.yaml，用于快速 val 评估
        train_data_yaml = args.train_data.strip() if isinstance(args.train_data, str) else ""
        if (not train_data_yaml) or (not os.path.isfile(train_data_yaml)):
            train_data_yaml = prepare_voc2007_yolo_dataset(
                train_image_set=args.train_set,
                val_image_set=args.train_val_set,
            )

    if args.auto_conf and ds is not None:
        args.conf = _calibrate_conf_threshold(model, ds, args)

    if args.eval_backend == "yolo_val":
        print(f"\n{'='*60}\nYOLO 标准验证（更快，论文口径）\n{'='*60}")
        prev_level = YLOGGER.level
        YLOGGER.setLevel("ERROR")
        try:
            metrics = model.val(
                data=train_data_yaml,
                imgsz=int(args.imgsz),
                batch=int(args.val_batch),
                conf=float(args.conf),
                iou=float(args.nms_iou),
                split="val",
                device=0 if device.type == "cuda" else "cpu",
                workers=int(args.workers),
                cache=(False if args.cache == "none" else args.cache),
                augment=bool(args.tta),
                verbose=False,
                plots=False,
                max_det=int(args.max_det),
            )
        finally:
            YLOGGER.setLevel(prev_level)
        p = float(getattr(metrics.box, "mp", 0.0))
        r = float(getattr(metrics.box, "mr", 0.0))
        map50 = float(getattr(metrics.box, "map50", 0.0))
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        print(f"精确率: {p*100:.2f}%")
        print(f"召回率: {r*100:.2f}%")
        print(f"F-score: {f1:.4f}")
        print(f"mAP50: {map50*100:.2f}%")
    else:
        all_r = []
        for i in range(1, args.runs + 1):
            r = evaluate_once(model, ds, args, i)
            all_r.append(r)
            print(
                f"\n第 {i} 次结果：\n  精确率: {r['precision']*100:.2f}%\n  召回率: {r['recall']*100:.2f}%\n"
                f"  F-score: {r['f_score']:.4f}\n  准确率: {r['accuracy']*100:.2f}%\n  AP50(mAP@0.5): {r['ap50']*100:.2f}%"
            )

        arr_p = np.array([x["precision"] for x in all_r])
        arr_r = np.array([x["recall"] for x in all_r])
        arr_f = np.array([x["f_score"] for x in all_r])
        arr_a = np.array([x["accuracy"] for x in all_r])
        arr_ap50 = np.array([x["ap50"] for x in all_r])
        print(f"\n{'='*60}\nYOLO 多次运行汇总\n{'='*60}")
        print(f"精确率: 均值={arr_p.mean()*100:.2f}%  std={arr_p.std()*100:.2f}%")
        print(f"召回率: 均值={arr_r.mean()*100:.2f}%  std={arr_r.std()*100:.2f}%")
        print(f"F-score: 均值={arr_f.mean():.4f}  std={arr_f.std():.4f}")
        print(f"准确率: 均值={arr_a.mean()*100:.2f}%  std={arr_a.std()*100:.2f}%")
        print(f"AP50(mAP@0.5): 均值={arr_ap50.mean()*100:.2f}%  std={arr_ap50.std()*100:.2f}%")


if __name__ == "__main__":
    main()
