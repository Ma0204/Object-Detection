# -*- coding: utf-8 -*-
"""
通用工具：VOC2007 下载/加载、Faster R-CNN 加载、伪标注生成、简单评估、保存可视化等。

说明：
- 本项目使用 torchvision 的 `fasterrcnn_resnet50_fpn`（COCO 预训练）。
- “白盒攻击”针对检测模型的 loss 做梯度（需要 targets）。
  这里使用“自监督伪标注”：先用干净图像推理得到高置信预测框作为 targets，再最大化 loss 攻击输入。
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
import shutil
import stat
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import ImageDraw
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)


DATA_DIR = "./data"
VOC_TAR_URL = "https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_06-Nov-2007.tar"
VOC_TAR_NAME = "VOCtrainval_06-Nov-2007.tar"
VOC_TEST_TAR_URL = "https://data.brainchip.com/dataset-mirror/voc/VOCtest_06-Nov-2007.tar"
VOC_TEST_TAR_NAME = "VOCtest_06-Nov-2007.tar"
VOCDEVKIT_DIR = os.path.join(DATA_DIR, "VOCdevkit")
VOC2007_DIR = os.path.join(VOCDEVKIT_DIR, "VOC2007")

VOC_DET_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]
VOC_DET_CLASS_TO_ID = {n: i for i, n in enumerate(VOC_DET_CLASSES)}

def safe_extract_tar(tar_path: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)

    def _ensure_dir(path: str) -> None:
        if os.path.isdir(path):
            return
        if os.path.exists(path) and (not os.path.isdir(path)):
            try:
                os.remove(path)
            except Exception:
                try:
                    os.chmod(path, stat.S_IWRITE)
                    os.remove(path)
                except Exception:
                    pass
        os.makedirs(path, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            name = m.name.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                continue
            out_path = os.path.join(dst_dir, *name.split("/"))
            if m.isdir():
                _ensure_dir(out_path)
                continue
            parent = os.path.dirname(out_path)
            _ensure_dir(parent)
            f = tar.extractfile(m)
            if f is None:
                continue
            with f:
                with open(out_path, "wb") as wf:
                    shutil.copyfileobj(f, wf)


def download_voc2007_trainval() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    jpeg_dir = os.path.join(VOC2007_DIR, "JPEGImages")
    if os.path.isdir(jpeg_dir):
        print(f"已存在 VOC2007 目录: {VOC2007_DIR}")
        return DATA_DIR
    if os.path.exists(VOC2007_DIR) and (not os.path.isdir(jpeg_dir)):
        print(f"检测到残留/不完整目录，先清理: {VOC2007_DIR}")
        try:
            shutil.rmtree(VOC2007_DIR, ignore_errors=True)
        except Exception:
            try:
                os.remove(VOC2007_DIR)
            except Exception:
                pass
    if os.path.exists(VOCDEVKIT_DIR) and (not os.path.isdir(jpeg_dir)):
        try:
            shutil.rmtree(VOCDEVKIT_DIR, onerror=lambda func, path, exc: (os.chmod(path, stat.S_IWRITE), func(path)))
        except Exception:
            pass

    tar_path = os.path.join(DATA_DIR, VOC_TAR_NAME)
    if not os.path.isfile(tar_path):
        print("正在下载 VOC2007 trainval 数据集（约 450MB，可能需要较长时间）...")
        req = urllib.request.Request(VOC_TAR_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            with open(tar_path, "wb") as f:
                f.write(r.read())

    print("正在解压 VOC2007 trainval ...")
    safe_extract_tar(tar_path, DATA_DIR)

    if not os.path.isdir(jpeg_dir):
        raise RuntimeError(f"VOC2007 解压后仍未找到 JPEGImages: {jpeg_dir}")
    print(f"VOC2007 已解压到: {VOC2007_DIR}")
    return DATA_DIR


def load_voc2007_trainval_dataset() -> VOCDetection:
    root_for_voc = download_voc2007_trainval()
    transform_base = transforms.ToTensor()
    return VOCDetection(
        root=root_for_voc,
        year="2007",
        image_set="trainval",
        download=False,
        transform=transform_base,
        target_transform=None,
    )


def download_voc2007_test() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    # test 划分是否可用，以 test.txt 作为存在判定更稳妥
    test_ids = os.path.join(VOC2007_DIR, "ImageSets", "Main", "test.txt")
    if os.path.isfile(test_ids):
        return DATA_DIR
    download_voc2007_trainval()

    tar_path = os.path.join(DATA_DIR, VOC_TEST_TAR_NAME)
    if not os.path.isfile(tar_path):
        print("正在下载 VOC2007 test 数据集（约 430MB，可能需要较长时间）...")
        req = urllib.request.Request(VOC_TEST_TAR_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            with open(tar_path, "wb") as f:
                f.write(r.read())

    print("正在解压 VOC2007 test ...")
    safe_extract_tar(tar_path, DATA_DIR)
    if not os.path.isfile(test_ids):
        raise RuntimeError(f"VOC2007 test 解压后仍未找到 test 划分文件: {test_ids}")
    return DATA_DIR


def load_voc2007_dataset(image_set: str = "trainval") -> VOCDetection:
    image_set = str(image_set).strip().lower()
    if image_set == "test":
        root_for_voc = download_voc2007_test()
    else:
        root_for_voc = download_voc2007_trainval()
    transform_base = transforms.ToTensor()
    return VOCDetection(
        root=root_for_voc,
        year="2007",
        image_set=image_set,
        download=False,
        transform=transform_base,
        target_transform=None,
    )


def prepare_voc2007_yolo_dataset(
    output_root: str = "./data/voc2007_yolo",
    train_image_set: str = "trainval",
    val_image_set: str = "test",
) -> str:
    """
    将 VOC2007 XML 标注转换为 YOLO txt，并生成可直接给 Ultralytics 使用的 data.yaml。
    返回生成的 yaml 路径。
    """
    download_voc2007_trainval()
    use_test_for_val = str(val_image_set).strip().lower() == "test"
    if use_test_for_val:
        try:
            download_voc2007_test()
        except Exception as e:
            print(f"警告：下载/加载 VOC2007 test 失败，将回退到 train/val 划分。原因: {e}")
            use_test_for_val = False
    jpeg_dir = os.path.join(VOC2007_DIR, "JPEGImages")
    ann_dir = os.path.join(VOC2007_DIR, "Annotations")
    set_dir = os.path.join(VOC2007_DIR, "ImageSets", "Main")

    train_ids_path = os.path.join(set_dir, "train.txt")
    trainval_ids_path = os.path.join(set_dir, "trainval.txt")
    val_ids_path = os.path.join(set_dir, "val.txt")
    test_ids_path = os.path.join(set_dir, "test.txt")

    def _read_ids(p: str) -> List[str]:
        if not os.path.isfile(p):
            return []
        with open(p, "r", encoding="utf-8") as f:
            return [x.strip() for x in f.readlines() if x.strip()]

    train_image_set = str(train_image_set).strip().lower()
    if train_image_set == "trainval":
        train_ids = _read_ids(trainval_ids_path)
    else:
        train_ids = _read_ids(train_ids_path)
    val_ids = _read_ids(test_ids_path) if use_test_for_val else _read_ids(val_ids_path)
    if not train_ids and not val_ids:
        tv = _read_ids(trainval_ids_path)
        n_tr = int(0.9 * len(tv))
        train_ids = tv[:n_tr]
        val_ids = tv[n_tr:]

    images_train = os.path.join(output_root, "images", "train")
    images_val = os.path.join(output_root, "images", "val")
    labels_train = os.path.join(output_root, "labels", "train")
    labels_val = os.path.join(output_root, "labels", "val")
    for d in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(d, exist_ok=True)

    def _link_or_copy(src: str, dst: str) -> None:
        if os.path.exists(dst):
            return
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)

    def _convert_one(img_id: str, split: str) -> None:
        xml_path = os.path.join(ann_dir, f"{img_id}.xml")
        img_path = os.path.join(jpeg_dir, f"{img_id}.jpg")
        if not os.path.isfile(xml_path) or not os.path.isfile(img_path):
            return

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            return
        w = float(size.findtext("width", default="0"))
        h = float(size.findtext("height", default="0"))
        if w <= 1 or h <= 1:
            return

        yolo_lines: List[str] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="")
            if name not in VOC_DET_CLASS_TO_ID:
                continue
            b = obj.find("bndbox")
            if b is None:
                continue
            xmin = float(b.findtext("xmin", default="0"))
            ymin = float(b.findtext("ymin", default="0"))
            xmax = float(b.findtext("xmax", default="0"))
            ymax = float(b.findtext("ymax", default="0"))
            if xmax <= xmin or ymax <= ymin:
                continue
            cx = ((xmin + xmax) / 2.0) / w
            cy = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            cls = VOC_DET_CLASS_TO_ID[name]
            yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if split == "train":
            img_dst = os.path.join(images_train, f"{img_id}.jpg")
            lbl_dst = os.path.join(labels_train, f"{img_id}.txt")
        else:
            img_dst = os.path.join(images_val, f"{img_id}.jpg")
            lbl_dst = os.path.join(labels_val, f"{img_id}.txt")

        _link_or_copy(img_path, img_dst)
        with open(lbl_dst, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    for i in train_ids:
        _convert_one(i, "train")
    for i in val_ids:
        _convert_one(i, "val")

    yaml_path = os.path.join(output_root, "voc2007_only.yaml")
    names_yaml = "\n".join([f"  {i}: {n}" for i, n in enumerate(VOC_DET_CLASSES)])
    yaml_text = (
        f"path: {output_root}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    return yaml_path


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    coco_classes: List[str]


def load_fasterrcnn_coco(device: torch.device) -> ModelBundle:
    print("加载 Faster R-CNN（COCO 预训练）...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    coco_classes: List[str] = weights.meta["categories"]
    return ModelBundle(model=model, coco_classes=coco_classes)


VOC_CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]
VOC_TO_COCO = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "sofa": "couch",
    "train": "train",
    "tvmonitor": "tv",
}
COCO_TO_VOC_ID: Dict[int, int] = {}


def build_yolo_voc_model(device: torch.device, weights: str = "yolov8n.pt"):
    try:
        from ultralytics import YOLO as _YOLO
    except ModuleNotFoundError as e:
        raise ImportError(
            "未安装 ultralytics。请在运行本脚本的「同一个 Python」里安装，在 CMD 中执行:\n"
            "  python -m pip install ultralytics\n"
            "若系统有多个 Python，请先确认当前解释器路径:\n"
            "  where python\n"
            "  python -c \"import sys; print(sys.executable)\""
        ) from e
    except Exception as e:
        raise ImportError(
            "ultralytics 已安装但导入失败，请根据下方原始错误排查（常见为 torch 版本/CUDA DLL）。\n"
            f"  {type(e).__name__}: {e}\n"
            "可尝试: python -m pip install -U ultralytics torch"
        ) from e
    model = _YOLO(weights)
    model.to(str(device))
    names = model.model.names if hasattr(model.model, "names") else {}
    coco_name_to_id = {str(v): int(k) for k, v in names.items()} if isinstance(names, dict) else {}
    COCO_TO_VOC_ID.clear()
    for vid, vname in enumerate(VOC_CLASSES):
        if vname == "background":
            continue
        cname = VOC_TO_COCO.get(vname)
        if cname in coco_name_to_id:
            COCO_TO_VOC_ID[coco_name_to_id[cname]] = vid
    return model


@torch.no_grad()
def infer_yolo(
    model,
    img_01: torch.Tensor,
    *,
    imgsz: int = 640,
    use_tta: bool = False,
    predict_conf: float = 0.001,
    predict_iou: float = 0.7,
    augment: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    # 使用 HWC numpy 输入，让 Ultralytics 自动做 letterbox，避免 BCHW 尺寸必须 stride 对齐的报错。
    img_np = (img_01.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    dev_arg = img_01.device.index if img_01.is_cuda else "cpu"
    aug = bool(use_tta) if augment is None else bool(augment)
    res = model.predict(
        source=img_np,
        verbose=False,
        conf=float(predict_conf),
        iou=float(predict_iou),
        device=dev_arg,
        imgsz=int(imgsz),
        augment=aug,
    )[0]
    boxes_xyxy = res.boxes.xyxy if res.boxes is not None else torch.zeros((0, 4), device=img_01.device)
    scores = res.boxes.conf if res.boxes is not None else torch.zeros((0,), device=img_01.device)
    cls_ids = res.boxes.cls.long() if res.boxes is not None else torch.zeros((0,), dtype=torch.long, device=img_01.device)
    keep_idx: List[int] = []
    mapped: List[int] = []
    for i, c in enumerate(cls_ids.tolist()):
        if c in COCO_TO_VOC_ID:
            keep_idx.append(i)
            mapped.append(COCO_TO_VOC_ID[c])
    if len(keep_idx) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=img_01.device),
            "scores": torch.zeros((0,), dtype=torch.float32, device=img_01.device),
            "labels": torch.zeros((0,), dtype=torch.long, device=img_01.device),
        }
    keep_t = torch.tensor(keep_idx, dtype=torch.long, device=boxes_xyxy.device)
    return {
        "boxes": boxes_xyxy.index_select(0, keep_t).to(img_01.device),
        "scores": scores.index_select(0, keep_t).to(img_01.device),
        "labels": torch.tensor(mapped, dtype=torch.long, device=img_01.device),
    }


def yolo_whitebox_objective(
    model,
    img_01: torch.Tensor,
    topk: int = 300,
    target_conf: float = 0.05,
) -> torch.Tensor:
    """
    YOLO 白盒可微目标（越大代表检测越强，攻击时最小化该目标）：
    - 兼容 YOLOv8(4+nc) 与早期 YOLO(5+nc) 输出头
    - 仅聚焦高置信候选(top-k)，比全量均值更容易打掉有效检测
    - 引入阈值化项 ReLU(conf-target_conf)，更贴近“让检测掉到阈值以下”的攻击目标
    """
    # 白盒路径直接调用 model.model() 时，不会像 predict() 那样自动 letterbox。
    # 这里先把输入可微缩放到 stride(32) 对齐，避免 FPN concat 维度不一致报错。
    x = img_01.unsqueeze(0)
    _, _, h, w = x.shape
    stride = 32
    new_h = max(stride, ((int(h) + stride - 1) // stride) * stride)
    new_w = max(stride, ((int(w) + stride - 1) // stride) * stride)
    if new_h != int(h) or new_w != int(w):
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    out = model.model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, (list, tuple)):
        out = out[0]

    if not isinstance(out, torch.Tensor) or out.dim() != 3:
        return out.float().abs().mean()

    p = out
    # 统一为 [B, N, C]；Ultralytics 常见是 [B, C, N]
    if p.shape[1] <= 128 and p.shape[2] > 128:
        p = p.transpose(1, 2).contiguous()

    if p.shape[-1] < 6:
        return p.float().abs().mean()

    nc = getattr(model.model, "nc", None)
    has_obj = False
    if isinstance(nc, int):
        if p.shape[-1] == nc + 5:
            has_obj = True
        elif p.shape[-1] == nc + 4:
            has_obj = False
        else:
            # 回退：按 YOLOv8 头处理（4+nc）
            has_obj = False
    else:
        # 无法读取 nc 时的启发式：默认 YOLOv8 风格
        has_obj = False

    if has_obj:
        # 5+nc：obj 在第 5 维，类别从第 6 维开始
        obj = p[..., 4].sigmoid()
        cls_max = p[..., 5:].sigmoid().max(dim=-1).values
        conf = obj * cls_max
        logit_term = p[..., 4]
    else:
        # 4+nc（YOLOv8）：无显式 obj，类别从第 5 维开始
        cls_prob = p[..., 4:].sigmoid()
        conf = cls_prob.max(dim=-1).values
        # 用最大类别 logit 强化梯度（避免仅在高置信区梯度变弱）
        logit_term = p[..., 4:].max(dim=-1).values

    flat_conf = conf.reshape(-1)
    flat_logit = logit_term.reshape(-1)
    if flat_conf.numel() == 0:
        return p.float().abs().mean()

    k = min(int(topk), int(flat_conf.numel()))
    top_idx = torch.topk(flat_conf, k=k, largest=True).indices
    top_conf = flat_conf.index_select(0, top_idx)
    top_logit = flat_logit.index_select(0, top_idx)

    # 联合目标：
    # 1) 阈值化置信度项：优先把高分候选压到 target_conf 以下
    # 2) 置信度均值项：整体削弱检测响应
    # 3) logit 项：增强梯度稳定性，避免仅在概率空间更新变慢
    conf_margin = F.relu(top_conf - float(target_conf))
    return conf_margin.mean() + 0.5 * top_conf.mean() + 0.1 * top_logit.mean()


def voc_target_to_boxes_and_labels(target: dict) -> Tuple[torch.Tensor, List[str]]:
    anno = target["annotation"]
    objs = anno.get("object", [])
    if isinstance(objs, dict):
        objs = [objs]

    boxes: List[List[float]] = []
    labels: List[str] = []
    for obj in objs:
        name = obj["name"]
        bnd = obj["bndbox"]
        xmin = float(bnd["xmin"])
        ymin = float(bnd["ymin"])
        xmax = float(bnd["xmax"])
        ymax = float(bnd["ymax"])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)

    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), []
    return torch.tensor(boxes, dtype=torch.float32), labels


def compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def filter_pred(pred: Dict[str, torch.Tensor], conf_thresh: float) -> Dict[str, torch.Tensor]:
    keep = pred["scores"] >= conf_thresh
    return {
        "boxes": pred["boxes"][keep],
        "scores": pred["scores"][keep],
        "labels": pred["labels"][keep],
    }


def _sort_pred_by_score_desc(fp: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """按置信度降序排列预测，保证 ref_topk 截断取到最高分的一批。"""
    n = int(fp["boxes"].shape[0])
    if n <= 1:
        return fp
    order = torch.argsort(fp["scores"], descending=True)
    return {
        "boxes": fp["boxes"][order],
        "scores": fp["scores"][order],
        "labels": fp["labels"][order],
    }


def build_blackbox_attack_refs(
    fp_ref: Dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_labels: List[str],
    device: torch.device,
    ref_topk: int,
    mode: str,
    *,
    hybrid_suppress_iou: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构造黑盒攻击用的参考框（与 pred 同设备、xyxy）。

    - pred：沿用「干净高分框」作参考（旧行为）。
    - gt：仅用 VOC GT 框 + 类别，score 置 1，使 yolo_matched_suppression_objective
      直接压「与 GT 同类且 IoU 重叠」的预测，更贴近 GT-ASR 评估。
    - hybrid：GT 全部保留，再并入未被任一 GT（同类+IoU）解释的高分预测，兼顾漏检与误检。
    """
    fp_ref = _sort_pred_by_score_desc(fp_ref)
    mode = (mode or "gt").strip().lower()
    k = max(1, int(ref_topk))

    def _pred_slice() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(fp_ref["boxes"].shape[0])
        if n == 0:
            return (
                torch.zeros((0, 4), device=device, dtype=torch.float32),
                torch.zeros((0,), device=device, dtype=torch.long),
                torch.zeros((0,), device=device, dtype=torch.float32),
            )
        kk = min(k, n)
        return (
            fp_ref["boxes"][:kk].to(device=device, dtype=torch.float32),
            fp_ref["labels"][:kk].to(device=device, dtype=torch.long),
            fp_ref["scores"][:kk].to(device=device, dtype=torch.float32),
        )

    if mode == "pred":
        return _pred_slice()

    name_to_id = {n: i for i, n in enumerate(VOC_CLASSES)}
    gb_list: List[List[float]] = []
    gl_list: List[int] = []
    if gt_boxes.numel() > 0 and gt_labels:
        n_gt = int(gt_boxes.shape[0])
        for i in range(n_gt):
            if i >= len(gt_labels):
                break
            name = gt_labels[i]
            if name not in name_to_id:
                continue
            lid = int(name_to_id[name])
            if lid <= 0:
                continue
            gb_list.append(gt_boxes[i].tolist())
            gl_list.append(lid)

    if mode == "gt":
        if not gb_list:
            return _pred_slice()
        rb = torch.tensor(gb_list, device=device, dtype=torch.float32)
        rl = torch.tensor(gl_list, device=device, dtype=torch.long)
        rs = torch.ones((rb.shape[0],), device=device, dtype=torch.float32)
        return rb, rl, rs

    if mode != "hybrid":
        return _pred_slice()

    if not gb_list:
        return _pred_slice()
    rb = torch.tensor(gb_list, device=device, dtype=torch.float32)
    rl = torch.tensor(gl_list, device=device, dtype=torch.long)
    rs = torch.ones((rb.shape[0],), device=device, dtype=torch.float32)

    extra_b: List[torch.Tensor] = []
    extra_l: List[torch.Tensor] = []
    extra_s: List[torch.Tensor] = []
    npr = int(fp_ref["boxes"].shape[0])
    for j in range(npr):
        if rb.shape[0] + len(extra_b) >= k:
            break
        pb = fp_ref["boxes"][j]
        pl = int(fp_ref["labels"][j].item())
        pbl = pb.detach().cpu().tolist()
        skip = False
        for bi in range(rb.shape[0]):
            if int(rl[bi].item()) != pl:
                continue
            if compute_iou(pbl, rb[bi].detach().cpu().tolist()) >= float(hybrid_suppress_iou):
                skip = True
                break
        if not skip:
            extra_b.append(fp_ref["boxes"][j : j + 1].to(device=device, dtype=torch.float32))
            extra_l.append(fp_ref["labels"][j : j + 1].to(device=device, dtype=torch.long))
            extra_s.append(fp_ref["scores"][j : j + 1].to(device=device, dtype=torch.float32))
    if not extra_b:
        return rb, rl, rs
    eb = torch.cat(extra_b, dim=0)
    el = torch.cat(extra_l, dim=0)
    es = torch.cat(extra_s, dim=0)
    rem = k - rb.shape[0]
    if rem <= 0:
        return rb, rl, rs
    eb = eb[:rem]
    el = el[:rem]
    es = es[:rem]
    return (
        torch.cat([rb, eb], dim=0),
        torch.cat([rl, el], dim=0),
        torch.cat([rs, es], dim=0),
    )


def make_pseudo_targets_from_pred(
    pred: Dict[str, torch.Tensor],
    conf_thresh: float = 0.5,
    topk: int = 30,
) -> Dict[str, torch.Tensor]:
    fp = filter_pred(pred, conf_thresh=conf_thresh)
    if fp["boxes"].numel() == 0:
        # torchvision detection loss 需要至少 1 个 box，否则可能报错；
        # 这里返回一个“空目标”占位（label=1、box=0）以避免崩溃。
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=pred["boxes"].device, dtype=pred["boxes"].dtype)
        labels = torch.ones((1,), device=pred["labels"].device, dtype=pred["labels"].dtype)
    else:
        k = min(topk, fp["boxes"].shape[0])
        boxes = fp["boxes"][:k]
        labels = fp["labels"][:k]

    return {"boxes": boxes.detach(), "labels": labels.detach()}


def detection_confidence_objective(
    pred: Dict[str, torch.Tensor],
    conf_thresh: float = 0.0,
    topk: int = 50,
    eval_conf: Optional[float] = None,
) -> torch.Tensor:
    """
    黑盒攻击的目标函数（越小越好）：top-k 置信度和。
    - 仅依赖输出 scores，不需要梯度。
    - eval_conf 非空时叠加 margin 项，与评估阈值对齐。
    """
    scores = pred["scores"]
    if conf_thresh > 0:
        scores = scores[scores >= conf_thresh]
    if scores.numel() == 0:
        return torch.tensor(0.0, device=pred["boxes"].device, dtype=torch.float32)
    k = min(topk, scores.numel())
    top_scores = torch.topk(scores, k=k).values
    if eval_conf is None:
        return top_scores.sum()
    ec = float(eval_conf)
    margin = F.relu(top_scores - ec).sum()
    return margin + 0.02 * top_scores.sum()


def yolo_matched_suppression_objective(
    cur_pred: Dict[str, torch.Tensor],
    ref_boxes: torch.Tensor,
    ref_labels: torch.Tensor,
    ref_scores: torch.Tensor,
    topk: int,
    iou_match: float,
    eval_conf: Optional[float] = None,
) -> torch.Tensor:
    """
    YOLO/VOC 检测黑盒目标（越小越好）：在干净参考框上，压制当前图中同类 IoU 匹配的高分预测。
    无参考框时退化为 top-k 置信度和。

    eval_conf：与评估 filter_pred(conf_thresh=eval_conf) 对齐时，对每个 ref 使用
    rs * ReLU(best_match_score - eval_conf) + 小系数 * rs * best，使零阶优化直接压低
    「超过检测阈值的匹配框」，缓解「目标已很小但 ASR 不涨」的错位。
    """
    if ref_boxes.numel() == 0:
        return detection_confidence_objective(cur_pred, topk=topk, eval_conf=eval_conf)
    cur_boxes = cur_pred["boxes"]
    cur_labels = cur_pred["labels"]
    cur_scores = cur_pred["scores"]
    if cur_boxes.numel() == 0:
        return torch.tensor(0.0, device=ref_boxes.device, dtype=torch.float32)

    total = torch.tensor(0.0, device=ref_boxes.device, dtype=torch.float32)
    k = min(int(topk), int(ref_boxes.shape[0]))
    floor = 0.06
    for i in range(k):
        rb = ref_boxes[i]
        rl = ref_labels[i]
        rs = ref_scores[i]
        best = torch.tensor(0.0, device=ref_boxes.device, dtype=torch.float32)
        for cb, cl, cs in zip(cur_boxes, cur_labels, cur_scores):
            if int(cl.item()) != int(rl.item()):
                continue
            iou = compute_iou(rb.tolist(), cb.tolist())
            if iou >= float(iou_match):
                best = torch.maximum(best, cs)
        if eval_conf is None:
            total = total + rs * best
        else:
            ec = float(eval_conf)
            total = total + rs * (F.relu(best - ec) + floor * best)
    return total


def yolo_spurious_promotion_objective(
    cur_pred: Dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_label_ids: torch.Tensor,
    iou_match: float,
    topk_spur: int,
) -> torch.Tensor:
    """
    促「假阳性 / 多检」向的代理（越小越好）：等价于推高不与任一 GT 构成 TP 的检测框分数。

    TP 判定：与评估一致思路——按预测分数从高到低贪心匹配，预测与某 GT 同类且 IoU≥iou_match
    则记为该 GT 的 TP（每 GT 至多一框、每框至多一 GT）。未被匹配的预测为「伪框候选」，
    对其 top-k 分数取负权和，使整体目标最小化时倾向于拉高这些框的置信度。
    """
    cur_boxes = cur_pred["boxes"]
    cur_labels = cur_pred["labels"]
    cur_scores = cur_pred["scores"]
    device = cur_boxes.device
    n_p = int(cur_boxes.shape[0])
    n_g = int(gt_boxes.shape[0])
    if n_p == 0 or n_g == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    order = torch.argsort(cur_scores, descending=True).tolist()
    gt_used = [False] * n_g
    pred_is_tp = [False] * n_p
    for pi in order:
        pb = cur_boxes[int(pi)].detach().cpu().tolist()
        pl = int(cur_labels[int(pi)].item())
        for gi in range(n_g):
            if gt_used[gi]:
                continue
            if int(gt_label_ids[gi].item()) != pl:
                continue
            gb = gt_boxes[int(gi)].detach().cpu().tolist()
            if compute_iou(gb, pb) >= float(iou_match):
                pred_is_tp[int(pi)] = True
                gt_used[gi] = True
                break

    spur: List[torch.Tensor] = []
    for pi in range(n_p):
        if not pred_is_tp[pi]:
            spur.append(cur_scores[pi])
    if not spur:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    ss = torch.stack(spur, dim=0)
    k = min(max(1, int(topk_spur)), int(ss.shape[0]))
    topv = torch.topk(ss, k=k).values
    return -topv.sum()


def yolo_wrongclass_at_gt_objective(
    cur_pred: Dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_label_ids: torch.Tensor,
    iou_match: float,
) -> torch.Tensor:
    """
    促「框仍在但类错」向的代理（越小越好）：对每个 GT，在 IoU≥iou_match 的预测中取**错误类**上的
    最高分数，累加其负值；最小化该和等价于拉高与 GT 重叠但类别不一致的检测置信度。
    """
    cur_boxes = cur_pred["boxes"]
    cur_labels = cur_pred["labels"]
    cur_scores = cur_pred["scores"]
    device = cur_boxes.device
    if gt_boxes.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    total = torch.tensor(0.0, device=device, dtype=torch.float32)
    n_g = int(gt_boxes.shape[0])
    n_p = int(cur_boxes.shape[0])
    for gi in range(n_g):
        gb = gt_boxes[int(gi)].detach().cpu().tolist()
        gl = int(gt_label_ids[int(gi)].item())
        best_wrong = torch.tensor(0.0, device=device, dtype=torch.float32)
        for pi in range(n_p):
            pl = int(cur_labels[int(pi)].item())
            if pl == gl:
                continue
            pb = cur_boxes[int(pi)].detach().cpu().tolist()
            if compute_iou(gb, pb) >= float(iou_match):
                best_wrong = torch.maximum(best_wrong, cur_scores[int(pi)])
        total = total - best_wrong
    return total


def project_linf_01(x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    """将 x 投影到以 x0 为中心、L∞ 半径 eps 的球上，并裁剪到 [0,1]。"""
    x = torch.max(torch.min(x, x0 + float(eps)), x0 - float(eps))
    return ensure_tensor_01(x)


def spatial_mask_from_boxes_xyxy(
    boxes_xyxy: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    expand_frac: float = 0.2,
) -> torch.Tensor:
    """
    构造 [1,H,W] 的 0/1 掩膜：所有框并集并按框尺度外扩（检测版「目标区域」集中扰动）。

    boxes_xyxy: [N,4] xyxy，须与 [C,H,W] 图像的 H,W 同一坐标系（VOC + ToTensor 下为像素级）。
    expand_frac: 每边外扩 max(w,h)*expand_frac。N=0 或有效面积为 0 时退化为全 1。
    """
    h_i, w_i = int(height), int(width)
    mask = torch.zeros((1, h_i, w_i), device=device, dtype=dtype)
    if boxes_xyxy.numel() == 0:
        mask.fill_(1.0)
        return mask
    n = int(boxes_xyxy.shape[0])
    ef = float(expand_frac)
    for i in range(n):
        x1, y1, x2, y2 = boxes_xyxy[i].detach().tolist()
        bw = max(1.0, float(x2) - float(x1))
        bh = max(1.0, float(y2) - float(y1))
        pad = ef * max(bw, bh)
        xa = max(0.0, float(x1) - pad)
        ya = max(0.0, float(y1) - pad)
        xb = min(float(w_i - 1), float(x2) + pad)
        yb = min(float(h_i - 1), float(y2) + pad)
        xi1, yi1 = int(round(xa)), int(round(ya))
        xi2, yi2 = int(round(xb)), int(round(yb))
        xi2 = max(xi1, min(w_i - 1, xi2))
        yi2 = max(yi1, min(h_i - 1, yi2))
        mask[:, yi1 : yi2 + 1, xi1 : xi2 + 1] = 1.0
    if float(mask.sum()) < 1.0:
        mask.fill_(1.0)
    return mask


def compute_detection_loss_whitebox(
    model: torch.nn.Module,
    img: torch.Tensor,
    pseudo_targets: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    计算 detection loss，用于白盒攻击（最大化 loss）。
    注意：需要 model.train() 才会返回 loss dict。
    """
    losses: Dict[str, torch.Tensor] = model([img], [pseudo_targets])  # type: ignore[assignment]
    total = sum(v for v in losses.values())
    return total


def save_detection_viz(
    img_tensor_01: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    coco_classes: List[str],
    out_path: str,
    conf_thresh: float = 0.5,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fp = filter_pred(pred, conf_thresh=conf_thresh)
    img_pil = transforms.ToPILImage()(img_tensor_01.detach().cpu())
    draw = ImageDraw.Draw(img_pil)
    for box, score, label in zip(fp["boxes"].cpu(), fp["scores"].cpu(), fp["labels"].cpu()):
        x1, y1, x2, y2 = box.tolist()
        name = coco_classes[int(label)] if int(label) < len(coco_classes) else "unknown"
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        draw.text((x1, y1), f"{name} {float(score):.2f}", fill="yellow")
    img_pil.save(out_path)


def ensure_tensor_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def input_diversity(
    x: torch.Tensor,
    prob: float = 0.7,
    scale_min: float = 0.9,
) -> torch.Tensor:
    """
    DI: 随机缩放+填充，保持可微并回到原尺寸。
    """
    if prob <= 0.0 or float(torch.rand(1).item()) > float(prob):
        return x
    c, h, w = x.shape
    s = float(scale_min) + (1.0 - float(scale_min)) * float(torch.rand(1).item())
    nh = max(32, int(round(h * s)))
    nw = max(32, int(round(w * s)))
    xr = F.interpolate(x.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)
    pad_h = max(0, h - nh)
    pad_w = max(0, w - nw)
    top = int(torch.randint(0, pad_h + 1, (1,)).item()) if pad_h > 0 else 0
    left = int(torch.randint(0, pad_w + 1, (1,)).item()) if pad_w > 0 else 0
    bottom = pad_h - top
    right = pad_w - left
    return F.pad(xr, (left, right, top, bottom), value=0.0)


def ti_smooth_grad(grad: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    TI: 对梯度做高斯平滑，提升平移鲁棒性与攻击稳定性。
    """
    k = int(max(3, kernel_size))
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k, device=grad.device, dtype=grad.dtype) - (k - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2.0 * float(sigma) * float(sigma)))
    g1 = g1 / g1.sum().clamp_min(1e-12)
    g2 = (g1[:, None] * g1[None, :]).to(grad.dtype)
    ker = g2.view(1, 1, k, k).repeat(grad.shape[0], 1, 1, 1)
    gs = F.conv2d(grad.unsqueeze(0), ker, padding=k // 2, groups=grad.shape[0]).squeeze(0)
    return gs


@torch.no_grad()
def infer_one(model: torch.nn.Module, img: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    return model([img])[0]


@dataclass(frozen=True)
class SimpleMetrics:
    total_gt: int
    matched_gt: int
    num_imgs_with_obj: int
    num_imgs_detected: int

    @property
    def recall(self) -> float:
        return 0.0 if self.total_gt <= 0 else (self.matched_gt / self.total_gt)

    @property
    def img_acc(self) -> float:
        return 0.0 if self.num_imgs_with_obj <= 0 else (self.num_imgs_detected / self.num_imgs_with_obj)


@dataclass(frozen=True)
class AttackSuccess:
    gt_success: float
    img_success: float
    gt_success_count: int
    gt_detected_clean: int
    img_success_count: int
    img_detected_clean: int


def _match_gt_flags(
    gt_boxes: torch.Tensor,
    gt_labels: List[str],
    pred_boxes: torch.Tensor,
    pred_names: List[str],
    coco_classes: List[str],
    iou_thresh: float,
) -> Tuple[List[bool], bool]:
    """
    对每个 GT（只统计 COCO 中存在的类别）返回是否被命中，同时返回“图像是否至少命中一个GT”。
    """
    flags: List[bool] = []
    img_has_match = False
    for g_box, g_name in zip(gt_boxes, gt_labels):
        if g_name not in coco_classes:
            continue
        g_box_list = g_box.tolist()
        matched = False
        for p_box, p_name in zip(pred_boxes, pred_names):
            if p_name != g_name:
                continue
            if compute_iou(g_box_list, p_box.tolist()) >= iou_thresh:
                matched = True
                break
        flags.append(matched)
        if matched:
            img_has_match = True
    return flags, img_has_match


def evaluate_batch(
    *,
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    coco_classes: List[str],
    num_eval: int = 200,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    input_transform=None,
    attack_fn=None,
    save_dir: str | None = None,
    save_vis_n: int = 0,
    save_prefix: str = "",
) -> Tuple[SimpleMetrics, SimpleMetrics, AttackSuccess | None]:
    """
    批量评估：
    - clean 指标：对原图（或 input_transform 后）做推理
    - pert 指标：若提供 attack_fn，则对 attack 后图像推理；否则与 clean 相同
    - attack success：只在提供 attack_fn 时返回

    指标定义与现有 base/noise 脚本保持一致：
    - matched_gt / total_gt（简单召回率）
    - 图像级检测率：含GT图片中，至少命中1个GT 的比例

    攻击成功率（两种口径）：
    - gt_success：在 clean 下“能命中的 GT”中，被攻击后变为“不命中”的比例
    - img_success：在 clean 下“图像至少命中1个GT”的图片中，被攻击后变为“0命中”的比例
    """
    n = min(int(num_eval), len(dataset))
    clean_total_gt = 0
    clean_matched_gt = 0
    clean_imgs_with_obj = 0
    clean_imgs_detected = 0

    pert_total_gt = 0
    pert_matched_gt = 0
    pert_imgs_with_obj = 0
    pert_imgs_detected = 0

    gt_detected_clean = 0
    gt_success_count = 0
    img_detected_clean = 0
    img_success_count = 0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for idx in range(n):
        img_tensor, target = dataset[idx]
        if input_transform is not None:
            img_tensor = input_transform(img_tensor)
        img_tensor = img_tensor.to(device)

        gt_boxes, gt_labels = voc_target_to_boxes_and_labels(target)
        total_gt_here = int(gt_boxes.size(0))

        # --- clean pred ---
        with torch.no_grad():
            pred_clean = infer_one(model, img_tensor)
        fp_c = filter_pred(pred_clean, conf_thresh=conf_thresh)
        pred_names_c = [
            coco_classes[int(label)] if int(label) < len(coco_classes) else "unknown"
            for label in fp_c["labels"].detach().cpu()
        ]
        flags_clean, img_has_match_clean = _match_gt_flags(
            gt_boxes, gt_labels, fp_c["boxes"].detach().cpu(), pred_names_c, coco_classes, iou_thresh
        )

        clean_total_gt += total_gt_here
        if total_gt_here > 0:
            clean_imgs_with_obj += 1
        clean_matched_gt += int(sum(flags_clean))
        if total_gt_here > 0 and img_has_match_clean:
            clean_imgs_detected += 1

        # --- pert pred ---
        if attack_fn is None:
            adv_tensor = img_tensor
        else:
            adv_tensor = attack_fn(img_tensor, pred_clean)  # pred_clean 可用于伪标注/加速
            adv_tensor = ensure_tensor_01(adv_tensor).to(device)

        with torch.no_grad():
            pred_pert = infer_one(model, adv_tensor)
        fp_p = filter_pred(pred_pert, conf_thresh=conf_thresh)
        pred_names_p = [
            coco_classes[int(label)] if int(label) < len(coco_classes) else "unknown"
            for label in fp_p["labels"].detach().cpu()
        ]
        flags_pert, img_has_match_pert = _match_gt_flags(
            gt_boxes, gt_labels, fp_p["boxes"].detach().cpu(), pred_names_p, coco_classes, iou_thresh
        )

        pert_total_gt += total_gt_here
        if total_gt_here > 0:
            pert_imgs_with_obj += 1
        pert_matched_gt += int(sum(flags_pert))
        if total_gt_here > 0 and img_has_match_pert:
            pert_imgs_detected += 1

        # --- attack success ---
        if attack_fn is not None:
            # flags_* 是“只统计 COCO 类别 GT”的序列，两边长度一致
            for c_ok, p_ok in zip(flags_clean, flags_pert):
                if c_ok:
                    gt_detected_clean += 1
                    if not p_ok:
                        gt_success_count += 1
            if total_gt_here > 0 and img_has_match_clean:
                img_detected_clean += 1
                if not img_has_match_pert:
                    img_success_count += 1

        # --- save viz ---
        if save_dir is not None and idx < int(save_vis_n):
            save_detection_viz(
                img_tensor.detach().cpu(),
                pred_clean,
                coco_classes,
                os.path.join(save_dir, f"{save_prefix}clean_{idx:04d}.png"),
                conf_thresh=conf_thresh,
            )
            save_detection_viz(
                adv_tensor.detach().cpu(),
                pred_pert,
                coco_classes,
                os.path.join(save_dir, f"{save_prefix}pert_{idx:04d}.png"),
                conf_thresh=conf_thresh,
            )

    clean_m = SimpleMetrics(
        total_gt=clean_total_gt,
        matched_gt=clean_matched_gt,
        num_imgs_with_obj=clean_imgs_with_obj,
        num_imgs_detected=clean_imgs_detected,
    )
    pert_m = SimpleMetrics(
        total_gt=pert_total_gt,
        matched_gt=pert_matched_gt,
        num_imgs_with_obj=pert_imgs_with_obj,
        num_imgs_detected=pert_imgs_detected,
    )

    atk: AttackSuccess | None
    if attack_fn is None:
        atk = None
    else:
        gt_success = 0.0 if gt_detected_clean <= 0 else (gt_success_count / gt_detected_clean)
        img_success = 0.0 if img_detected_clean <= 0 else (img_success_count / img_detected_clean)
        atk = AttackSuccess(
            gt_success=gt_success,
            img_success=img_success,
            gt_success_count=gt_success_count,
            gt_detected_clean=gt_detected_clean,
            img_success_count=img_success_count,
            img_detected_clean=img_detected_clean,
        )

    return clean_m, pert_m, atk

