import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.image_transforms import center_to_corners_format

from config import Config
from moved_dataset import MovedObjectDetrDataset, detr_collate_fn


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh=0.5):
    matched_gt = set()
    tp = 0
    fp = 0
    for pb, pl in zip(pred_boxes, pred_labels):
        best_iou = 0.0
        best_idx = None
        for idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in matched_gt or gl != pl:
                continue
            iou = compute_iou(gb, pb)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_thresh and best_idx is not None:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def draw_boxes(image_path: Path, gt_boxes, pred_boxes, pred_scores, pred_labels, out_path: Path, score_thresh=0.7):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    for (x, y, w, h) in gt_boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    for (x, y, w, h), score in zip(pred_boxes, pred_scores):
        if score < score_thresh:
            continue
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.putText(img, f"{score:.2f}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def parse_args():
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Evaluate DETR moved-object model.")
    parser.add_argument("--images_root", type=str, default=str(script_dir / "cv_data_hw2"))
    parser.add_argument("--test_json", type=str, default=str(script_dir / "annotations/annotations_test.json"))
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(script_dir / "outputs/checkpoints/detr_option2_all_epoch19.pth"),
        help="Path to .pth checkpoint saved by train.py",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/detr-resnet-50",
        help="Base DETR model name to load weights/config from",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--vis_dir", type=str, default=str(script_dir / "results/vis"))
    parser.add_argument("--max_vis", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build processor and model from the base model name, then load .pth state dict
    processor = DetrImageProcessor.from_pretrained(args.model_name)
    model = DetrForObjectDetection.from_pretrained(
        args.model_name,
        num_labels=cfg.NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)

    dataset = MovedObjectDetrDataset(args.test_json, args.images_root, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detr_collate_fn,
    )

    total_tp = total_fp = total_fn = 0
    vis_count = 0

    for batch_idx, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]
        target_sizes = torch.stack([lab["orig_size"] for lab in labels]).to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=args.score_thresh
        )

        for i, res in enumerate(results):
            gt = labels[i]
            # Convert GT from normalized cxcywh to absolute xyxy
            gt_boxes_cxcywh = gt["boxes"]
            orig_h, orig_w = gt["orig_size"].tolist()
            gt_boxes_xyxy = center_to_corners_format(gt_boxes_cxcywh)
            gt_boxes_xyxy[:, 0::2] *= orig_w
            gt_boxes_xyxy[:, 1::2] *= orig_h
            gt_boxes = gt_boxes_xyxy.tolist()
            gt_labels = gt["class_labels"].tolist()
            pred_boxes = res["boxes"].tolist()
            pred_scores = res["scores"].tolist()
            pred_labels = res["labels"].tolist()

            tp, fp, fn = match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, args.iou_thresh)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            if vis_count < args.max_vis:
                img_info = dataset.images[batch_idx * loader.batch_size + i]
                img_path = Path(args.images_root) / img_info["file_name"]
                vis_path = Path(args.vis_dir) / f"vis_{vis_count:03d}.png"
                draw_boxes(img_path, gt_boxes, pred_boxes, pred_scores, pred_labels, vis_path, score_thresh=args.score_thresh)
                vis_count += 1

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
