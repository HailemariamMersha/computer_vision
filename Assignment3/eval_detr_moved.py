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
    matches: List[Tuple[int, int]] = []  # (gt_idx, pred_idx)
    for pred_idx, (pb, pl) in enumerate(zip(pred_boxes, pred_labels)):
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
            matches.append((best_idx, pred_idx))
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matches


def _color_from_index(idx: int) -> Tuple[int, int, int]:
    # Deterministic pseudo-random color derived from index (BGR for OpenCV).
    r = 60 + (37 * idx) % 190
    g = 60 + (91 * idx) % 190
    b = 60 + (53 * idx) % 190
    return int(b), int(g), int(r)


def draw_side_by_side(
    frame1_path: Path,
    frame2_path: Path,
    gt_boxes,
    pred_boxes,
    pred_scores,
    pred_labels,
    matches: List[Tuple[int, int]],
    out_path: Path,
    score_thresh=0.7,
):
    img1 = cv2.imread(str(frame1_path))
    img2 = cv2.imread(str(frame2_path))
    if img1 is None or img2 is None:
        return

    matched_pred_to_gt = {pred_idx: gt_idx for gt_idx, pred_idx in matches}
    matched_gt = {gt_idx for gt_idx, _ in matches}

    # Draw GT boxes on final frame
    for gt_idx, (x, y, w, h) in enumerate(gt_boxes):
        color = _color_from_index(gt_idx)
        thickness = 2 if gt_idx in matched_gt else 1
        cv2.rectangle(img2, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
        cv2.putText(
            img2,
            f"GT {gt_idx}",
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # Draw predictions on final frame
    for pred_idx, ((x, y, w, h), score) in enumerate(zip(pred_boxes, pred_scores)):
        if score < score_thresh:
            continue
        if pred_idx in matched_pred_to_gt:
            color = _color_from_index(matched_pred_to_gt[pred_idx])
            label = f"P{pred_idx} {score:.2f}"
        else:
            color = (0, 0, 255)  # red for unmatched predictions
            label = f"P{pred_idx}* {score:.2f}"
        cv2.rectangle(img2, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(
            img2,
            label,
            (int(x), int(y) + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # Concatenate initial/final frames
    if img1.shape[0] != img2.shape[0]:
        scale = img2.shape[0] / img1.shape[0]
        img1 = cv2.resize(img1, (int(img1.shape[1] * scale), img2.shape[0]))
    vis = cv2.hconcat([img1, img2])

    # Titles
    cv2.putText(vis, "Initial", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(
        vis,
        "Final (GT+Pred)",
        (img1.shape[1] + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


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
    parser.add_argument("--max_vis", type=int, default=30)
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=str(script_dir / "outputs/eval/metrics.txt"),
        help="Where to write precision/recall summary.",
    )
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

            tp, fp, fn, matches = match_predictions(
                gt_boxes, gt_labels, pred_boxes, pred_labels, args.iou_thresh
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn

            if vis_count < args.max_vis:
                img_info = dataset.images[batch_idx * loader.batch_size + i]
                frame1_name = img_info.get("file_name_frame1", img_info["file_name"])
                frame2_name = img_info.get("file_name_frame2", img_info["file_name"])
                img1_path = Path(args.images_root) / frame1_name
                img2_path = Path(args.images_root) / frame2_name
                vis_path = Path(args.vis_dir) / f"vis_{vis_count:03d}.png"
                draw_side_by_side(
                    img1_path,
                    img2_path,
                    gt_boxes,
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    matches,
                    vis_path,
                    score_thresh=args.score_thresh,
                )
                vis_count += 1

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}")
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as f:
            f.write(
                "\n".join(
                    [
                        f"checkpoint: {args.ckpt}",
                        f"test_json: {args.test_json}",
                        f"score_thresh: {args.score_thresh}",
                        f"iou_thresh: {args.iou_thresh}",
                        f"precision: {precision:.6f}",
                        f"recall: {recall:.6f}",
                    ]
                )
            )
        print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
