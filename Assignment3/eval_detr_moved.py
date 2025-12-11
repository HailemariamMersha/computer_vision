import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from transformers import DetrForObjectDetection, DetrImageProcessor

from config import Config
from dataset import MovedObjectDataset, collate_fn_with_processor


def compute_pr(outputs, targets, processor, device, score_thresh=0.5, iou_thresh=0.5):
    target_sizes = torch.tensor([t["size"] for t in targets], device=device)
    processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=score_thresh)
    tp = fp = fn = 0
    all_matches: List[List[Tuple[int, int]]] = []

    for preds, tgt in zip(processed, targets):
        keep = preds["scores"] >= score_thresh
        pred_boxes = preds["boxes"][keep].to(device)
        pred_labels = preds["labels"][keep].to(device)
        gt_boxes = tgt["boxes"].to(device)
        gt_labels = tgt["class_labels"].to(device)

        matches: List[Tuple[int, int]] = []
        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            all_matches.append(matches)
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched = set()
        for i in range(len(pred_boxes)):
            gi = torch.argmax(ious[i]).item()
            if gi in matched:
                continue
            if ious[i, gi] >= iou_thresh and pred_labels[i] == gt_labels[gi]:
                tp += 1
                matched.add(gi)
                matches.append((gi, i))
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)
        all_matches.append(matches)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall, tp, fp, fn, all_matches


def _color_from_index(idx: int) -> Tuple[int, int, int]:
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

    for pred_idx, ((x, y, w, h), score) in enumerate(zip(pred_boxes, pred_scores)):
        if score < score_thresh:
            continue
        if pred_idx in matched_pred_to_gt:
            color = _color_from_index(matched_pred_to_gt[pred_idx])
            label = f"P{pred_idx} {score:.2f}"
        else:
            color = (0, 0, 255)
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

    if img1.shape[0] != img2.shape[0]:
        scale = img2.shape[0] / img1.shape[0]
        img1 = cv2.resize(img1, (int(img1.shape[1] * scale), img2.shape[0]))
    vis = cv2.hconcat([img1, img2])

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
    parser = argparse.ArgumentParser(description="Evaluate DETR moved-object model (processor-based, no COCO).")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(script_dir / "outputs/checkpoints/detr_option2_all_best.pth"),
        help="Path to checkpoint saved by train.py",
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
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DetrImageProcessor.from_pretrained(
        args.model_name, size={"longest_edge": max(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)}
    )
    model = DetrForObjectDetection.from_pretrained(
        args.model_name,
        num_labels=cfg.NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)

    dataset = MovedObjectDataset(cfg, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_processor(processor),
    )

    total_tp = total_fp = total_fn = 0
    vis_count = 0

    for batch in loader:
        if batch is None:
            continue
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["orig_targets"]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        prec, rec, tp, fp, fn, matches = compute_pr(
            outputs, labels, processor, device, args.score_thresh, args.iou_thresh
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        processed = processor.post_process_object_detection(
            outputs, target_sizes=batch["orig_target_sizes"].to(device), threshold=args.score_thresh
        )

        for i, res in enumerate(processed):
            gt = labels[i]
            gt_boxes = gt["boxes"].tolist()
            pred_boxes = res["boxes"].tolist()
            pred_scores = res["scores"].tolist()
            pred_labels = res["labels"].tolist()

            if vis_count < args.max_vis:
                meta = gt.get("meta", {})
                img1_path = Path(meta.get("img1_path", ""))
                img2_path = Path(meta.get("img2_path", ""))
                vis_path = Path(args.vis_dir) / f"vis_{vis_count:03d}.png"
                draw_side_by_side(
                    img1_path,
                    img2_path,
                    gt_boxes,
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    matches[i] if i < len(matches) else [],
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
                        f"split: {args.split}",
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
