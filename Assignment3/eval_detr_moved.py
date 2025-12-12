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


def parse_match_file(path: Path) -> Tuple[List[List[float]], List[List[float]]]:
    init_boxes: List[List[float]] = []
    final_boxes: List[List[float]] = []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        init_parts = lines[i].split()
        final_parts = lines[i + 1].split()
        if len(init_parts) < 6 or len(final_parts) < 6:
            continue
        _, x0, y0, w0, h0, _ = init_parts[:6]
        _, x1, y1, w1, h1, _ = final_parts[:6]
        init_boxes.append([float(x0), float(y0), float(x0) + float(w0), float(y0) + float(h0)])
        final_boxes.append([float(x1), float(y1), float(x1) + float(w1), float(y1) + float(h1)])
    return init_boxes, final_boxes


def draw_four_panel(frame1_path: Path, frame2_path: Path, gt_init, gt_final, pred_boxes, pred_scores, out_path: Path):
    img1_gt = cv2.imread(str(frame1_path))
    img2_gt = cv2.imread(str(frame2_path))
    img1_pred = cv2.imread(str(frame1_path))
    img2_pred = cv2.imread(str(frame2_path))
    if any(x is None for x in [img1_gt, img2_gt, img1_pred, img2_pred]):
        return

    # GT: initial (green) on initial frame, final (green) on final frame
    for i, (x1, y1, x2, y2) in enumerate(gt_init):
        cv2.rectangle(img1_gt, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img1_gt, f"GT{i}", (int(x1), int(max(12, y1 - 4))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    for i, (x1, y1, x2, y2) in enumerate(gt_final):
        cv2.rectangle(img2_gt, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img2_gt, f"GT{i}", (int(x1), int(max(12, y1 - 4))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Predictions: draw on both initial and final (red)
    for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        x1, y1, x2, y2 = box
        for img in (img1_pred, img2_pred):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"P{i}:{score:.2f}",
                (int(x1), int(max(12, y1 - 4))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    top = cv2.hconcat([img1_gt, img2_gt])
    bot = cv2.hconcat([img1_pred, img2_pred])
    vis = cv2.vconcat([top, bot])
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
        args.model_name,
        size={
            "shortest_edge": min(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
            "longest_edge": max(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
        },
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
            if vis_count >= args.max_vis:
                break
            gt = labels[i]
            gt_boxes = gt["boxes"].tolist()
            pred_boxes = res["boxes"].tolist()
            pred_scores = res["scores"].tolist()

            meta = gt.get("meta", {})
            frame1_path = Path(meta.get("img1_path", ""))
            frame2_path = Path(meta.get("img2_path", ""))
            ann_path = Path(meta.get("ann_path", ""))
            if frame1_path and frame2_path and ann_path.exists():
                gt_init_boxes, gt_final_boxes = parse_match_file(ann_path)
                vis_path = Path(args.vis_dir) / f"vis_{vis_count:03d}.png"
                draw_four_panel(
                    frame1_path,
                    frame2_path,
                    gt_init_boxes,
                    gt_final_boxes,
                    pred_boxes,
                    pred_scores,
                    vis_path,
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
