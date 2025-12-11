import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from transformers import DetrImageProcessor

from config import Config
from dataset import MovedObjectDataset
from model import create_model


def draw_boxes(
    img,
    boxes: List[Tuple[float, float, float, float]],
    labels: List[int],
    color,
    prefix="",
    scores: List[float] | None = None,
):
    for i, (box, lab) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f"{prefix}{i}:{lab}"
        if scores is not None:
            txt = f"{txt} {scores[i]:.2f}"
        cv2.putText(
            img,
            txt,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize DETR predictions on frame pairs (processor-based).")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/detr_option2_all_best.pth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.DEVICE)

    dataset = MovedObjectDataset(cfg, split=args.split)
    if len(dataset) == 0:
        print("Dataset is empty; ensure matched annotations exist.")
        return

    processor = DetrImageProcessor.from_pretrained(
        cfg.MODEL_NAME, size={"longest_edge": max(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)}
    )
    model = create_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use first N samples
    indices = list(range(min(args.num_samples, len(dataset))))
    for idx in indices:
        diff, target = dataset[idx]
        meta = target.get("meta", {})

        img2_path = meta.get("img2_path")
        if not img2_path:
            continue
        img2 = cv2.imread(img2_path)
        if img2 is None:
            continue

        with torch.no_grad():
            enc = processor(images=[diff], return_tensors="pt")
            outputs = model(pixel_values=enc["pixel_values"].to(device))
            processed = processor.post_process_object_detection(
                outputs, target_sizes=[target["size"]], threshold=args.score_thresh
            )
        preds = processed[0]
        pred_boxes = preds["boxes"].tolist()
        pred_scores = preds["scores"].tolist()
        pred_labels = preds["labels"].tolist()

        gt_boxes = target["boxes"].tolist()
        gt_labels = target["class_labels"].tolist()

        # Two views of frame2: left with GT (green), right with predictions (red + labels/scores)
        gt_vis = draw_boxes(img2.copy(), gt_boxes, gt_labels, (0, 255, 0), prefix="GT")
        pred_vis = draw_boxes(img2.copy(), pred_boxes, pred_labels, (0, 0, 255), prefix="P", scores=pred_scores)

        combined = cv2.hconcat([gt_vis, pred_vis])
        out_path = out_dir / f"frame2_gt_pred_{args.split}_{idx:03d}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
