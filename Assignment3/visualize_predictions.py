import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from transformers.image_transforms import center_to_corners_format

from config import Config
from dataset import MovedObjectDataset, collate_fn
from model import create_model


def to_xyxy_norm(pred_boxes: torch.Tensor) -> torch.Tensor:
    """Convert DETR cxcywh boxes (normalized) to xyxy (normalized)."""
    return center_to_corners_format(pred_boxes)


def denorm_boxes(boxes: torch.Tensor, width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """Scale normalized xyxy boxes to pixel coordinates."""
    scaled = []
    for x1, y1, x2, y2 in boxes:
        scaled.append(
            (
                int(x1.item() * width),
                int(y1.item() * height),
                int(x2.item() * width),
                int(y2.item() * height),
            )
        )
    return scaled


def draw_boxes(img, boxes, color, label_prefix=""):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label_prefix:
            cv2.putText(
                img,
                f"{label_prefix}{i}",
                (x1, max(10, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize DETR predictions side-by-side for frame pairs.")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/detr_option2_all_epoch19.pth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.DEVICE)

    # Dataset with access to original image paths
    dataset = MovedObjectDataset(cfg, split=args.split)
    if len(dataset) == 0:
        print("Dataset is empty; ensure matched annotations exist.")
        return

    # Load model + checkpoint
    model = create_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick first N samples
    indices = list(range(min(args.num_samples, len(dataset))))
    for idx in indices:
        diff, target = dataset[idx]
        sample = dataset.samples[idx]

        # Load original frames and resize for side-by-side display
        img1 = cv2.imread(sample["img1_path"])
        img2 = cv2.imread(sample["img2_path"])
        img1 = cv2.resize(img1, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
        img2 = cv2.resize(img2, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))

        # Predict
        with torch.no_grad():
            outputs = model(pixel_values=torch.stack([diff]).to(device))
        pred_boxes_xyxy_norm = to_xyxy_norm(outputs.pred_boxes[0].cpu())
        pred_scores = outputs.logits[0].softmax(-1)[:, :-1].max(-1)

        # Filter by score threshold
        keep = pred_scores.values >= args.score_thresh
        pred_boxes_xyxy_norm = pred_boxes_xyxy_norm[keep]

        # Convert boxes to pixel coords
        pred_boxes_px = denorm_boxes(pred_boxes_xyxy_norm, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)

        # Ground truth boxes (already normalized xyxy in target)
        gt_boxes_px = denorm_boxes(target["boxes"], cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)

        # Draw GT on left (green), predictions on right (red)
        img1_vis = draw_boxes(img1.copy(), gt_boxes_px, (0, 255, 0), label_prefix="GT")
        img2_vis = draw_boxes(img2.copy(), pred_boxes_px, (0, 0, 255), label_prefix="P")

        combined = cv2.hconcat([img1_vis, img2_vis])
        out_path = out_dir / f"pair_{args.split}_{idx:03d}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
