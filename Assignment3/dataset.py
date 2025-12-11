import os
import random
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageChops

from config import Config


def parse_matched_annotation_file(path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Reads a matched annotation file where every two rows correspond to one matched object:
    first row is the old box (ignored), second row is the new box (used as GT).
    Returns boxes (x_min, y_min, x_max, y_max) and labels in pixel coords.
    """
    boxes: List[List[float]] = []
    labels: List[int] = []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        parts = lines[i + 1].split()
        if len(parts) < 6:
            continue
        _, x, y, w, h, cls_id = parts[:6]
        x, y, w, h = map(float, (x, y, w, h))
        cls_id = int(cls_id)
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(cls_id)

    return boxes, labels


def build_samples(config: Config) -> List[Dict[str, str]]:
    """
    Reads index.txt and maps each pair to its matched annotation path if it exists.
    """
    index_path = os.path.join(config.DATA_ROOT, "index.txt")
    samples: List[Dict[str, str]] = []
    with open(index_path, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != 4:
                continue
            img1_rel, ann1_rel, img2_rel, ann2_rel = parts
            img1_path = os.path.join(config.DATA_ROOT, img1_rel)
            img2_path = os.path.join(config.DATA_ROOT, img2_rel)
            ann2_name = os.path.basename(ann2_rel)
            folder_name = os.path.basename(os.path.dirname(img1_rel))
            candidates = [
                os.path.join(config.MATCHED_ANN_DIR, folder_name, ann2_name),
                os.path.join(
                    config.MATCHED_ANN_DIR,
                    f"{folder_name}-{os.path.splitext(os.path.basename(img1_rel))[0]}-{os.path.splitext(os.path.basename(img2_rel))[0]}_match.txt",
                ),
            ]
            matched_ann_path = next((p for p in candidates if os.path.exists(p)), None)
            if not matched_ann_path:
                continue
            samples.append(
                {
                    "img1_path": img1_path,
                    "img2_path": img2_path,
                    "ann_path": matched_ann_path,
                }
            )
    return samples


class MovedObjectDataset(Dataset):
    """
    Pixel-diff dataset using PIL + DetrImageProcessor downstream (no manual resizing).
    """

    def __init__(self, config: Config, split: str = "train"):
        self.config = config
        samples = build_samples(config)
        if not samples:
            raise ValueError("No matched annotation samples found. Run the labeller first.")

        random.seed(config.RANDOM_SEED)
        random.shuffle(samples)
        split_idx = int(len(samples) * config.TRAIN_SPLIT)
        if split == "train":
            self.samples = samples[:split_idx]
        elif split == "val":
            self.samples = samples[split_idx:]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        sample = self.samples[idx]
        img1 = Image.open(sample["img1_path"]).convert("RGB")
        img2 = Image.open(sample["img2_path"]).convert("RGB")

        # pixel-wise absolute diff in native resolution
        diff = ImageChops.difference(img2, img1)
        width, height = diff.size

        boxes, labels = parse_matched_annotation_file(sample["ann_path"])

        # filter tiny/degenerate boxes
        filtered_boxes = []
        filtered_labels = []
        for (x_min, y_min, x_max, y_max), cls in zip(boxes, labels):
            if (x_max - x_min) <= 1 or (y_max - y_min) <= 1:
                continue
            filtered_boxes.append([x_min, y_min, x_max, y_max])
            filtered_labels.append(cls)

        if not filtered_boxes:
            # fallback dummy to avoid processor/DETR crash; will be filtered during collate
            filtered_boxes = [[0.0, 0.0, 1.0, 1.0]]
            filtered_labels = [0]

        target = {
            "boxes": torch.tensor(filtered_boxes, dtype=torch.float32),
            "class_labels": torch.tensor(filtered_labels, dtype=torch.int64),
            "size": (height, width),
            "image_id": torch.tensor([idx]),
            "meta": sample,
        }
        return diff, target


def collate_fn_with_processor(processor):
    """
    Builds a collate_fn that runs DetrImageProcessor on-the-fly and skips empty samples.
    """

    def _collate(batch):
        images, targets = zip(*batch)
        coco_targets = []
        filtered_images = []
        filtered_targets = []
        for img, t in zip(images, targets):
            ann_list = []
            for box, label in zip(t["boxes"], t["class_labels"]):
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                if w <= 1 or h <= 1:
                    continue
                ann_list.append(
                    {
                        "image_id": int(t["image_id"].item()),
                        "bbox": [x1, y1, w, h],
                        "category_id": int(label.item()),
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )

            if not ann_list:
                continue

            filtered_images.append(img)
            filtered_targets.append(t)
            coco_targets.append({"image_id": int(t["image_id"].item()), "annotations": ann_list})

        if not filtered_images:
            return None

        encoding = processor(images=filtered_images, annotations=coco_targets, return_tensors="pt")
        # keep originals for eval/metrics
        encoding["orig_targets"] = filtered_targets
        encoding["orig_target_sizes"] = torch.tensor([t["size"] for t in filtered_targets], dtype=torch.long)
        return encoding

    return _collate
