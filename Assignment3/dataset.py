import os
import random
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from config import Config


def parse_matched_annotation_file(path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Reads a matched annotation file where every two rows correspond to one matched object:
    first row is the old box (ignored), second row is the new box (used as GT).
    Returns boxes (x_min, y_min, x_max, y_max) and labels.
    """
    boxes: List[List[float]] = []
    labels: List[int] = []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Process pairs of lines
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        # second line in the pair is the new box
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
            # Prefer new nested layout; fall back to flat files produced by the labeller.
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

        self.img_transform = T.Compose(
            [
                T.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img1 = Image.open(sample["img1_path"]).convert("RGB")
        img2 = Image.open(sample["img2_path"]).convert("RGB")

        orig_w, orig_h = img2.size
        img1_t = self.img_transform(img1)
        img2_t = self.img_transform(img2)

        diff = torch.abs(img2_t - img1_t)

        boxes, labels = parse_matched_annotation_file(sample["ann_path"])
        if boxes:
            # scale boxes from original size to resized size
            scale_x = self.config.IMAGE_WIDTH / orig_w
            scale_y = self.config.IMAGE_HEIGHT / orig_h
            scaled_boxes = []
            for x_min, y_min, x_max, y_max in boxes:
                x_min_r = x_min * scale_x
                x_max_r = x_max * scale_x
                y_min_r = y_min * scale_y
                y_max_r = y_max * scale_y
                scaled_boxes.append([x_min_r, y_min_r, x_max_r, y_max_r])
            boxes = scaled_boxes

            # normalize to [0,1]
            boxes = [
                [
                    x_min / self.config.IMAGE_WIDTH,
                    y_min / self.config.IMAGE_HEIGHT,
                    x_max / self.config.IMAGE_WIDTH,
                    y_max / self.config.IMAGE_HEIGHT,
                ]
                for x_min, y_min, x_max, y_max in boxes
            ]

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_tensor, "class_labels": labels_tensor}
        return diff, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
