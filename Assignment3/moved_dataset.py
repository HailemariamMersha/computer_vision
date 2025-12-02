import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import DetrImageProcessor


class MovedObjectDetrDataset(Dataset):
    """
    PyTorch dataset that reads COCO JSON produced by build_coco_annotations.py
    and returns inputs compatible with DetrForObjectDetection.
    """

    def __init__(self, coco_json_path: str, images_root: str, processor: DetrImageProcessor):
        self.images_root = Path(images_root)
        self.processor = processor

        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        self.images: List[Dict] = coco["images"]
        self.annotations: List[Dict] = coco["annotations"]
        self.categories = {c["id"]: c["name"] for c in coco["categories"]}

        self.ann_by_image: Dict[int, List[Dict]] = {}
        for ann in self.annotations:
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = self.images_root / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self.ann_by_image.get(img_id, [])

        # The processor will build target dicts for DETR
        encoding = self.processor(
            images=image,
            annotations={"image_id": img_id, "annotations": anns},
            return_tensors="pt",
        )
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding


def detr_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader; stacks pixel_values and keeps labels as list.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}
