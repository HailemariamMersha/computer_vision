import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import DetrImageProcessor
from PIL import ImageChops


class MovedObjectDetrDataset(Dataset):
    """
    PyTorch dataset that reads COCO JSON produced by build_coco_annotations.py
    and returns inputs compatible with DetrForObjectDetection.
    Option 2: uses pixel-wise difference between frame1 and frame2 as the image input.
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
        # Prefer explicit frame paths (written by updated build_coco_annotations.py); fallback to frame2 only.
        frame1_name = img_info.get("file_name_frame1", img_info["file_name"])
        frame2_name = img_info.get("file_name_frame2", img_info["file_name"])
        img1_path = self.images_root / frame1_name
        img2_path = self.images_root / frame2_name

        image1 = Image.open(img1_path).convert("RGB")
        image2 = Image.open(img2_path).convert("RGB")

        # Compute pixel-wise absolute difference (Option 2)
        if image1.size != image2.size:
            # Align sizes if needed (rare for provided data)
            image2 = image2.resize(image1.size)
        diff_image = ImageChops.difference(image1, image2)

        anns = self.ann_by_image.get(img_id, [])

        # The processor will build target dicts for DETR using the diff image
        encoding = self.processor(
            images=diff_image,
            annotations={"image_id": img_id, "annotations": anns},
            return_tensors="pt",
        )
        # Separate tensor fields from label list; squeeze batch on tensors only
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]
        return {"pixel_values": pixel_values, "labels": labels}


def detr_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader; stacks pixel_values and keeps labels as list.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}
