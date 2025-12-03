import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

# Category mapping from VIRAT labels
CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "other_vehicle"},
    {"id": 4, "name": "other_object"},
    {"id": 5, "name": "bike"},
]


def _parse_annotation_line(line: str) -> Tuple[int, float, float, float, float, int]:
    parts = line.strip().split()
    if len(parts) < 6:
        raise ValueError(f"Malformed annotation line: '{line}'")
    match_id = int(parts[0])
    x, y, w, h = map(float, parts[1:5])
    cls_id = int(parts[5])
    return match_id, x, y, w, h, cls_id


def _load_frame2_boxes(match_file: Path) -> List[Dict]:
    """
    The labeller writes two rows per matched object: first for frame1, second for frame2.
    We only keep the frame2 rows (even lines in zero-based indexing).
    """
    boxes = []
    with match_file.open("r") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]

    # Take every second line starting at index 1 (frame2)
    for i in range(1, len(lines), 2):
        match_id, x, y, w, h, cls_id = _parse_annotation_line(lines[i])
        boxes.append(
            {
                "match_id": match_id,
                "bbox": [x, y, w, h],
                "category_id": cls_id,
            }
        )
    return boxes


def _image_from_match_file(match_file: Path) -> Tuple[str, str, str]:
    """
    Derive folder and frame2 filename from '<folder>-<img1>-<img2>_match.txt'.
    Returns (folder_name, frame1_png_name, frame2_png_name).
    """
    stem = match_file.stem  # e.g., Pair_S_...-S_...-S_..._match
    if not stem.endswith("_match"):
        raise ValueError(f"Unexpected match file name: {match_file.name}")
    stem = stem.replace("_match", "")
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse match file name: {match_file.name}")
    folder, img1, img2 = parts[0], parts[1], parts[2]
    return folder, f"{img1}.png", f"{img2}.png"


def build_coco_from_matched_annotations(
    matched_dir: str,
    images_root: str,
    output_json: str,
    split: str = "train",
    use_single_class: bool = False,
    files_subset: List[Path] = None,
) -> None:
    """
    Read matched annotation txt files and build a COCO JSON.
    """
    matched_path = Path(matched_dir)
    image_root_path = Path(images_root)
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = files_subset if files_subset is not None else sorted(matched_path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No matched annotation files found in {matched_dir}")

    categories = [{"id": 1, "name": "moved_object"}] if use_single_class else CATEGORIES
    coco: Dict = {
        "info": {"description": f"Moved object detection ({split})"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    ann_id = 1
    for img_id, match_file in enumerate(files, start=1):
        folder, frame1_name, frame2_name = _image_from_match_file(match_file)
        rel_path_frame1 = str(Path("data") / folder / frame1_name)
        rel_path_frame2 = str(Path("data") / folder / frame2_name)
        img_path = image_root_path / rel_path_frame2
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for {match_file.name}: {img_path}")
        frame1_path = image_root_path / rel_path_frame1
        if not frame1_path.exists():
            raise FileNotFoundError(f"Frame1 image not found for {match_file.name}: {frame1_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        height, width = img.shape[:2]

        boxes = _load_frame2_boxes(match_file)
        coco["images"].append(
            {
                "id": img_id,
                # Standard COCO field kept as frame2 for compatibility with processors/evaluators
                "file_name": rel_path_frame2,
                # Extra fields to allow Option 2 (pixel-wise diff)
                "file_name_frame1": rel_path_frame1,
                "file_name_frame2": rel_path_frame2,
                "width": width,
                "height": height,
            }
        )

        for box in boxes:
            category_id = 1 if use_single_class else box["category_id"]
            x, y, w, h = box["bbox"]
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    with out_path.open("w") as f:
        json.dump(coco, f)
    print(f"Wrote {len(coco['images'])} images and {len(coco['annotations'])} boxes to {out_path}")


def split_and_build(
    matched_dir: str,
    images_root: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_single_class: bool = False,
) -> None:
    """
    Shuffle matched files and create train/val/test COCO JSONs.
    """
    matched_path = Path(matched_dir)
    files = sorted(matched_path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No matched annotation files found in {matched_dir}")

    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def _write(files_subset: List[Path], split_name: str):
        build_coco_from_matched_annotations(
            matched_dir=matched_dir,
            images_root=images_root,
            output_json=str(output_root / f"annotations_{split_name}.json"),
            split=split_name,
            use_single_class=use_single_class,
            files_subset=files_subset,
        )

    _write(train_files, "train")
    _write(val_files, "val")
    _write(test_files, "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Build COCO annotations from matched outputs.")
    parser.add_argument("--matched_dir", type=str, required=True, help="Directory with *_match.txt files.")
    parser.add_argument("--images_root", type=str, required=True, help="Root containing original images (cv_data_hw2).")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write annotations_*.json.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_class", action="store_true", help="Collapse to one 'moved_object' class.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_and_build(
        matched_dir=args.matched_dir,
        images_root=args.images_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_single_class=args.single_class,
    )
