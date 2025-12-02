#!/usr/bin/env python3
"""
Bipartite Object Matching on VIRAT Frame Pairs
Author: Hailemariam Mesha (hbm9834
Python >= 3.9

Usage (examples):
  pip install -r requirements.txt #Install dependencies
  python bipartite_match_virat.py --index ./cv_data_hw2/index.txt --out outputs
  # By default, produces 5 report panels.
  # Use --n-report -1 to process all pairs (not recommended — many folders).

This script:
  1) Loads frame pairs and annotation text files from an index file or folder.
  2) Parses object annotations (id, class, bbox) from text files.
  3) Builds a cost matrix combining (1 - IoU), centroid distance, and class mismatch.
  4) Applies the Hungarian algorithm to perform bipartite matching.
  5) Draws same-colored bounding boxes on matched objects in both frames.
  6) Exports a small number (default = 5) of side-by-side panels for the report.

"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path
import cv2

# importing the Hungarian algorithm (SciPy)
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


@dataclass
class Obj:
    oid: int
    cls: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class FrameAnn:
    image_path: str
    ann_path: Optional[str]
    objects: List[Obj]


@dataclass
class Pair:
    name: str
    frame1: FrameAnn
    frame2: FrameAnn


def load_image_bgr(path: str):
    """Read an image from disk in BGR format."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def iou(boxA, boxB) -> float:
    (xA1, yA1, xA2, yA2) = boxA
    (xB1, yB1, xB2, yB2) = boxB
    interX1, interY1 = max(xA1, xB1), max(yA1, yB1)
    interX2, interY2 = min(xA2, xB2), min(yA2, yB2)
    iw, ih = max(0.0, interX2 - interX1), max(0.0, interY2 - interY1)
    inter = iw * ih
    areaA = max(0.0, (xA2 - xA1)) * max(0.0, (yA2 - yA1))
    areaB = max(0.0, (xB2 - xB1)) * max(0.0, (yB2 - yB1))
    denom = areaA + areaB - inter + 1e-9
    return float(inter / denom) if denom > 0 else 0.0


def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def l2(a, b) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def random_color(seed: int):
    """Generate a deterministic random color given a seed."""
    # Ensure seed is within valid range [0, 2**32 - 1]
    seed = seed % (2**32)
    rs = np.random.RandomState(seed)
    return (int(rs.randint(0, 255)), int(rs.randint(0, 255)), int(rs.randint(0, 255)))

# Annotation parser
def xywh_to_xyxy(x, y, w, h):
    return (x, y, x + w, y + h)


def parse_annotation(txt_path: Optional[str], image_shape=None) -> List[Obj]:
    """Parse an annotation file into a list of Obj instances."""
    objs: List[Obj] = []
    if txt_path is None or not os.path.exists(txt_path):
        return objs

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.startswith("#")]

    idx_auto = 0
    for ln in lines:
        # Split on whitespace and commas
        parts = [p for p in re.split(r"[, \s]+", ln.strip()) if p]
        if len(parts) < 5:
            continue

        # Convert strings to numbers
        nums = []
        for p in parts:
            try:
                nums.append(float(p) if "." in p else int(p))
            except ValueError:
                pass

        # Support formats:
        # VIRAT format: field_id object_id frame_id x y width height class_id (8 numbers)
        # Generic format: id cls x y w h [extra fields] (6+ numbers)
        # Generic format: cls x y w h [extra fields] (5 numbers, auto-assign id)
        
        oid = None
        cls_ = None
        x = y = w = h = None

        if len(nums) >= 8:
            # VIRAT format: field_id(0) object_id(1) frame_id(2) x(3) y(4) width(5) height(6) class_id(7)
            oid = int(nums[1])  # object_id
            x, y, w, h = map(float, nums[3:7])  # x, y, width, height
            cls_ = int(nums[7])  # class_id
        elif len(nums) >= 6 and isinstance(nums[0], int) and isinstance(nums[1], int):
            # Format: id cls x y w h [extra fields]
            oid = int(nums[0])
            cls_ = int(nums[1])
            x, y, w, h = map(float, nums[2:6])
        elif len(nums) >= 5 and isinstance(nums[0], int):
            # Format: cls x y w h [extra fields] - auto-assign id
            oid = idx_auto
            idx_auto += 1
            cls_ = int(nums[0])
            x, y, w, h = map(float, nums[1:5])

        if x is None or oid is None or cls_ is None:
            continue

        # Convert to (x1, y1, x2, y2)
        box = xywh_to_xyxy(x, y, w, h)
        x1, y1, x2, y2 = box

        # Ensure coordinates are within image bounds
        if image_shape is not None:
            h_img, w_img = image_shape[:2]
            x1 = float(np.clip(x1, 0, w_img - 1))
            x2 = float(np.clip(x2, 0, w_img - 1))
            y1 = float(np.clip(y1, 0, h_img - 1))
            y2 = float(np.clip(y2, 0, h_img - 1))

        objs.append(Obj(oid=oid, cls=cls_, bbox=(x1, y1, x2, y2)))
    return objs



# Load pairs from index file
def pairs_from_index(index_path: str) -> List[Pair]:
    """Read the index file and return a list of Pair objects."""
    pairs: List[Pair] = []
    base = Path(index_path).parent
    with open(index_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != 4:
                continue
            f1, a1, f2, a2 = [str(base / p) for p in parts]
            name = f"pair_{i:05d}"
            pairs.append(Pair(
                name=name,
                frame1=FrameAnn(f1, a1, []),
                frame2=FrameAnn(f2, a2, []),
            ))
    return pairs


# Matching
def build_cost_matrix(A: List[Obj], B: List[Obj], image_shape, w_iou=1.0, w_cent=0.3, w_cls=0.5) -> np.ndarray:
    # a cost matrix combining IoU, centroid distance, and class penalty.
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)

    H, W = image_shape[:2]
    diag = float(np.hypot(W, H)) + 1e-9
    C = np.zeros((len(A), len(B)), dtype=np.float32)

    for i, oa in enumerate(A):
        ca = centroid(oa.bbox)
        for j, ob in enumerate(B):
            cb = centroid(ob.bbox)
            iou_val = iou(oa.bbox, ob.bbox)
            d = l2(ca, cb) / diag
            cls_pen = 0.0 if oa.cls == ob.cls else 1.0
            C[i, j] = (w_iou * (1.0 - iou_val)) + (w_cent * d) + (w_cls * cls_pen)
    return C


def hungarian_match(C: np.ndarray):
    if C.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if linear_sum_assignment is None:
        rows, cols = [], []
        Ccopy = C.copy()
        while True:
            idx = np.unravel_index(np.argmin(Ccopy, axis=None), Ccopy.shape)
            r, c = int(idx[0]), int(idx[1])
            if np.isinf(Ccopy[r, c]):
                break
            rows.append(r)
            cols.append(c)
            Ccopy[r, :] = np.inf
            Ccopy[:, c] = np.inf
            if len(rows) == C.shape[0] or len(cols) == C.shape[1]:
                break
        return np.array(rows), np.array(cols)
    return linear_sum_assignment(C)


# Visualization
def draw_matches(pair: Pair, max_cost: float = 2.0, w_iou=1.0, w_cent=0.3, w_cls=0.5):
    #Draw same-colored boxes for matched objects across two frames.
    img1 = load_image_bgr(pair.frame1.image_path)
    img2 = load_image_bgr(pair.frame2.image_path)

    A = parse_annotation(pair.frame1.ann_path, img1.shape)
    B = parse_annotation(pair.frame2.ann_path, img2.shape)

    C = build_cost_matrix(A, B, img1.shape, w_iou=w_iou, w_cent=w_cent, w_cls=w_cls)
    rows, cols = hungarian_match(C)

    # Assign consistent colors for matched pairs
    colors: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for r, c in zip(rows, cols):
        if C[r, c] <= max_cost:
            colors[(A[r].oid, B[c].oid)] = random_color((A[r].oid * 73856093) ^ (B[c].oid * 19349663))

    vis1, vis2 = img1.copy(), img2.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_box(vis, obj, color, tag):
        x1, y1, x2, y2 = map(int, obj.bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, tag, (x1, max(0, y1 - 5)), font, 0.6, color, 2, cv2.LINE_AA)

    matchedA = {oidA for (oidA, _) in colors.keys()}
    matchedB = {oidB for (_, oidB) in colors.keys()}

    for (oidA, oidB), color in colors.items():
        objA = next((o for o in A if o.oid == oidA), None)
        objB = next((o for o in B if o.oid == oidB), None)
        if objA:
            draw_box(vis1, objA, color, f"id{objA.oid}/c{objA.cls}")
        if objB:
            draw_box(vis2, objB, color, f"id{objB.oid}/c{objB.cls}")

    gray = (180, 180, 180)
    for a in A:
        if a.oid not in matchedA:
            draw_box(vis1, a, gray, f"id{a.oid}/c{a.cls}")
    for b in B:
        if b.oid not in matchedB:
            draw_box(vis2, b, gray, f"id{b.oid}/c{b.cls}")

    # Concatenate side-by-side
    H = max(vis1.shape[0], vis2.shape[0])
    W = vis1.shape[1] + vis2.shape[1]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:vis1.shape[0], :vis1.shape[1]] = vis1
    canvas[:vis2.shape[0], vis1.shape[1]:vis1.shape[1] + vis2.shape[1]] = vis2
    cv2.putText(canvas, f"{pair.name} | matches={len(colors)}", (10, 25),
                font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
# Main
def main():
    ap = argparse.ArgumentParser(description="VIRAT Frame Pair Bipartite Matching")
    ap.add_argument("--index", type=str, required=True, help="Path to index.txt listing image/annotation pairs")
    ap.add_argument("--out", type=str, default="outputs", help="Output folder for generated panels")
    ap.add_argument("--w-iou", type=float, default=1.0, help="Weight for (1 - IoU) term")
    ap.add_argument("--w-cent", type=float, default=0.3, help="Weight for centroid distance term")
    ap.add_argument("--w-cls", type=float, default=0.5, help="Weight for class mismatch penalty")
    ap.add_argument("--max-cost", type=float, default=2.0, help="Max allowed cost for a valid match")
    ap.add_argument("--n-report", type=int, default=5,
                    help="Number of pairs to export (set to -1 to process all pairs)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pairs = pairs_from_index(args.index)
    if not pairs:
        raise SystemExit("No pairs found in index file.")

    # Shuffle pairs for random selection of examples
    rng = np.random.default_rng(42)
    order = np.arange(len(pairs))
    rng.shuffle(order)

    limit = len(pairs) if args.n_report < 0 else min(args.n_report, len(pairs))
    print(f"Processing {limit} pairs (out of {len(pairs)} total)...")

    for i in range(limit):
        pair = pairs[order[i]]
        panel = draw_matches(pair, max_cost=args.max_cost,
                             w_iou=args.w_iou, w_cent=args.w_cent, w_cls=args.w_cls)
        out_path = os.path.join(args.out, f"{pair.name}_panel.jpg")
        cv2.imwrite(out_path, panel)

    print(f"Saved {limit} panels to '{args.out}/'.")


if __name__ == "__main__":
    main()
