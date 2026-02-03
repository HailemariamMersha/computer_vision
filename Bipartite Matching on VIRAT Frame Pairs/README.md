# Bipartite Object Matching on VIRAT Frame Pairs

**Author:** Hailemariam Mesha (hbm9834)  
**Date:** 2025-11-04

## Approach
### 1. Annotation Parsing
- Parsed per-frame VIRAT annotations (8 space-separated fields: `field_id object_id frame_id x y width height class_id`) into objects with `(oid, class, bbox)`.
- Normalized boxes to `(x1, y1, x2, y2)` format (top-left, bottom-right corners).
- Also supports generic 5–6 number formats (`cls x y w h` or `id cls x y w h`) for flexibility.
- Clipped all bounding boxes to image boundaries to ensure valid coordinates.

### 2. Cost Matrix Construction
Built a cost matrix combining three weighted signals for each pair of objects (frame 1 ↔ frame 2):
- **IoU-based cost**: `(1 - IoU)` — favors highly overlapping candidates.
- **Spatial cost**: Normalized centroid distance (distance / image diagonal) — keeps matches spatially plausible.
- **Class cost**: Class-mismatch penalty (0 if classes match, 1 otherwise) — enforces label consistency.

**Cost formula:**
$$C_{i,j} = w_{\text{iou}} \cdot (1 - \text{IoU}) + w_{\text{cent}} \cdot d_{\text{norm}} + w_{\text{cls}} \cdot p_{\text{cls}}$$

### 3. Assignment & Matching
- Solved the assignment problem via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), finding the minimum-cost perfect matching.
- Greedy fallback available if SciPy is unavailable (recommended to keep SciPy for optimal results).
- Filtered matches: only pairs with cost ≤ `--max-cost` threshold are accepted.

### 4. Visualization
- Assigned random colors to matched object pairs (seeded by object IDs for reproducibility).
- Drew same-colored bounding boxes on both frames for matched pairs.
- Unmatched objects drawn in gray to distinguish them.
- Generated side-by-side panel visualizations with match count and object labels (`id{oid}/c{class}`).

## Why This Works for Object Tracking

Bipartite matching is a principled approach for temporal object association:
- Each object in frame $t$ is matched to **at most one** object in frame $t+1$ (and vice versa).
- The cost matrix encodes multiple cues (overlap, proximity, class consistency), mirroring DETR-style assignment.
- The Hungarian algorithm guarantees the optimal assignment under the given cost function.
- This is more robust than greedy nearest-neighbor or NMS-based approaches when objects have complex motion or partial occlusion.
- In the VIRAT dataset, this helps track objects across frame boundaries despite appearance changes, scene complexity, and diverse object classes (persons, vehicles, etc.).

## Parameters

All parameters are configurable via command-line arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--w-iou` | 1.0 | Weight on `(1 - IoU)` cost term. Higher values prioritize bounding box overlap. |
| `--w-cent` | 0.3 | Weight on normalized centroid distance. Higher values encourage spatially nearby matches. |
| `--w-cls` | 0.5 | Weight on class-mismatch penalty. Higher values encourage same-class matches. |
| `--max-cost` | 2.0 | Maximum allowed cost to accept a match. Matches above this threshold are discarded. |
| `--n-report` | 5 | Number of frame pairs to visualize and export. |
| `--index` | — | Path to index.txt file (recommended input method). |
| `--root` | — | Alternative: path to extracted dataset root folder. |
| `--zip` | — | Alternative: path to cv_data_hw2.zip archive. |
| `--out` | `outputs` | Output directory for panel visualizations and CSV. |
| `--save-csv` | — | Flag: also export a CSV file of matches. |

## Running the Code

### Installation
```bash
pip install -r requirements.txt
```

### Basic Command (Recommended: Index File)
```bash
python bipartite_match_virat.py --index ./cv_data_hw2/index.txt --out outputs --n-report 5 --save-csv
```

### Alternative: From Extracted Folder
```bash
python bipartite_match_virat.py --root ./cv_data_hw2 --out outputs --n-report 5 --save-csv
```

### Alternative: From Zip Archive
```bash
python bipartite_match_virat.py --zip cv_data_hw2.zip --out outputs --n-report 5 --save-csv --tmp temp_extract
```

### Custom Weight Configuration
Adjust cost weights to prioritize different matching criteria:
```bash
python bipartite_match_virat.py --index ./cv_data_hw2/index.txt --out outputs --n-report 5 \
  --w-iou 1.5 --w-cent 0.5 --w-cls 0.3 --max-cost 1.5
```

### Output Files
- **Panels**: `outputs/pair_XXXXX_panel.jpg` — side-by-side frame visualization with matched objects in color, unmatched in gray.
- **CSV** (if `--save-csv`): `outputs/matches.csv` — contains `pair, row_idx, col_idx, cost` for each match.

## Sample Outputs (attach screenshots)
Insert 5 different pair panels saved to the `outputs/` directory, showing same-colored boxes across the two frames.

1. Pair A — varied object counts and classes
2. Pair B — different background
3. Pair C — dense scene
4. Pair D — small objects
5. Pair E — mixed vehicles/persons

## Assumptions & Implementation Details

### Annotation Format Support
The parser is robust to multiple formats:
- **VIRAT format (primary)**: 8 fields: `field_id object_id frame_id x y width height class_id`
- **Generic 6-field format**: `id cls x y width height`
- **Generic 5-field format**: `cls x y width height` (IDs auto-assigned incrementally)

### Data Processing
- All bounding boxes are clipped to image boundaries (no out-of-bounds coordinates).
- Boxes are internally stored as `(x1, y1, x2, y2)` regardless of input format.
- Coordinates are expected in pixel units (not normalized).

### Matching Behavior
- Only matches with cost ≤ `--max-cost` are retained; all others are discarded.
- Unmatched objects appear in gray on the visualization panels, making assignment gaps transparent.
- Colors for matched pairs are reproducibly seeded by object ID pairs, enabling consistent visualization across runs.

### Hungarian Algorithm
- If SciPy is installed: uses `scipy.optimize.linear_sum_assignment` for optimal O(n³) complexity assignment.
- If SciPy unavailable: falls back to a greedy algorithm. **For production use, keep SciPy installed.**

### Visualization Output
- Each panel shows both frames side-by-side with a title indicating pair name and match count.
- Object labels: `id{oid}/c{class}` — helps verify correctness of both matching and parsing.
- Same color ⟹ matched pair; gray ⟹ unmatched object.
