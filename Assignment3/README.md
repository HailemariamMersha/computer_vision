# Assignment 3 – Moved Object Detection (Option 2: Pixel Diff into DETR)

This repo implements Option 2 as described in the handout: take the pixel-wise difference between two frames and fine-tune `facebook/detr-resnet-50` on the moved objects. The pipeline is processor-based (no COCO JSON needed for training/eval).

## Layout
- `data_ground_truth_labeller.py` – provided matcher; generates `matched_annotations/` and `visual_matches/` (IoU filter happens here).
- `config.py` – paths, hyperparameters, and utility for creating output dirs.
- `dataset.py` – Option 2 dataset: loads both frames, computes abs diff, filters tiny boxes, and feeds `DetrImageProcessor` directly (no manual resizing/COCO).
- `model.py` – DETR loader and fine-tuning strategy helper.
- `train.py` – processor-based training loop on pixel diffs; saves best checkpoint only.
- `ablations.py` – runs strategy/LR sweeps and evaluates them (precision/recall).
- `eval_detr_moved.py` – processor-based eval + four-panel visuals (initial/final GT on top, initial/final preds on bottom).
- `slurm/task3_job.slurm` – example SLURM script (update paths before submitting).

## Quickstart (Option 2, local paths)
```bash
# 0) from Assignment3/
conda activate computer_vision  # env with torch/transformers/PIL/torchvision

# 1) Generate matched annotations (IoU-based filtering handled inside)
python data_ground_truth_labeller.py  # writes to ./matched_annotations by default

# 2) Train on pixel-wise differences (processor-based, best checkpoint only)
python train.py --strategy all  # or backbone_only/transformer_only/head_only

# 3) Optional: run ablations + eval
python ablations.py
```

## HPC notes
- Set paths in `config.py` (DATA_ROOT, MATCHED_ANN_DIR, OUTPUT_DIR) to your `/scratch/<netid>/...`.
- Ensure `data_ground_truth_labeller.py` writes matched annotations to the same `MATCHED_ANN_DIR`.
- Update `slurm/task3_job.slurm` with your SIF/overlay/PROJECT_ROOT and submit with `sbatch slurm/task3_job.slurm`.

## Outputs
- `matched_annotations/` – matcher txt files (2 lines/object; second line is frame2 box used for GT).
- `outputs/checkpoints/` – DETR weights per epoch.
- `outputs/logs/`, `outputs/visualizations/` – placeholders for logs/vis if you add them.
