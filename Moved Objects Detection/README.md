
# Moved Object Detection (using Pixel Diff into DETR)

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
- `outputs/checkpoints/` – DETR weights (best per run).
- `outputs/eval/metrics.txt` – baseline val precision/recall at the best checkpoint.
- `outputs/eval_vis/` – four-panel visuals (e.g., `vis_000.png`, `vis_023.png`).
- `outputs/ablation_results.csv` – summary table for all ablations.
- `outputs/ablation_metrics/*.txt` – per-ablation precision/recall.
- `outputs/ablation_eval_vis/<run>/` – visuals for each ablation run.

## Results (from outputs)
- Baseline (all, 40 epochs, best at epoch 37): val loss 1.2018, val precision 0.302, val recall 0.787; eval precision 0.2079, eval recall 0.6833.
- Ablations (100 epochs each):

| Setting | LR | Val Loss | Val Precision | Val Recall |
|---------|----|----------|---------------|------------|
| all_lr1e-5 | 1e-5 | 1.0516 | 0.5280 | 0.7580 |
| all_lr5e-5 | 5e-5 | 1.4008 | 0.2470 | 0.6330 |
| backbone_lr1e-5 | 1e-5 | 1.8970 | 0.5050 | 0.6210 |
| transformer_lr1e-5 | 1e-5 | **1.3291** | **0.6000** | **0.7250** |
| head_lr1e-5 | 1e-5 | 1.4823 | 0.2080 | 0.7040 |

Transformer-only gave the best balance; all_lr1e-5 was close. Higher LR on all-params hurt precision.

## Evaluation quick command
```bash
# from Assignment3/
python eval_detr_moved.py --ckpt outputs/checkpoints/detr_option2_all_best.pth --split val
```
