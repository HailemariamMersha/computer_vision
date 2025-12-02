# Assignment 3 – Moved Object Detection (Option 2: Pixel Diff into DETR)

This repo implements the Option 2 pipeline: take frame differences via the provided matcher, convert to COCO, fine-tune `facebook/detr-resnet-50`, and evaluate.

## Layout
- `data_ground_truth_labeller.py` – provided matcher; generates `matched_annotations/` and `visual_matches/`.
- `build_coco_annotations.py` – converts matcher outputs to COCO JSON splits.
- `moved_dataset.py` – PyTorch dataset + collate for DETR.
- `train_detr_moved.py` – training loop.
- `eval_detr_moved.py` – evaluation + visuals.
- `slurm/task3_job.slurm` – example SLURM script.

## Quickstart (local paths)
```bash
# 0) from Assignment3/
conda activate computer_vision  # or similar env with torch/transformers/opencv

# 1) Generate matched annotations
python data_ground_truth_labeller.py \
  # defaults to cv_data_hw2 under this folder; override inside the script for HPC

# 2) Build COCO splits
python build_coco_annotations.py \
  --matched_dir matched_annotations \
  --images_root cv_data_hw2 \
  --output_dir annotations

# 3) Train
python train_detr_moved.py \
  --images_root cv_data_hw2 \
  --train_json annotations/annotations_train.json \
  --val_json annotations/annotations_val.json \
  --output_dir outputs/model

# 4) Evaluate + visualize
python eval_detr_moved.py \
  --images_root cv_data_hw2 \
  --test_json annotations/annotations_test.json \
  --model_dir outputs/model \
  --vis_dir results/vis
```

## HPC notes
- Set `base_dir`, `output_ann_dir`, and `visual_dir` in `data_ground_truth_labeller.py` to your `/scratch/<netid>/...` paths.
- Mirror the same paths in `build_coco_annotations.py` arguments and `train_detr_moved.py` flags.
- Edit `slurm/task3_job.slurm` to point to your SIF/overlay and `PROJECT_ROOT`, then submit with `sbatch slurm/task3_job.slurm`.

## Outputs
- `matched_annotations/` – matcher txt files (2 lines/object; second line is frame2 box).
- `annotations/annotations_{train,val,test}.json` – COCO-format labels.
- `outputs/model/` – fine-tuned DETR weights + processor.
- `results/vis/` – qualitative overlays (green=GT, red=pred).
