#!/bin/bash
#SBATCH --job-name=task3_detr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%j_task3.out
#SBATCH --error=%j_task3.err

# Update these paths for your cluster setup
SIF=/scratch/<netid>/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/<netid>/overlay-25GB-500K.ext3
PROJECT_ROOT=/scratch/<netid>/assignment3_moved_detr

singularity exec --bind /scratch --nv \
  --overlay ${OVERLAY}:ro \
  ${SIF} /bin/bash -c "
    source /ext3/env.sh
    conda activate computer_vision
    cd ${PROJECT_ROOT}
    # 1) generate matched annotations (run once)
    # python data_ground_truth_labeller.py
    # 2) build COCO splits (run once)
    # python build_coco_annotations.py --matched_dir matched_annotations --images_root cv_data_hw2 --output_dir annotations
    # 3) train
    python train_detr_moved.py --images_root cv_data_hw2 --train_json annotations/annotations_train.json --val_json annotations/annotations_val.json --output_dir outputs/model
  "
