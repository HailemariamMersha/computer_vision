#!/bin/bash
#SBATCH --job-name=task3_detr
#SBATCH --account=csci_ua_0480_042-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%j_task3.out
#SBATCH --error=%j_task3.err
# Run DETR training for task 3 (Option 2: pixel-wise diff)

SIF=/scratch/hbm9834/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/hbm9834/overlay-25GB-500K.ext3
PROJECT_ROOT=/scratch/hbm9834/Computer_vision/Assignment3
singularity exec --bind /scratch --nv \
  --overlay ${OVERLAY}:ro \
  ${SIF} /bin/bash -c "
    source /ext3/env.sh
    conda activate computer_vision
    cd ${PROJECT_ROOT}
    # Generate matched annotations once (uncomment if needed)
    # python data_ground_truth_labeller.py
    # Train on pixel-wise differences (config.py controls paths/hparams)
    python train.py --strategy all
  "
