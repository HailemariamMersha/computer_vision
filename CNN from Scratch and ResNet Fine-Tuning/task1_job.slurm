#!/bin/bash
#SBATCH --job-name=task1_mnist
#SBATCH --account=csci_ua_0480_042-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./%j_task1.out
#SBATCH --error=./%j_task1.err

SIF=/scratch/hbm9834/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/hbm9834/overlay-25GB-500K.ext3

singularity exec --bind /scratch --nv \
  --overlay ${OVERLAY}:ro \
  ${SIF} /bin/bash -c "
    source /ext3/env.sh
    conda activate computer_vision
    cd /scratch/hbm9834/Computer_vision
    python tests.py
    python cnn_from_scratch.py
"

