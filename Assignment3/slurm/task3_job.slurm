#!/bin/bash
#SBATCH --job-name=task3_detr
#SBATCH --account=csci_ua_0480_042-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%j_task3.out
#SBATCH --error=%j_task3.err
# End-to-end pipeline: labeller -> COCO JSON -> train -> eval (precision/recall)

SIF=/scratch/hbm9834/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/hbm9834/overlay-25GB-500K.ext3
PROJECT_ROOT=/scratch/hbm9834/Computer_vision/Assignment3
DATA_ROOT=${PROJECT_ROOT}/cv_data_hw2
MATCHED_DIR=${PROJECT_ROOT}/matched_annotations
ANNOTATIONS_DIR=${PROJECT_ROOT}/annotations
CKPT_DIR=${PROJECT_ROOT}/outputs/checkpoints
METRICS_OUT=${PROJECT_ROOT}/outputs/eval/metrics.txt
VIS_OUT=${PROJECT_ROOT}/outputs/eval_vis

singularity exec --bind /scratch --nv \
  --overlay ${OVERLAY}:ro \
  ${SIF} /bin/bash -c "
    set -euo pipefail
    source /ext3/env.sh
    # Avoid unset var errors from binutils hook during conda activate
    export ADDR2LINE=\$(which addr2line 2>/dev/null || true)
    export NM=\$(which nm 2>/dev/null || true)
    export OBJCOPY=\$(which objcopy 2>/dev/null || true)
    export OBJDUMP=\$(which objdump 2>/dev/null || true)
    export READELF=\$(which readelf 2>/dev/null || true)
    export STRINGS=\$(which strings 2>/dev/null || true)
    export STRIP=\$(which strip 2>/dev/null || true)
    export AR=\$(which ar 2>/dev/null || true)
    export CXXFILT=\$(which c++filt 2>/dev/null || true)
    conda activate computer_vision
    cd ${PROJECT_ROOT}
    echo '=== Step 1/4: Generate matched annotations ==='
    python data_ground_truth_labeller.py

    echo '=== Step 2/4: Build COCO annotations (train/val/test) ==='
    mkdir -p ${ANNOTATIONS_DIR}
    python build_coco_annotations.py \
      --matched_dir ${MATCHED_DIR} \
      --images_root ${DATA_ROOT} \
      --output_dir ${ANNOTATIONS_DIR}

    echo '=== Step 3/4: Train DETR (Option 2 pixel diff) ==='
    python train.py --strategy all
    baseline_ckpt=\$(ls -t ${CKPT_DIR}/detr_option2_all_epoch*.pth 2>/dev/null | head -n 1)

    echo '=== Step 4/4: Run ablations (loss comparison + eval) ==='
    python ablations.py \
      --images_root ${DATA_ROOT} \
      --test_json ${ANNOTATIONS_DIR}/annotations_test.json \
      --metrics_dir ${PROJECT_ROOT}/outputs/ablation_metrics \
      --eval_vis_dir ${PROJECT_ROOT}/outputs/ablation_eval_vis

    echo '=== Final: Evaluate baseline checkpoint ==='
    latest_ckpt=\${baseline_ckpt}
    if [ -z \"\${latest_ckpt}\" ]; then
      echo 'No checkpoint found for evaluation.' >&2
      exit 1
    fi
    python eval_detr_moved.py \
      --images_root ${DATA_ROOT} \
      --test_json ${ANNOTATIONS_DIR}/annotations_test.json \
      --ckpt \${latest_ckpt} \
      --vis_dir ${VIS_OUT} \
      --metrics_out ${METRICS_OUT}
  "
