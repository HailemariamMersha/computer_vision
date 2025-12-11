# Assignment 1 - CNN from Scratch & ResNet Fine-Tuning

## Repository Structure
- `cnn_from_scratch.py` — My implementation of a CNN framework from scratch and a tiny MNIST training run. I created a private repo and cloned it to HPC.
- `tests.py` — Stage-by-stage numeric gradient checks and sanity tests for your framework.
- `resnet18.py` — Fine-tuning ResNet-18 on EMNIST (linear probe, partial unfreeze, full fine-tune).
- `task1.slurm` — SLURM script to run `tests.py` and `cnn_from_scratch.py`.
- `task2.slurm` — SLURM script to run `resnet18.py`.
- `task1.out` / `task1.err` — Output and error logs from Task 1.
- `task2.out` / `task2.err` — Output and error logs from Task 2.
- `report.tex` — LaTeX report.
- `README.md` — (this file).

## Running Instructions

### Task 1 (CNN from Scratch on MNIST)
Submit the job with:
```bash
sbatch task1.slurm
```

This SLURM script will:
1. Run `tests.py` — performs numeric gradient checks, shape checks, and sanity tests.
2. Run `cnn_from_scratch.py` — trains the custom CNN on MNIST.

- **Logs**:
  - `task1.out`: contains all print statements from both scripts (e.g., "ALL TESTS PASSED", epoch losses/accuracies).
  - `task1.err`: contains any runtime errors, tracebacks, or warnings.

### Task 2 (ResNet-18 Fine-Tuning on EMNIST)
Submit the job with:
```bash
sbatch task2.slurm
```

This SLURM script will run `resnet18.py`, which:
- Loads and preprocesses EMNIST (byclass).
- Runs three regimes: linear probe, partial unfreeze, and full fine-tuning.
- Prints training/validation/test losses and accuracies.
- Outputs classification reports and confusion matrices.

- **Logs**:
  - `task2.out`: contains print statements (training logs, test results, classification report).
  - `task2.err`: contains runtime errors or warnings.

## Notes
- Task 2 took too long so I used the output from Google Colab. I added a half complete batch files from HPC. 
- Overlay image must be mounted in **read-only mode** (`:ro`) when submitting jobs, since multiple processes may access it.
- Ensure the conda environment (`computer_vision`) is activated inside the SLURM scripts.
- All figures for the report can be generated locally (e.g., Google Colab) if HPC plotting is not practical.

## Example Run Commands
```bash
# Run Task 1
sbatch task1.slurm

# Run Task 2
sbatch task2.slurm

# Inspect logs
less task1.out
less task1.err
less task2.out
less task2.err
```

Author: Hailemariam Mersha  
NetID: hbm9834  
