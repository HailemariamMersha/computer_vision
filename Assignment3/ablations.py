import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


"""
Run a small set of ablations by invoking train.py with different strategies/LRs.
Captures final train/val loss/metrics, saves the best checkpoint under a unique name,
and (optionally) evaluates each checkpoint on the processor-based val split (no COCO).
"""


ABLATIONS = [
    {"name": "all_lr1e-5", "strategy": "all", "lr": 1e-5, "epochs": 5},
    {"name": "all_lr5e-5", "strategy": "all", "lr": 5e-5, "epochs": 5},
    {"name": "head_lr1e-5", "strategy": "head_only", "lr": 1e-5, "epochs": 5},
    {"name": "head_lr5e-5", "strategy": "head_only", "lr": 5e-5, "epochs": 5},
    {"name": "backbone_lr1e-5", "strategy": "backbone_only", "lr": 1e-5, "epochs": 5},
]


def parse_losses(log: str):
    """
    Look for the last line that looks like:
    Epoch X/Y: train_loss=... val_loss=... val_prec=... val_rec=...
    """
    last_train = None
    last_val = None
    last_prec = None
    last_rec = None
    for line in log.strip().splitlines():
        if "train_loss=" in line and "val_loss=" in line:
            try:
                parts = line.strip().split()
                for p in parts:
                    if p.startswith("train_loss="):
                        last_train = float(p.split("=")[1])
                    if p.startswith("val_loss="):
                        last_val = float(p.split("=")[1])
                    if p.startswith("val_prec="):
                        last_prec = float(p.split("=")[1])
                    if p.startswith("val_rec="):
                        last_rec = float(p.split("=")[1])
            except ValueError:
                continue
    return last_train, last_val, last_prec, last_rec


def run_ablation(cfg, checkpoints_root: Path, args):
    cmd = [
        sys.executable,
        "train.py",
        "--strategy",
        cfg["strategy"],
        "--epochs",
        str(cfg["epochs"]),
        "--lr",
        str(cfg["lr"]),
    ]
    print(f"Running {cfg['name']} -> {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout
    stderr = proc.stderr
    if proc.returncode != 0:
        print(f"[{cfg['name']}] failed with code {proc.returncode}")
        if stdout:
            print("STDOUT:\n", stdout)
        if stderr:
            print("STDERR:\n", stderr)
        return None

    train_loss, val_loss, val_prec, val_rec = parse_losses(stdout)

    # Copy the best checkpoint to a unique filename for this ablation.
    final_ckpt = checkpoints_root / f"detr_option2_{cfg['strategy']}_best.pth"
    saved_ckpt = None
    if final_ckpt.exists():
        ablation_ckpt_dir = checkpoints_root / "ablations"
        ablation_ckpt_dir.mkdir(parents=True, exist_ok=True)
        saved_ckpt = ablation_ckpt_dir / f"{cfg['name']}.pth"
        shutil.copy2(final_ckpt, saved_ckpt)
        print(f"[{cfg['name']}] Saved checkpoint copy to {saved_ckpt}")
    else:
        print(f"[{cfg['name']}] Final checkpoint not found: {final_ckpt}")

    eval_metrics_path = None
    if saved_ckpt and not args.skip_eval:
        eval_metrics_path = eval_checkpoint(cfg["name"], saved_ckpt, args)

    return {
        "name": cfg["name"],
        "strategy": cfg["strategy"],
        "lr": cfg["lr"],
        "epochs": cfg["epochs"],
        "train_loss": train_loss,
        "val_loss": val_loss,
        "ckpt": str(saved_ckpt) if saved_ckpt else "",
        "eval_metrics": str(eval_metrics_path) if eval_metrics_path else "",
        "val_prec": val_prec,
        "val_rec": val_rec,
    }


def eval_checkpoint(name: str, ckpt_path: Path, args) -> Optional[Path]:
    metrics_path = Path(args.metrics_dir) / f"{name}.txt"
    vis_dir = Path(args.eval_vis_dir) / name
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "eval_detr_moved.py",
        "--ckpt",
        str(ckpt_path),
        "--metrics_out",
        str(metrics_path),
        "--vis_dir",
        str(vis_dir),
        "--score_thresh",
        str(args.score_thresh),
        "--iou_thresh",
        str(args.iou_thresh),
    ]
    print(f"[{name}] Evaluating checkpoint -> {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[{name}] eval failed with code {proc.returncode}")
        if proc.stdout:
            print("STDOUT:\n", proc.stdout)
        if proc.stderr:
            print("STDERR:\n", proc.stderr)
        return None
    if proc.stdout:
        print(proc.stdout.strip())
    print(f"[{name}] Wrote eval metrics to {metrics_path}")
    return metrics_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run training ablations (and optional evals).")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="outputs/ablation_metrics",
        help="Where to write per-ablation eval metrics.",
    )
    parser.add_argument(
        "--eval_vis_dir",
        type=str,
        default="outputs/ablation_eval_vis",
        help="Where to write per-ablation eval visualizations.",
    )
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--skip_eval", action="store_true", help="Run training only, skip eval.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints_root = Path("outputs/checkpoints")
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    results = []
    for cfg in ABLATIONS:
        res = run_ablation(cfg, checkpoints_root, args)
        if res:
            results.append(res)

    out_path = Path("outputs/ablation_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
        fieldnames=[
            "name",
            "strategy",
            "lr",
            "epochs",
            "train_loss",
            "val_loss",
            "val_prec",
            "val_rec",
            "ckpt",
            "eval_metrics",
        ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Saved ablation summary to {out_path}")
    for row in results:
        print(
            f"{row['name']}: strategy={row['strategy']} lr={row['lr']} "
            f"epochs={row['epochs']} train_loss={row['train_loss']} "
            f"val_loss={row['val_loss']} ckpt={row['ckpt']} eval_metrics={row['eval_metrics']}"
        )


if __name__ == "__main__":
    main()
