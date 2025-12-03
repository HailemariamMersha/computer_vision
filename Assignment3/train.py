import argparse
import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import MovedObjectDataset, collate_fn
from model import create_model, set_finetune_strategy


def move_targets_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device):
    moved = []
    for t in targets:
        moved.append(
            {
                "boxes": t["boxes"].to(device),
                "class_labels": t["class_labels"].to(device),
            }
        )
    return moved


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets in loader:
        images_tensor = torch.stack(images).to(device)
        targets = move_targets_to_device(targets, device)

        outputs = model(pixel_values=images_tensor, labels=targets)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images_tensor.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, targets in loader:
            images_tensor = torch.stack(images).to(device)
            targets = move_targets_to_device(targets, device)
            outputs = model(pixel_values=images_tensor, labels=targets)
            loss = outputs.loss
            batch_size = images_tensor.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DETR (Option 2: pixel-wise diff).")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "backbone_only", "transformer_only", "head_only"],
        help="Fine-tuning strategy.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    Config.create_dirs()

    train_dataset = MovedObjectDataset(config, split="train")
    val_dataset = MovedObjectDataset(config, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = create_model(config)
    set_finetune_strategy(model, args.strategy)
    device = torch.device(config.DEVICE)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    num_epochs = args.epochs if args.epochs is not None else config.NUM_EPOCHS
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        ckpt_path = os.path.join(
            config.CHECKPOINT_DIR, f"detr_option2_{args.strategy}_epoch{epoch + 1}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
