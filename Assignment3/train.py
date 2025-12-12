import argparse
import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from transformers import DetrImageProcessor

from config import Config
from dataset import MovedObjectDataset, collate_fn_with_processor
from model import create_model, set_finetune_strategy


def compute_pr(outputs, targets, processor, device, score_thresh=0.5, iou_thresh=0.5):
    target_sizes = torch.tensor([t["size"] for t in targets], device=device)
    processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=score_thresh)

    tp = fp = fn = 0
    for preds, tgt in zip(processed, targets):
        keep = preds["scores"] >= score_thresh
        pred_boxes = preds["boxes"][keep].to(device)
        pred_labels = preds["labels"][keep].to(device)
        gt_boxes = tgt["boxes"].to(device)
        gt_labels = tgt["class_labels"].to(device)

        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched = set()
        for i in range(len(pred_boxes)):
            gi = torch.argmax(ious[i]).item()
            if gi in matched:
                continue
            if ious[i, gi] >= iou_thresh and pred_labels[i] == gt_labels[gi]:
                tp += 1
                matched.add(gi)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall, tp, fp, fn


def train_one_epoch(model, loader, optimizer, device, processor):
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        if batch is None:
            continue
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batches += 1
    return total_loss / max(1, batches)


def evaluate(model, loader, device, processor):
    model.eval()
    total_loss = 0.0
    batches = 0
    total_tp = total_fp = total_fn = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            batches += 1

            prec, rec, tp, fp, fn = compute_pr(outputs, batch["orig_targets"], processor, device)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    avg_loss = total_loss / max(1, batches)
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    return avg_loss, precision, recall


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
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    # Allow simple CLI overrides for ablations.
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.weight_decay is not None:
        config.WEIGHT_DECAY = args.weight_decay
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    Config.create_dirs()

    # Explicit size to avoid deprecated max_size warnings in newer transformers
    processor = DetrImageProcessor.from_pretrained(
        config.MODEL_NAME,
        size={
            "shortest_edge": min(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
            "longest_edge": max(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        },
    )
    train_dataset = MovedObjectDataset(config, split="train")
    val_dataset = MovedObjectDataset(config, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn_with_processor(processor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn_with_processor(processor),
    )

    model = create_model(config)
    set_finetune_strategy(model, args.strategy)
    device = torch.device(config.DEVICE)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    num_epochs = args.epochs if args.epochs is not None else config.NUM_EPOCHS
    best_val = float("inf")
    best_ckpt = None
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, processor)
        val_loss, val_prec, val_rec = evaluate(model, val_loader, device, processor)
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_prec={val_prec:.3f} val_rec={val_rec:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = os.path.join(
                config.CHECKPOINT_DIR, f"detr_option2_{args.strategy}_best.pth"
            )
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, best_ckpt)
            print(f"Saved best checkpoint -> {best_ckpt}")

    if best_ckpt:
        print(f"Best val_loss={best_val:.4f} at {best_ckpt}")


if __name__ == "__main__":
    main()
