import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

from moved_dataset import MovedObjectDetrDataset, detr_collate_fn


def load_label_maps(coco_json: Path) -> Dict[str, Dict[int, str]]:
    import json

    with open(coco_json, "r") as f:
        data = json.load(f)
    id2label = {c["id"]: c["name"] for c in data["categories"]}
    label2id = {v: k for k, v in id2label.items()}
    return {"id2label": id2label, "label2id": label2id}


def parse_args():
    script_dir = Path(__file__).parent
    default_data_root = script_dir / "cv_data_hw2"
    default_ann_root = script_dir / "annotations"
    parser = argparse.ArgumentParser(description="Train DETR for moved-object detection (Option 2).")
    parser.add_argument("--images_root", type=str, default=str(default_data_root), help="Root containing cv_data_hw2.")
    parser.add_argument("--train_json", type=str, default=str(default_ann_root / "annotations_train.json"))
    parser.add_argument("--val_json", type=str, default=str(default_ann_root / "annotations_val.json"))
    parser.add_argument("--output_dir", type=str, default=str(script_dir / "outputs/model"), help="Where to save model.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to(device)
        labels: List[Dict] = [{k: v.to(device) for k, v in lab.items()} for lab in batch["labels"]]
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * pixel_values.size(0)
        total_samples += pixel_values.size(0)
    return total_loss / max(total_samples, 1)


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels: List[Dict] = [{k: v.to(device) for k, v in lab.items()} for lab in batch["labels"]]
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item() * pixel_values.size(0)
            total_samples += pixel_values.size(0)
    return total_loss / max(total_samples, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_maps = load_label_maps(Path(args.train_json))
    num_classes = len(label_maps["id2label"])

    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", ignore_mismatched_sizes=True
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        id2label=label_maps["id2label"],
        label2id=label_maps["label2id"],
        ignore_mismatched_sizes=True,
    ).to(device)

    train_dataset = MovedObjectDetrDataset(args.train_json, args.images_root, processor)
    val_dataset = MovedObjectDetrDataset(args.val_json, args.images_root, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detr_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detr_collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved model and processor to {output_dir}")


if __name__ == "__main__":
    main()
