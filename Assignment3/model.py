import torch
from transformers import DetrForObjectDetection

from config import Config


def create_model(config: Config):
    model = DetrForObjectDetection.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
    )
    return model


def set_finetune_strategy(model: torch.nn.Module, strategy: str):
    """
    strategy: "all", "backbone_only", "transformer_only", "head_only"
    """
    for p in model.parameters():
        p.requires_grad = False

    if strategy == "all":
        for p in model.parameters():
            p.requires_grad = True

    elif strategy == "backbone_only":
        for p in model.model.backbone.parameters():
            p.requires_grad = True

    elif strategy == "transformer_only":
        for name, p in model.named_parameters():
            if "encoder" in name or "decoder" in name:
                p.requires_grad = True

    elif strategy == "head_only":
        for p in model.class_labels_classifier.parameters():
            p.requires_grad = True
        for p in model.bbox_predictor.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown finetune strategy: {strategy}")
