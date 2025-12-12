import os
import torch


class Config:
    # Paths
    PROJECT_ROOT = "."
    DATA_ROOT = "/scratch/hbm9834/Computer_vision/Assignment3/cv_data_hw2"
    MATCHED_ANN_DIR = "/scratch/hbm9834/Computer_vision/Assignment3/matched_annotations"
    OUTPUT_DIR = "/scratch/hbm9834/Computer_vision/Assignment3/outputs"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

    # Training hyperparameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    WARMUP_STEPS = 0
    NUM_WORKERS = 4

    # Image / model config
    IMAGE_HEIGHT = 800
    IMAGE_WIDTH = 800
    MODEL_NAME = "facebook/detr-resnet-50"
    NUM_CLASSES = 6  # Unknown=0, person=1, car=2, other_vehicle=3, other_object=4, bike=5

    # Split config
    TRAIN_SPLIT = 0.8
    RANDOM_SEED = 42

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def create_dirs():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.VIS_DIR, exist_ok=True)
