import argparse

import torch
import yaml
from data.dataset import AerialDataset
from loguru import logger
from models.detector import AerialDetector
from training.trainer import Trainer


def train_model(config_path):
    logger.info(f"[INFO] Loading configuration from : {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("[INFO] Configurations loaded successfully")

    classes = config["data"]["classes"]
    train_dataset = AerialDataset(
        config["data"]["train_path"], classes=classes, train=True
    )
    val_dataset = AerialDataset(
        config["data"]["val_path"], classes=classes, train=False
    )
    logger.info(f"[INFO] Number of images in train set: {len(train_dataset)}")
    logger.info(f"[INFO] Number of images in val/test set: {len(val_dataset)}")

    model = AerialDetector(len(config["data"]["classes"]))
    logger.info("Model loaded successfully")

    trainer = Trainer(model, train_dataset, val_dataset, config)

    num_epochs = config.get("training", {}).get("epochs", 5)
    for epoch in range(num_epochs):
        logger.info(f"[INFO] Training | Epoch: {epoch + 1} / {num_epochs}")
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        logger.info(
            f"[INFO] Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        torch.save(model.state_dict(), trainer.save_dir / f"model_epoch_{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="Path to config files containing training related arguments.",
        type=str,
    )

    args = parser.parse_args()
    config_path = args.config_path
    train_model(config_path=config_path)
