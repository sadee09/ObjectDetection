from pathlib import Path

import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            collate_fn=self.collate_fn,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            collate_fn=self.collate_fn,
        )

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        self.save_dir = Path(config["training"]["save_dir"])
        self.save_dir.mkdir(exist_ok=True)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for images, targets in self.train_loader:
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        total_loss = 0

        with torch.inference_mode():
            for images, targets in self.val_loader:
                images = [image.to(self.device) for image in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

        return total_loss / len(self.val_loader)
