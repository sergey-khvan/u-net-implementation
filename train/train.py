import albumentations as A
import torch
import torch.nn.functional as F
import yaml
from addict import Dict
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset import EMDataset
from model import Unet


def calc_accuracy(preds, truth, thr):
    num_correct = 0
    num_pixels = 0
    preds = F.sigmoid(preds)
    preds[preds < thr] = 0
    preds[preds >= thr] = 1
    num_correct += (preds == truth).sum()
    num_pixels += torch.numel(preds)
    accuracy = num_correct / num_pixels
    return accuracy


def main():
    with open("./config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg = Dict(cfg)
    torch.manual_seed(42)
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wandb.init(project="u-net")

    # Load data
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(p=0.8, sigma=10),
            A.RandomGridShuffle(grid=(3, 3), p=0.2),
            ToTensorV2(),
        ]
    )

    train_dataset = EMDataset(
        img_dir=cfg.IMG_TRAIN_DIR, mask_dir=cfg.MASK_TRAIN_DIR, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = EMDataset(
        img_dir=cfg.IMG_TEST_DIR, mask_dir=cfg.MASK_TEST_DIR, transform=ToTensorV2()
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Load model
    if cfg.num_classes == 2:
        out_channels = 1
    else:
        out_channels = cfg.num_classes

    model = Unet(in_channels=cfg.in_channels, out_channels=out_channels)
    model.to(device)

    # Loss and Optimizer
    if cfg.num_classes == 2:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum
    )

    if cfg.load_from_checkpoint:
        checkpoint = torch.load(cfg.CHECKPOINTS_DIR)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch in range(cfg.num_epochs):
        train_running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        # Train
        model.train()
        for idx, (data, mask) in loop:
            data = data.float().to(device)
            mask = mask.float().unsqueeze(1).to(device)

            preds = model(data)
            loss = loss_fn(preds, mask)
            train_running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch:[{epoch}/{cfg.num_epochs}](Train)")
            loop.set_postfix(train_loss=loss.item())

        train_loss = train_running_loss / (idx + 1)

        # Val
        val_running_loss = 0.0
        run_accuracy = 0
        model.eval()
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
        with torch.no_grad():
            for idx, (data, mask) in loop:
                data = data.float().to(device)
                mask = mask.float().unsqueeze(1).to(device)

                preds = model(data)
                loss = loss_fn(preds, mask)
                val_running_loss += loss.item()
                accuracy = calc_accuracy(preds, mask, cfg.THR)
                run_accuracy += accuracy.item()
                loop.set_description(f"Epoch:[{epoch}/{cfg.num_epochs}](Val)")
                loop.set_postfix(val_loss=loss.item(), acc=accuracy.item())

        preds = F.sigmoid(preds)
        preds[preds <= cfg.THR] = 0.0
        preds[preds > cfg.THR] = 1.0

        pred_mask = wandb.Image(preds, caption="pred_masks")
        true_mask = wandb.Image(mask, caption="true_masks")

        val_loss = val_running_loss / (idx + 1)
        mean_accuracy = run_accuracy / (idx + 1)

        if epoch % 20 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                cfg.CHECKPOINTS_DIR,
            )

        wandb.log(
            {
                "true masks": true_mask,
                "pred masks": pred_mask,
                "train loss": train_loss,
                "val loss": val_loss,
                "accuracy(val)": mean_accuracy,
            }
        )

    wandb.finish()
    torch.save(model.state_dict(), f"weights/weights_{cfg.num_epochs}.pt")


if __name__ == "__main__":
    main()
