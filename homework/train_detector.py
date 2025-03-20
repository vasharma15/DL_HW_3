import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from models import load_model, save_model
from datasets.road_dataset import load_data
from tqdm import tqdm
import torch.nn.functional as F


def compute_loss(inputs, targets):
    logits, depth_pred = inputs
    seg_targets = targets['track']
    depth_targets = targets['depth']

    ce_loss = F.cross_entropy(logits, seg_targets)

    probs = F.softmax(logits, dim=1)
    batch_size = probs.shape[0]
    num_classes = probs.shape[1]

    dice_loss = 0
    for c in range(num_classes):
        target_one_hot = (seg_targets == c).float()

        pred_probs = probs[:, c]

        intersection = (pred_probs * target_one_hot).sum(dim=(1, 2))
        union = pred_probs.sum(dim=(1, 2)) + target_one_hot.sum(dim=(1, 2))

        dice_coeff = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_loss += (1.0 - dice_coeff).mean()

    dice_loss /= num_classes

    seg_loss = 0.5 * ce_loss + 0.5 * dice_loss

    depth_loss = F.l1_loss(depth_pred, depth_targets)

    lambda_seg = 1.5
    lambda_depth = 1.0

    total_loss = lambda_seg * seg_loss + lambda_depth * depth_loss

    losses_dict = {
        'seg_loss': seg_loss,
        'depth_loss': depth_loss,
        'total_loss': total_loss,
        'ce_loss': ce_loss,
        'dice_loss': dice_loss
    }

    return total_loss, losses_dict


def train(
        exp_dir: str = "logs",
        model_name: str = "classifier",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 2024,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Shivam", device)
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)
    model = load_model(model_name, **kwargs)
    model = model.to(device)

    model.train()

    train_data = load_data("../drive_data/train", shuffle=True, batch_size=batch_size, num_workers=0)
    val_data = load_data("../drive_data/val", shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    metrics = {"train_acc": [], "val_acc": []}
    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()

        model.train()
        train_loss = 0
        train_seg_loss = 0
        train_depth_loss = 0
        progress_bar = tqdm(train_data, desc=f"Epoch {epoch + 1}/{num_epoch} [Train]")

        for batch in progress_bar:
            images = batch['image'].to(device)
            targets = {
                'track': batch['track'].to(device),
                'depth': batch['depth'].to(device)
            }

            optimizer.zero_grad()
            predictions = model(images)

            loss, losses_dict = compute_loss(predictions, targets)

            loss.backward()
            optimizer.step()

            train_loss += losses_dict['total_loss']
            train_seg_loss += losses_dict['seg_loss']
            train_depth_loss += losses_dict['depth_loss']

            progress_bar.set_postfix(loss=f"{losses_dict['total_loss']:.4f}")

        avg_train_loss = train_loss / len(train_data)
        avg_train_seg_loss = train_seg_loss / len(train_data)
        avg_train_depth_loss = train_depth_loss / len(train_data)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_seg_loss = 0
        val_depth_loss = 0

        with torch.no_grad():
            progress_bar = tqdm(val_data, desc=f"Epoch {epoch + 1}/{num_epoch} [Val]")
            for batch in progress_bar:
                images = batch['image'].to(device)
                targets = {
                    'track': batch['track'].to(device),
                    'depth': batch['depth'].to(device)
                }

                predictions = model(images)
                _, losses_dict = compute_loss(predictions, targets)

                val_loss += losses_dict['total_loss']
                val_seg_loss += losses_dict['seg_loss']
                val_depth_loss += losses_dict['depth_loss']
                progress_bar.set_postfix(loss=f"{losses_dict['total_loss']:.4f}")

        avg_val_loss = val_loss / len(val_data)
        avg_val_seg_loss = val_seg_loss / len(val_data)
        avg_val_depth_loss = val_depth_loss / len(val_data)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epoch}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, Depth: {avg_train_depth_loss:.4f})")
        print(f"  Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, Depth: {avg_val_depth_loss:.4f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), log_dir / f"{model_name}.th")
            print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
