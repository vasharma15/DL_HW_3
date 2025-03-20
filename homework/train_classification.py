import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from models import load_model, save_model
from datasets.classification_dataset import load_data
import torch.nn as nn


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
    train_data = load_data("../classification_data/train", shuffle=True, batch_size=batch_size, num_workers=0)
    val_data = load_data("../classification_data/val", shuffle=False)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}
    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # Forward pass
            outputs = model(img)

            loss = loss_func(outputs, label)
            # Backpropagation
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update weights
            # Compute training accuracy
            preds = outputs.argmax(dim=1)
            train_acc = (preds == label).float().mean().item()
            metrics["train_acc"].append(train_acc)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            correct = 0
            total = 0
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                # Forward pass for validation data
                outputs = model(img)
                # Compute validation accuracy
                preds = outputs.argmax(dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)

            val_acc = correct / total
            metrics["val_acc"].append(val_acc)

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar("train_acc", epoch_train_acc, epoch)
        logger.add_scalar("val_acc", epoch_val_acc, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    save_model(model)

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
