"""Checkpoint and plotting utilities."""

import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def plot_history(history: Dict[str, list], title: str = "") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"],   label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"],   label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
