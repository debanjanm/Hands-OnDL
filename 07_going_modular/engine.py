"""Training and evaluation loops."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.train()
    total_loss = correct = n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return total_loss / len(loader), correct / n


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = correct = n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item()
        correct += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return total_loss / len(loader), correct / n


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: str,
    scheduler=None,
) -> Dict[str, list]:
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = eval_epoch(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"Epoch {epoch+1:3d}/{epochs}  "
              f"train_loss={tl:.4f} train_acc={ta:.4f}  "
              f"val_loss={vl:.4f} val_acc={va:.4f}")
    return history
