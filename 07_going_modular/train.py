"""CLI entry point for training.

Usage:
    python train.py --model tiny_cnn --epochs 20 --lr 1e-3 --batch_size 128
    python train.py --model resnet18 --epochs 10 --freeze_backbone
"""

import argparse

import torch

import data_setup
import engine
import model as model_module
import utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="tiny_cnn", choices=["tiny_cnn", "resnet18"])
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--batch_size",      type=int,   default=128)
    p.add_argument("--val_split",       type=float, default=0.1)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--save_path",       default="models/checkpoint.pth")
    p.add_argument("--data_dir",        default="./data")
    p.add_argument("--num_workers",     type=int,   default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = data_setup.get_cifar10(
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    if args.model == "tiny_cnn":
        net = model_module.tiny_cnn(num_classes=10).to(device)
    else:
        net = model_module.resnet18_transfer(
            num_classes=10, freeze_backbone=args.freeze_backbone
        ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    history = engine.train(
        net, train_loader, val_loader,
        optimizer, criterion, args.epochs, device, scheduler
    )

    utils.plot_history(history, title=f"{args.model} on CIFAR-10")
    utils.save_checkpoint(net, args.save_path)

    # Final test evaluation
    test_loss, test_acc = engine.eval_epoch(net, test_loader, criterion, device)
    print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
