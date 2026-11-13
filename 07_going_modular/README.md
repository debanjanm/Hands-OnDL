# Going Modular

Refactor of the CIFAR-10 training code from `05_cv/` into importable scripts. Each file has one job.

## Structure

```
07_going_modular/
  model.py       — model definitions (tiny_cnn, resnet18_transfer)
  engine.py      — train_epoch, eval_epoch, train loop
  data_setup.py  — DataLoader factories (CIFAR-10, ImageFolder)
  utils.py       — save/load checkpoint, plot_history
  train.py       — CLI entry point (argparse)
```

## Run

```bash
# Train tiny CNN for 10 epochs
python train.py --model tiny_cnn --epochs 10

# Transfer learning with frozen ResNet-18
python train.py --model resnet18 --epochs 5 --freeze_backbone

# Fine-tune ResNet-18 (unfrozen)
python train.py --model resnet18 --epochs 10 --lr 1e-4
```

## Why modular?

| Notebook | Script |
|---|---|
| Re-run whole notebook to change one param | `python train.py --lr 1e-4` |
| Hard to import pieces | `from engine import train` |
| Mixed logic + display | Separated concerns |
| Hard to version control outputs | Clean diffs on `.py` files |

## Reuse in notebooks

```python
import sys
sys.path.insert(0, '../07_going_modular')

from model import tiny_cnn
from engine import train
from data_setup import get_cifar10
from utils import save_checkpoint, plot_history
```
