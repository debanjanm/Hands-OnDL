# Hands-OnDL

Hands-on deep learning with PyTorch — structured for learning from scratch.

## Structure

```
01_basics/
  01_tensors.ipynb                 — tensor ops, shapes, dtype, broadcasting, GPU
  02_autograd.ipynb                — requires_grad, backward, zero_grad, manual GD
  03_datasets_dataloaders.ipynb    — Dataset, DataLoader, train/val split

02_neural_networks/
  PyTorch_Regression_California_Housing.ipynb   — tabular regression, MSE loss
  IMDB-TFIDF-FNN-PyTorch.py                     — sentiment classification, TF-IDF + FNN

03_training_loop/
  01_manual_training_loop.ipynb    — 5-step loop, train/eval modes, checkpointing
  02_losses_and_optimizers.ipynb   — MSE/BCE/CE, SGD/Adam/AdamW, LR schedulers

── Specializations ──────────────────────────────────────

04_nlp/
  RNN-PyTorch.py                   — RNN text classifier on 20 Newsgroups

05_cv/
  01_cnn_mnist.ipynb               — Conv2d, MaxPool2d, MNIST digit classification
```

## Prerequisites

```
pip install torch torchvision scikit-learn pandas kagglehub matplotlib
```

## Recommended order

`01_basics` → `02_neural_networks` → `03_training_loop` → then pick a specialization: `04_nlp` or `05_cv`
