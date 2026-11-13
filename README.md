# Hands-OnDL

Hands-on deep learning with PyTorch — structured for learning from scratch.

## Structure

```
01_basics/
  01_tensors.ipynb                 — tensor ops, shapes, dtype, broadcasting, GPU
  02_autograd.ipynb                — requires_grad, backward, zero_grad, manual GD
  03_datasets_dataloaders.ipynb    — Dataset, DataLoader, train/val split
  04_custom_datasets.ipynb         — ImageFolder, CSV-backed Dataset, transforms v2, num_workers

02_neural_networks/
  PyTorch_Regression_California_Housing.ipynb   — tabular regression, MSE loss
  IMDB-TFIDF-FNN-PyTorch.py                     — sentiment classification, TF-IDF + FNN
  03_end_to_end_workflow.ipynb     — 6-step DL template: data → model → loss → train → eval → save

03_training_loop/
  01_manual_training_loop.ipynb    — 5-step loop, train/eval modes, checkpointing
  02_losses_and_optimizers.ipynb   — MSE/BCE/CE, SGD/Adam/AdamW, LR schedulers

── Specializations ──────────────────────────────────────

04_nlp/
  01_rnn_text_classification.ipynb — RNN text classifier on 20 Newsgroups
  02_lstm_text_classification.ipynb — LSTM, GRU, BiLSTM; pack_padded_sequence, grad clipping

05_cv/
  01_cnn_mnist.ipynb               — Conv2d, MaxPool2d, MNIST digit classification
  02_cifar10_transfer_learning.ipynb — ResNet-18, feature extraction vs fine-tuning, differential LRs

── Production Skills ─────────────────────────────────────

06_experiment_tracking/
  01_tensorboard.ipynb             — SummaryWriter, scalars/images/graph/hparams, compare runs

07_going_modular/
  model.py / engine.py / data_setup.py / utils.py / train.py
                                   — refactor notebooks into importable scripts + argparse CLI
```

## Prerequisites

```
pip install torch torchvision scikit-learn pandas kagglehub matplotlib
```

## Recommended order

`01_basics` → `02_neural_networks` → `03_training_loop` → specialization (`04_nlp` or `05_cv`) → `06_experiment_tracking` → `07_going_modular`
