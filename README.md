# Hands-OnDL

Hands-on PyTorch — structured for learning from scratch.

## Structure

```
01_basics/
  01_tensors.ipynb                 — tensor ops, shapes, dtype, broadcasting, GPU
  02_autograd.ipynb                — requires_grad, backward, zero_grad, manual GD
  03_datasets_dataloaders.ipynb    — Dataset, DataLoader, train/val split

02_neural_networks/
  PyTorch_Regression_California_Housing.ipynb   — tabular regression, MSE loss
  IMDB-TFIDF-FNN-PyTorch.py                     — sentiment classification, TF-IDF + FNN

03_training_loop/                  — (coming: manual loop, loss/optimizer comparison)

04_nlp/
  RNN-PyTorch.py                   — RNN text classifier on 20 Newsgroups

05_llm_apis/
  chatbot_qa_prompt.md             — prompt template for chatbot QA evaluation
```

## Prerequisites

```
pip install torch scikit-learn pandas kagglehub
```

## Recommended order

`01_basics` → `02_neural_networks` → `03_training_loop` → `04_nlp` → `05_llm_apis`
