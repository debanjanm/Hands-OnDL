import os
import re
import string

import kagglehub
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()


def load_data(test_size=0.2, random_state=42):
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    df = pd.read_csv(os.path.join(path, "IMDB Dataset.csv"))
    df['text'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return train_test_split(df, test_size=test_size, random_state=random_state)


def split_train(train_df, test_size=0.2, random_state=42):
    return train_test_split(
        train_df['text'], train_df['label'],
        test_size=test_size, random_state=random_state
    )


class SentimentFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1),          nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train(model, X_train, y_train, X_val, y_val, lr=1e-3, batch_size=32, epochs=5):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train, y_train.unsqueeze(1).float()), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val.unsqueeze(1).float()),   batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += ((model(xb) >= 0.5).float() == yb).sum().item()
                total += yb.size(0)
        print(f"Epoch {epoch}/{epochs}  loss={total_loss/len(train_loader):.4f}  val_acc={correct/total:.4f}")


def evaluate(model, X_test, y_test, batch_size=32):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(X_test, y_test.unsqueeze(1).float()), batch_size=batch_size):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += ((model(xb) >= 0.5).float() == yb).sum().item()
            total += yb.size(0)
    print(f"Test Accuracy: {correct/total:.4f}")


def main():
    train_df, test_df = load_data()
    X_train_text, X_val_text, y_train, y_val = split_train(train_df)

    vec = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train = torch.tensor(vec.fit_transform(X_train_text).toarray(), dtype=torch.float32)
    X_val   = torch.tensor(vec.transform(X_val_text).toarray(),       dtype=torch.float32)
    X_test  = torch.tensor(vec.transform(test_df['text']).toarray(),  dtype=torch.float32)
    y_test  = torch.tensor(test_df['label'].values)

    model = SentimentFNN(input_dim=X_train.shape[1])
    train(model, X_train, torch.tensor(y_train.values), X_val, torch.tensor(y_val.values))
    evaluate(model, X_test, y_test)

    sample = "This film was a fantastic journey into character-driven storytelling."
    x = torch.tensor(vec.transform([clean_text(sample)]).toarray(), dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        sentiment = 'Positive' if model(x).item() >= 0.5 else 'Negative'
    print(f"Review: {sample}\nSentiment: {sentiment}")


if __name__ == '__main__':
    main()
