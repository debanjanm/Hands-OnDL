### Module: data_pipeline.py
# data_pipeline.py
# data_pipeline.py
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
import kagglehub
import os

class DataPipeline:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def clean_text(text):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        text = text.lower()
        return text

    def load_data(self):
        # Download the dataset using kagglehub
        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        csv_file = os.path.join(path, "IMDB Dataset.csv")

        # Load the dataset
        df = pd.read_csv(csv_file)
        # Clean the reviews
        df['text'] = df['review'].apply(self.clean_text)
        # Map sentiments to binary labels
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

        # Split the data into training and testing sets
        train_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )
        return train_df, test_df

    def split_train(self, train_df):
        return train_test_split(
            train_df['text'], train_df['label'],
            test_size=self.test_size, random_state=self.random_state
        )

### Module: feature_pipeline.py
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

class FeaturePipeline:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    def fit_transform(self, texts):
        X = self.vectorizer.fit_transform(texts)
        return torch.tensor(X.toarray(), dtype=torch.float32)

    def transform(self, texts):
        X = self.vectorizer.transform(texts)
        return torch.tensor(X.toarray(), dtype=torch.float32)

### Module: model.py
import torch
import torch.nn as nn

class SentimentFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

### Module: train_pipeline.py
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, lr=1e-3, batch_size=32, epochs=5, device=None):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def train(self, X_train, y_train, X_val, y_val):
        train_ds = TensorDataset(X_train, y_train.unsqueeze(1).float())
        val_ds   = TensorDataset(X_val, y_val.unsqueeze(1).float())
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            # Validation
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = self.model(xb)
                    predicted = (preds >= 0.5).float()
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()
            val_acc = correct / total
            print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

    def evaluate(self, X_test, y_test):
        test_ds = TensorDataset(X_test, y_test.unsqueeze(1).float())
        test_loader = DataLoader(test_ds, batch_size=self.batch_size)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
                predicted = (preds >= 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        print(f"Test Accuracy: {correct/total:.4f}")

### Module: inference_pipeline.py
import torch
# from data_pipeline import DataPipeline

class InferencePipeline:
    def __init__(self, model, vectorizer, device=None):
        self.model = model
        self.vectorizer = vectorizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict(self, text: str) -> str:
        dp = DataPipeline()
        cleaned = dp.clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        x = torch.tensor(vec.toarray(), dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
        return 'Positive' if pred.item() >= 0.5 else 'Negative'

### Main script: main.py
# from data_pipeline import DataPipeline
# from feature_pipeline import FeaturePipeline
# from model import SentimentFNN
# from train_pipeline import Trainer
# from inference_pipeline import InferencePipeline
import torch


def main():
    # Load and split
    dp = DataPipeline()
    train_df, test_df = dp.load_data()
    X_train_text, X_val_text, y_train, y_val = dp.split_train(train_df)

    # Feature transform
    fp = FeaturePipeline()
    X_train = fp.fit_transform(X_train_text)
    X_val   = fp.transform(X_val_text)
    X_test  = fp.transform(test_df['text'])
    y_test  = torch.tensor(test_df['label'].values)

    # Model, trainer
    model = SentimentFNN(input_dim=X_train.shape[1])
    trainer = Trainer(model=model)
    trainer.train(X_train, torch.tensor(y_train.values), X_val, torch.tensor(y_val.values))
    trainer.evaluate(X_test, y_test)

    # Inference
    inf = InferencePipeline(model, fp.vectorizer)
    sample = "This film was a fantastic journey into character-driven storytelling."
    print(f"Review: {sample}\nSentiment: {inf.predict(sample)}")

if __name__ == '__main__':
    main()
