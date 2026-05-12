import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# 1. Load and Preprocess Data
# -------------------------
# Load the 20 Newsgroups dataset (remove headers/footers/quotes for cleaner text)
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = newsgroups.data
labels = newsgroups.target
num_classes = len(newsgroups.target_names)

# Basic tokenizer: lowercase, remove non-alphanumeric characters, and split on whitespace
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

# Build vocabulary from all texts (you may opt to build from training data only)
vocab = {"<PAD>": 0, "<UNK>": 1}
for text in texts:
    for token in tokenize(text):
        if token not in vocab:
            vocab[token] = len(vocab)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Function to convert text to sequence of token IDs
def text_to_sequence(text, vocab, max_len):
    tokens = tokenize(text)
    seq = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    # Truncate or pad sequence to fixed max_len
    if len(seq) < max_len:
        seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

max_seq_length = 100  # fixed sequence length for each text sample

# Convert all texts to padded sequences
sequences = [text_to_sequence(text, vocab, max_seq_length) for text in texts]
sequences = np.array(sequences, dtype=np.int64)
labels = np.array(labels, dtype=np.int64)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Convert numpy arrays to torch tensors
X_train = torch.tensor(X_train)
X_test  = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test  = torch.tensor(y_test)

# -------------------------
# 2. Device Selection: Use MPS (Apple M2) if available, otherwise CPU
# -------------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# -------------------------
# 3. Define the RNN Model for Text Classification
# -------------------------
class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # shape: [batch_size, seq_len, embed_dim]
        # Initialize hidden state on the same device as x
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(device)
        out, _ = self.rnn(x, h0)  # out: [batch_size, seq_len, hidden_size]
        # Use the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# -------------------------
# 4. Hyperparameters and Model Setup
# -------------------------
embed_dim = 64
hidden_size = 128
num_layers = 1
num_epochs = 5         # For demonstration; consider more epochs for real training
batch_size = 64
learning_rate = 0.001

model = RNNTextClassifier(vocab_size, embed_dim, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------
# 5. Training Loop
# -------------------------
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(num_epochs):
    for inputs, labels_batch in train_loader:
        inputs, labels_batch = inputs.to(device), labels_batch.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# -------------------------
# 6. Inference on Test Data
# -------------------------
model.eval()
with torch.no_grad():
    test_inputs = X_test.to(device)
    outputs = model(test_inputs)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted.cpu() == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
