import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import softmax
import re

# Define custom dataset
class AbstractsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text)
        tokens = tokens[:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Tokenizer: Simple word-to-index mapping
def simple_tokenizer(text, word2idx):
    return [word2idx.get(word, 0) for word in text.split()]

# CNN model for text classification
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, num_filters, filter_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_size)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch_size, 1, seq_len, embed_size)
        convs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # Apply conv layers
        pools = [torch.max(conv, 2)[0] for conv in convs]  # Max pooling
        out = torch.cat(pools, 1)  # Concatenate pooled features
        out = self.fc(out)  # Fully connected layer
        return out

# Load and preprocess data
folder_path = "abstracts"
classes = ["astronomy", "psychology", "sociology"]
texts, labels = [], []

for label, cls in enumerate(classes):
    file_path = os.path.join(folder_path, f"{cls}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.read().split("\n")
        lines = [preprocess_text(line) for line in lines]
        texts.extend(lines)
        labels.extend([label] * len(lines))

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Build vocabulary
word2idx = {word: idx + 1 for idx, word in enumerate(set(" ".join(texts).split()))}
vocab_size = len(word2idx) + 1

# Hyperparameters
embed_size = 128
num_filters = 100
filter_sizes = [3, 4, 5]
num_classes = len(classes)
max_len = 200
batch_size = 32
epochs = 10
lr = 0.001
k_folds = 5

# Tokenize data
texts_tokenized = [simple_tokenizer(text, word2idx) for text in texts]

def train_and_evaluate_model(train_texts, train_labels, val_texts, val_labels):
    train_dataset = AbstractsDataset(train_texts, train_labels, lambda x: x, max_len)
    val_dataset = AbstractsDataset(val_texts, val_labels, lambda x: x, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(vocab_size, embed_size, num_classes, num_filters, filter_sizes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            preds = torch.argmax(softmax(outputs, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return accuracy, precision, recall, f1

# Perform k-fold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(texts_tokenized)):
    print(f"\nFold {fold+1}/{k_folds}")
    train_texts = [texts_tokenized[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts_tokenized[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    accuracy, precision, recall, f1 = train_and_evaluate_model(train_texts, train_labels, val_texts, val_labels)
    fold_metrics.append((accuracy, precision, recall, f1))
    print(f"Fold {fold+1} Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")

# Aggregate metrics
avg_accuracy = np.mean([m[0] for m in fold_metrics])
avg_precision = np.mean([m[1] for m in fold_metrics])
avg_recall = np.mean([m[2] for m in fold_metrics])
avg_f1 = np.mean([m[3] for m in fold_metrics])

print("\nAggregated Metrics:")
print(f"  Average Accuracy: {avg_accuracy:.4f}")
print(f"  Average Precision: {avg_precision:.4f}")
print(f"  Average Recall: {avg_recall:.4f}")
print(f"  Average F1-score: {avg_f1:.4f}")