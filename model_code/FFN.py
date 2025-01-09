import os
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# Load data
def load_data(folder):
    categories = ['astronomy', 'sociology', 'psychology']
    texts, labels = [], []
    for i, category in enumerate(categories):
        file_path = os.path.join(folder, f"{category}.txt")
        with open(file_path, 'r', encoding='utf-8') as f:
            abstracts = f.read().split('\n')
            texts.extend(abstracts)
            labels.extend([i] * len(abstracts))
    return texts, labels

# Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Training and evaluation
def train_and_evaluate(texts, labels, k=5):
    y = np.array(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    metrics = []
    confusion_matrices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y), 1):
        # Split the data
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Vectorize within the fold
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(train_texts).toarray()
        X_val = vectorizer.transform(val_texts).toarray()

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        # Initialize the model
        model = FeedforwardNN(input_dim=X_train.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_preds = torch.argmax(val_outputs, axis=1).numpy()

        acc = accuracy_score(y_val, val_preds)
        report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
        metrics.append({
            "accuracy": acc,
            "precision": report['macro avg']['precision'],
            "recall": report['macro avg']['recall'],
            "f1_score": report['macro avg']['f1-score']
        })

        conf_matrix = confusion_matrix(y_val, val_preds)
        normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        confusion_matrices.append((conf_matrix, normalized_conf_matrix))

        print(f"Fold {fold} Confusion Matrix (Counts):\n{conf_matrix}")
        print(f"Fold {fold} Confusion Matrix (Ratios):\n{normalized_conf_matrix}\n")

    # Average metrics
    avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    avg_conf_matrix = sum(cm[0] for cm in confusion_matrices)
    avg_normalized_conf_matrix = avg_conf_matrix / avg_conf_matrix.sum(axis=1, keepdims=True)

    return avg_metrics, avg_conf_matrix, avg_normalized_conf_matrix

# Main
if __name__ == "__main__":
    folder = "abstracts_processed"
    texts, labels = load_data(folder)

    avg_metrics, avg_conf_matrix, avg_normalized_conf_matrix = train_and_evaluate(texts, labels, k=5)

    print("Average Metrics:")
    print(avg_metrics)
    print("\nAverage Confusion Matrix (Counts):")
    print(avg_conf_matrix)
    print("\nAverage Confusion Matrix (Ratios):")
    print(avg_normalized_conf_matrix)
