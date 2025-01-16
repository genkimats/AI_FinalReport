import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch.nn.functional as F

# Load data
def load_data(folder):
    categories = ['astronomy', 'sociology', 'psychology']
    texts, labels = [], []
    for i, category in enumerate(categories):
        file_path = os.path.join(folder, f"processed_{category}.txt")
        with open(file_path, 'r', encoding='utf-8') as f:
            abstracts = f.read().split('\n')
            texts.extend(abstracts)
            labels.extend([i] * len(abstracts))
    return texts, labels

# Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = F.softmax(x, dim=1)  # Apply softmax to output logits across class dimension
        return x

# Training function
def train_model(hyperparams, X_train, y_train, X_val, y_val, input_dim):
    model = FeedforwardNN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    criterion = nn.CrossEntropyLoss()

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=hyperparams['batch_size'], shuffle=True)

    for epoch in range(hyperparams['epochs']):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{hyperparams['epochs']} - Loss: {total_loss:.4f}")

    return model

# Training and evaluation
def train_and_evaluate(texts, labels, k=5):
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    class_labels = ['Astronomy', 'Sociology', 'Psychology']
    metrics = []
    confusion_matrices = []

    hyperparams = {
        'lr': 0.01,
        'batch_size': 32,
        'epochs': 20
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y), 1):
        print(f"Fold {fold}/{k}")

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

        # Train model
        best_model = train_model(hyperparams, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_train.shape[1])

        # Final Evaluation
        best_model.eval()
        with torch.no_grad():
            val_outputs = best_model(X_val_tensor)
            val_preds = torch.argmax(val_outputs, axis=1).numpy()

        acc = accuracy_score(y_val, val_preds)
        report = classification_report(y_val, val_preds, target_names=class_labels, output_dict=True, zero_division=0)

        # Print per-class metrics for this fold
        print(f"Fold {fold} Class-wise Metrics:")
        for label in class_labels:
            precision = report[label]['precision']
            recall = report[label]['recall']
            f1_score = report[label]['f1-score']
            print(f"  {label}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1_score:.4f}")
        print()

        metrics.append({
            "accuracy": acc,
            "precision": report['macro avg']['precision'],
            "recall": report['macro avg']['recall'],
            "f1_score": report['macro avg']['f1-score']
        })

        conf_matrix = confusion_matrix(y_val, val_preds)
        confusion_matrices.append(conf_matrix)

    # Aggregate confusion matrix
    aggregated_conf_matrix = sum(confusion_matrices)
    normalized_conf_matrix = aggregated_conf_matrix / aggregated_conf_matrix.sum(axis=1, keepdims=True)

    print("\nAggregated Confusion Matrix (Counts):")
    print(aggregated_conf_matrix)

    print("\nAggregated Confusion Matrix (Ratios per Field, 4 Decimals):")
    print(np.round(normalized_conf_matrix, decimals=4))

    # Average metrics
    avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}

    print("\nAverage Metrics:")
    print(avg_metrics)

    return avg_metrics

# Main
if __name__ == "__main__":
    folder = "abstracts_processed_1000"
    texts, labels = load_data(folder)

    avg_metrics = train_and_evaluate(texts, labels, k=5)
