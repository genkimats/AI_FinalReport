import os
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# Load data
def load_data(folder):
    categories = ['astronomy', 'psychology', 'sociology']
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
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

# Training and evaluation
def train_and_evaluate(texts, labels, k=5):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts).toarray()
    y = np.array(labels)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics = []
    confusion_matrices = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

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

    return avg_metrics, avg_conf_matrix, avg_normalized_conf_matrix, model, vectorizer

# Main
if __name__ == "__main__":
    folder = "abstracts"
    texts, labels = load_data(folder)

    avg_metrics, avg_conf_matrix, avg_normalized_conf_matrix, trained_model, tfidf_vectorizer = train_and_evaluate(texts, labels, k=5)

    print("Average Metrics:")
    print(avg_metrics)
    print("\nAverage Confusion Matrix (Counts):")
    print(avg_conf_matrix)
    print("\nAverage Confusion Matrix (Ratios):")
    print(avg_normalized_conf_matrix)

    # Save model and vectorizer
    torch.save(trained_model.state_dict(), "FFN_5layers.pth")
    joblib.dump(tfidf_vectorizer, "FFN_5layers_tfidf_vectorizer.pkl")
