import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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

# Custom Dataset for BERT
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# BERT-based Classification Model
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

# Training function
def train_model(hyperparams, train_loader, val_loader, model, device):
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    best_accuracy = 0

    for epoch in range(hyperparams['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{hyperparams['epochs']} - Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, axis=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model, best_accuracy

# Training and evaluation
def train_and_evaluate(texts, labels, bert_model_name, k=5):
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    max_length = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y = np.array(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    metrics = []
    confusion_matrices = []

    hyperparams = {
        'lr': 2e-5,
        'batch_size': 16,
        'epochs': 4
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y), 1):
        print(f"Fold {fold}/{k}")

        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TextDataset(train_texts, y_train, tokenizer, max_length)
        val_dataset = TextDataset(val_texts, y_val, tokenizer, max_length)

        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

        model = BERTClassifier(bert_model_name, num_classes=3)
        best_model, best_accuracy = train_model(hyperparams, train_loader, val_loader, model, device)

        print(f"Fold {fold} Best Accuracy: {best_accuracy:.4f}")

    print("Training Complete.")

# Main
if __name__ == "__main__":
    folder = "abstracts_lemmatization"
    texts, labels = load_data(folder)

    bert_model_name = "bert-base-uncased"
    train_and_evaluate(texts, labels, bert_model_name, k=5)
