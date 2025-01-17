import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load data
def load_data(input_dir):
    """
    Load abstract data from input directory.

    Args:
        input_dir (string): Input directory containing processed abstracts.

    Returns:
        list(string), list(string): list of abstracts, list of labels
    """
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
class FeedforwardNN(nn.Module): # Define a Feedforward Neural Network class by inheriting from nn.Module
    def __init__(self, input_dim):
        """
        Initialize a Feedforward Neural Network. 
        The network consists of 3 fully connected layers with ReLU activation

        Args:
            input_dim (int): Input dimension of the network (number of features)
        """
        super(FeedforwardNN, self).__init__()
        
        # First group of layers
        self.fc1 = nn.Linear(input_dim, 256) # Fully connected layer with 256 output features
        self.bn1 = nn.BatchNorm1d(256) # Batch normalization layer
        self.relu1 = nn.ReLU() # ReLU activation function for non-linearity
        self.dropout1 = nn.Dropout(0.3) # Dropout layer with 30% probability
        
        # Second group of layers
        self.fc2 = nn.Linear(256, 128) # Fully connected layer with 128 output features
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Third group of layers
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        
        # Output layer
        self.fc4 = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1) # Softmax activation function for multi-class classification

    def forward(self, x):
        """
        Forward pass of the network.
        This function defines how input data is processed through the network.
        It is not called directly, but by passing input data to the model object.

        Args:
            x (Tensor): Input data for the network.

        Returns:
            Tensor: Output of the network representing either Astronomy, Psychology, or Sociology.
        """
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
        x = self.softmax(x)
        return x

# Training function
def train_model(hyperparams, X_train, y_train, X_val, y_val, input_dim):
    model = FeedforwardNN(input_dim) # Initialize the model
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr']) # Adam optimizer for parameter updates
    criterion = nn.CrossEntropyLoss() 
    
    # Create a DataLoader for training data
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=hyperparams['batch_size'], shuffle=True) 
    
    # Training loop
    for epoch in range(hyperparams['epochs']):
        model.train()
        total_loss = 0
        # Iterate over batches of training data
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad() 
            outputs = model(batch_X) # Forward pass (calls model.forward defined above)
            loss = criterion(outputs, batch_y) # Compute loss for the batch
            loss.backward() # Compute gradients
            optimizer.step() # Update model parameters
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{hyperparams['epochs']} - Loss: {total_loss:.4f}")

    return model

# Training and evaluation
def train_and_evaluate(texts, labels, k=5):
    """
    Train and evaluate a Feedforward Neural Network model on the given data.
    Uses k-fold cross-validation for evaluation.

    Args:
        texts (list(string)): List of abstracts.
        labels (list(string)): List of labels.
        k (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
        dict(string, np.float64): Average metrics over all folds (accuracy, precision, recall, f1_score).
    """
    y = np.array(labels) # Convert labels to numpy array
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42) # Initialize stratified k-fold cross-validation 
    
    # Class labels for the dataset
    class_labels = ['Astronomy', 'Sociology', 'Psychology']
    metrics = []
    confusion_matrices = []
    
    # Hyperparameters for training
    hyperparams = {
        'lr': 0.01,
        'batch_size': 32,
        'epochs': 20
    }
    
    # Iterate over folds for cross-validation
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

        # Convert to tensors to use in training
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        # Train model on the training data
        best_model = train_model(hyperparams, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_train.shape[1])

        # Final Evaluation 
        best_model.eval()
        with torch.no_grad():
            val_outputs = best_model(X_val_tensor)
            val_preds = torch.argmax(val_outputs, axis=1).numpy()

        acc = accuracy_score(y_val, val_preds) # Compute accuracy using sklearn's accuracy_score
        # Compute classification report using sklearn's classification_report
        report = classification_report(y_val, val_preds, target_names=class_labels, output_dict=True, zero_division=0)

        # Print per-class metrics for this fold
        print(f"Fold {fold} Class-wise Metrics:")
        for label in class_labels:
            precision = report[label]['precision']
            recall = report[label]['recall']
            f1_score = report[label]['f1-score']
            print(f"  {label}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1_score:.4f}")
        print()

        metrics.append({ # Append metrics for this fold to the list
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
    folder = "abstracts_processed"
    texts, labels = load_data(folder)

    avg_metrics = train_and_evaluate(texts, labels, k=5)
