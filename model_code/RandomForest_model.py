import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import pickle

# Load data
def load_data(folder_path):
    data = []
    labels = []
    for file_name, label in zip(["processed_astronomy.txt", "processed_psychology.txt", "processed_sociology.txt"], ["Astronomy", "Psychology", "Sociology"]):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            abstracts = file.read().strip().split('\n')
            data.extend(abstracts)
            labels.extend([label] * len(abstracts))
    return data, labels

# Load and preprocess data
folder_path = "abstracts_processed_1000"
data, labels = load_data(folder_path)

# Convert to DataFrame
df = pd.DataFrame({"text": data, "label": labels})
y = np.array(df["label"])

# Manual Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aggregated_confusion_matrix = np.zeros((3, 3))  # 3 classes: Astronomy, Psychology, Sociology
fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1s = []

class_labels = ["Astronomy", "Psychology", "Sociology"]

all_test_labels = []
all_predictions = []

for train_index, test_index in skf.split(df["text"], y):
    # Split data into training and testing
    X_train_text = df["text"].iloc[train_index]
    X_test_text = df["text"].iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    
    # Fit vectorizer on training data only
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()
    
    # Train Random Forest model
    model = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=80, min_samples_split=5)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Store predictions and true labels for overall metrics
    all_test_labels.extend(y_test)
    all_predictions.extend(y_pred)
    
    # Compute confusion matrix for this fold and add to aggregated matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    aggregated_confusion_matrix += cm
    
    # Compute metrics for this fold
    fold_accuracies.append(accuracy_score(y_test, y_pred))
    fold_precisions.append(precision_score(y_test, y_pred, average='weighted'))
    fold_recalls.append(recall_score(y_test, y_pred, average='weighted'))
    fold_f1s.append(f1_score(y_test, y_pred, average='weighted'))

# Average metrics across folds
avg_accuracy = np.mean(fold_accuracies)
avg_precision = np.mean(fold_precisions)
avg_recall = np.mean(fold_recalls)
avg_f1 = np.mean(fold_f1s)

# Class-wise metrics across all folds
class_precisions, class_recalls, class_f1s, _ = precision_recall_fscore_support(
    all_test_labels, all_predictions, labels=class_labels
)

# Normalize the aggregated confusion matrix (row-wise)
normalized_cm = aggregated_confusion_matrix.astype('float') / aggregated_confusion_matrix.sum(axis=1)[:, np.newaxis]

# Print overall metrics
print(f"Cross-Validation Accuracy: {avg_accuracy:.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Overall Precision (weighted): {avg_precision:.4f}")
print(f"Overall Recall (weighted): {avg_recall:.4f}")
print(f"Overall F1-Score (weighted): {avg_f1:.4f}")

# Print class-wise metrics
print("\nClass-wise Metrics:")
for label, precision, recall, f1 in zip(class_labels, class_precisions, class_recalls, class_f1s):
    print(f"{label}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# Print confusion matrix
print("\nAggregated Confusion Matrix (Counts):")
for row, label in zip(aggregated_confusion_matrix, class_labels):
    print(f"{label}: {row}")

print("\nNormalized Confusion Matrix (Ratios):")
print("          " + "  ".join(f"{label[:4]}" for label in class_labels))  # Shortened class labels for formatting
for row, label in zip(normalized_cm, class_labels):
    print(f"{label[:4]}   " + "  ".join(f"{val:.4f}" for val in row))

# Train final model on full training data and save
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"]).toarray()
final_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=50, min_samples_split=5)
final_model.fit(X, y)

# Save the model
model_file = "models/random_forest_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(final_model, f)

# Save the vectorizer
vectorizer_file = "models/random_forest_vectorizer.pkl"
with open(vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"Saved model as {model_file} and vectorizer as {vectorizer_file}.")
