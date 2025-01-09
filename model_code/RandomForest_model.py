"""
Random forest

Cross-Validation Accuracy: 0.7967 ± 0.0218
Accuracy: 0.7833
Precision: 0.7869
Recall: 0.7833
F1-Score: 0.7797
Confusion Matrix (Ratio):
[[0.98333333 0.         0.01666667]
 [0.05970149 0.64179104 0.29850746]
 [0.0754717  0.18867925 0.73584906]]
Saved model as random_forest_model.pkl and vectorizer as random_forest_vectorizer.pkl.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle

# Load data
def load_data(folder_path):
    data = []
    labels = []
    for file_name, label in zip(["astronomy.txt", "psychology.txt", "sociology.txt"], ["Astronomy", "Psychology", "Sociology"]):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            abstracts = file.read().strip().split('\n')
            data.extend(abstracts)
            labels.extend([label] * len(abstracts))
    return data, labels

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Load and preprocess data
folder_path = "abstracts"
data, labels = load_data(folder_path)
data = [preprocess_text(text) for text in data]

# Convert to DataFrame
df = pd.DataFrame({"text": data, "label": labels})

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f} \u00b1 {np.std(cv_scores):.4f}")

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Astronomy", "Psychology", "Sociology"])
cm_ratio = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row (true class)

print("Confusion Matrix (Ratio):")
print(cm_ratio)

# Save the model
model_file = "models/random_forest_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer
vectorizer_file = "models/random_forest_vectorizer.pkl"
with open(vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"Saved model as {model_file} and vectorizer as {vectorizer_file}.")