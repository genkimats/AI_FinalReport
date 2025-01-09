import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import re

# Step 1: Read the Data
def read_abstracts(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    abstracts = re.findall(r'\"(.*?)\"', content, re.DOTALL)
    return pd.DataFrame({'abstract': abstracts, 'label': label})

# Reading data from the text files
astronomy_df = read_abstracts('abstracts/astronomy.txt', 'Astronomy')
psychology_df = read_abstracts('abstracts/psychology.txt', 'Psychology')
sociology_df = read_abstracts('abstracts/sociology.txt', 'Sociology')

# Combining all data into a single DataFrame
data = pd.concat([astronomy_df, psychology_df, sociology_df], ignore_index=True)

# Step 2: Data Preprocessing
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(data['abstract'])
y = data['label']

# Step 3: Model Selection
model = LogisticRegression(max_iter=1000)

# Step 4: Cross-Validation
# Performing cross-validation and getting predictions
y_pred_cv = cross_val_predict(model, X_tfidf, y, cv=5)

# Evaluating the cross-validated predictions
print("Cross-Validation Classification Report:")
print(classification_report(y, y_pred_cv))
print("Cross-Validation Confusion Matrix:")
print(confusion_matrix(y, y_pred_cv))

# Step 5: Final Train-Test Split and Evaluation
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Training the model on the training set
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model on the test set
print("Test Set Classification Report:")
print(classification_report(y_test, y_pred))
print("Test Set Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Save the Model
with open('models/text_classifier.pkl', 'wb') as file:
    pickle.dump((model, vectorizer), file)

print("Model and vectorizer saved to text_classifier.pkl")
