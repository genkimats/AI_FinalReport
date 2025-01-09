import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
X_tfidf = vectorizer.fit_transform(data['abstract']).toarray()

# Encode the labels
y = LabelEncoder().fit_transform(data['label'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 3: Neural Network Model
def create_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 output classes (Astronomy, Psychology, Sociology)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the model
model = create_model(X_train.shape[1])

# Step 4: Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

# Step 5: Evaluate the Model
# Evaluate on the test set
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)

print("Test Set Classification Report:")
print(classification_report(y_test, y_pred))
print("Test Set Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Save the Model and Vectorizer
model.save('models/text_classifier_nn.h5')
with open('models/vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Neural network model and vectorizer saved.")
