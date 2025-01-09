import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
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

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Step 3: Define the Neural Network Model
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

# Initialize the model
input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = 3  # 3 output classes
model = TextClassifier(input_dim, hidden_dim, output_dim)

# Load pre-trained weights if available
try:
    model.load_state_dict(torch.load('models/text_classifier_nn.pth'))
    print("Loaded pre-trained model.")
except FileNotFoundError:
    print("No pre-trained model found. Initializing a new model.")

# Freeze the first layer for fine-tuning
for param in model.fc1.parameters():
    param.requires_grad = False

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)  # Lower learning rate

# Step 4: Train the Model (Fine-Tuning)
def train_model(model, criterion, optimizer, X_train, y_train, epochs=5, batch_size=32):
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor)

# Step 5: Evaluate the Fine-Tuned Model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
        print("Test Set Classification Report:")
        print(classification_report(y_test.numpy(), predictions.numpy()))
        print("Test Set Confusion Matrix:")
        print(confusion_matrix(y_test.numpy(), predictions.numpy()))

evaluate_model(model, X_test_tensor, y_test_tensor)

# Step 6: Save the Fine-Tuned Model and Vectorizer
torch.save(model.state_dict(), 'models/text_classifier_nn_finetuned.pth')
with open('models/vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Fine-tuned PyTorch model and vectorizer saved.")
