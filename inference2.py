import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Step 1: Load the Saved Model and Vectorizer
model = load_model('models/text_classifier_nn.h5')
with open('models/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Step 2: Define the Inference Function
def predict_category(text):
    # Preprocess the input text
    text_tfidf = vectorizer.transform([text]).toarray()
    
    # Perform prediction
    predictions = model.predict(text_tfidf)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Map the predicted class index back to the label
    label_map = {0: 'Astronomy', 1: 'Psychology', 2: 'Sociology'}
    return label_map[predicted_class], predictions[0]

# Step 3: Test the Inference
if __name__ == "__main__":
    sample_text = "The study of celestial objects and the universe beyond Earth's atmosphere."
    predicted_label, probabilities = predict_category(sample_text)
    
    print("Input Text:", sample_text)
    print("Predicted Label:", predicted_label)
    print("Class Probabilities:", probabilities)
