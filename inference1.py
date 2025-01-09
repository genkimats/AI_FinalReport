import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load the saved model and vectorizer
with open('models/text_classifier.pkl', 'rb') as file:
    model, vectorizer = pickle.load(file)


# Step 2: Define a function to make predictions
def predict_category(abstracts):
    # Preprocess the abstracts
    abstracts_tfidf = vectorizer.transform(abstracts)

    # Make predictions
    predictions = model.predict(abstracts_tfidf)

    return predictions


# Step 3: Test the function with new abstracts
new_abstracts = [
    "This study explores the effects of galaxy formation on the overall structure of the universe.",
    "The impact of cognitive behavioral therapy on patients with anxiety disorders was examined.",
    "An analysis of social networks and their influence on modern communication patterns."
]

predicted_categories = predict_category(new_abstracts)

# Print the predictions
for abstract, category in zip(new_abstracts, predicted_categories):
    print(f"Abstract: {abstract}\nPredicted Category: {category}\n")
