import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
import os

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Function to clean and tokenize text
def preprocess_text(text):
    # Remove non-alphanumeric characters except line breaks and convert text to lowercase
    text = re.sub(r'[^a-zA-Z\n]', ' ', text).lower()
    # Split text by line breaks
    lines = text.split('\n')
    # Tokenize each line
    tokens_per_line = [nltk.word_tokenize(line) for line in lines]
    # Remove stopwords from each line
    stop_words = set(stopwords.words('english'))
    tokens_per_line = [[word for word in tokens if word not in stop_words] for tokens in tokens_per_line]
    return tokens_per_line

# Function to lemmatize text
def lemmatize_text(tokens_per_line):
    lemmatizer = WordNetLemmatizer()
    lemmatized_lines = [' '.join([lemmatizer.lemmatize(token) for token in tokens]) for tokens in tokens_per_line]
    return '\n'.join(lemmatized_lines)

# Function to stem text
def stem_text(tokens_per_line):
    stemmer = PorterStemmer()
    stemmed_lines = [' '.join([stemmer.stem(token) for token in tokens]) for tokens in tokens_per_line]
    return '\n'.join(stemmed_lines)

# Main function to process a text file
def process_text_file(input_file, output_file, mode="lemmatize"):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            text = infile.read()

        # Preprocess text
        tokens_per_line = preprocess_text(text)

        # Process text based on the selected mode
        if mode == "lemmatize":
            processed_text = lemmatize_text(tokens_per_line)
        elif mode == "stem":
            processed_text = stem_text(tokens_per_line)
        else:
            raise ValueError("Invalid mode. Choose either 'lemmatize' or 'stem'.")

        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(processed_text)

        print(f"Text processing ({mode}) completed. Output saved to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# Replace 'input.txt' with the path to your input file
# Replace 'output.txt' with the path to your desired output file
# Use mode="stem" for stemming or mode="lemmatize" for lemmatization
input_directory = 'abstracts_processed/'
input_files = ['psychology.txt', 'sociology.txt', 'astronomy.txt']
output_directory = 'abstracts_lemmatized/'
for file in os.listdir(input_directory):
    process_text_file(input_directory+file, output_directory+file, mode="lemmatize")