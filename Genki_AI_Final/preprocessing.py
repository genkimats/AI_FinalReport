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

# Main function to process a text file
def process_text_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            text = infile.read()

        # Preprocess text
        tokens_per_line = preprocess_text(text)

        # Process text based on the selected mode
        lemmatizer = WordNetLemmatizer()
        lemmatized_list = [' '.join([lemmatizer.lemmatize(token) for token in tokens]) for tokens in tokens_per_line]
        lemmatized_lines = '\n'.join(lemmatized_list)

        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(lemmatized_lines)

        print(f"Text processing completed. Output saved to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# Replace 'input.txt' with the path to your input file
# Replace 'output.txt' with the path to your desired output file
# Use mode="stem" for stemming or mode="lemmatize" for lemmatization
input_directory = 'abstracts'
input_files = ['psychology.txt', 'sociology.txt', 'astronomy.txt']
output_directory = 'abstracts_processed'
if not os.path.exists(output_directory):
        os.makedirs(output_directory)
for file in os.listdir(input_directory):
    if file.startswith("."):
            continue
    process_text_file(f"{input_directory}/{file}", f"{output_directory}/processed_{file}")