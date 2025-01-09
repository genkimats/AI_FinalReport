import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess function
def preprocess_text(text):
    # Remove punctuations and characters other than [^a-zA-Z\s]
    # Convert text to lower case
    text = text.lower()

    # Remove punctuations and characters other than [^a-zA-Z\s]
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Rejoin the text
    processed_text = " ".join(tokens)

    return processed_text

# Process a file
def preprocess_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    processed_lines = [preprocess_text(line) for line in lines if line.strip()]

    return processed_lines

# Save processed data to a new file
def save_processed_file(processed_lines, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(processed_lines))

# Main function to process all files in a folder
def preprocess_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.startswith("."):
            continue
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)
        print(input_file_path)

        if os.path.isfile(input_file_path):
            print(f"Processing {file_name}...")
            processed_lines = preprocess_file(input_file_path)
            save_processed_file(processed_lines, output_file_path)

# Example usage
if __name__ == "__main__":
    input_folder = "abstracts"
    output_folder = "abstracts_processed"
    preprocess_folder(input_folder, output_folder)
    print("Preprocessing complete. Processed files are saved in the 'processed_abstracts' folder.")