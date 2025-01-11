import os
import requests
from bs4 import BeautifulSoup

# List of input files
input_files = ["paper_links/Astronomy_links.txt", "paper_links/Psychology_links.txt", "paper_links/Sociology_links.txt"]

# Directory to save the scraped content
output_dir = "scraped_data"
os.makedirs(output_dir, exist_ok=True)

# Function to scrape content from a URL
def scrape_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find_all(id="Abs1-content")
        return "\n\n".join([section.get_text(strip=True) for section in content])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# Process each file
for input_file in input_files:
    subject = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{subject}_content.txt")

    print(f"Processing {input_file}...")

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        count = 0
        for line in infile:
            count += 1
            url = line.strip()
            if url:
                print(f"{count} of {input_file}: {url}")
                content = scrape_content(url)
                if content:
                    outfile.write(content)

    print(f"Scraped content saved to {output_file}\n")

print("Scraping completed!")
