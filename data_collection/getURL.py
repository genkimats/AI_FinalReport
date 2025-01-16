import requests
from bs4 import BeautifulSoup

def get_urls(base_url, pages, html_class):
    urls = []
    for page in range(1, pages + 1):
        full_url = f"{base_url}{page}"
        try:
            response = requests.get(full_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', class_=html_class)
            for link in links:
                href = link.get('href')
                if href:
                    urls.append(f"https://link.springer.com{href}")
        except Exception as e:
            print(f"Failed to process page {page} of {base_url}: {e}")
    print('\n'.join(urls))
    return urls

def save_urls_to_file(file_name, urls):
    with open(file_name, 'w') as file:
        file.write('\n'.join(urls))

# Base URLs
base_urls = {
    "astronomy": "https://link.springer.com/search?new-search=true&query=&content-type=Research&facet-sub-discipline=%22Astronomy%2C+Astrophysics+and+Cosmology%22&featureFlags.show-entitlements=true&sortBy=newestFirst&page=",
    "psychology": "https://link.springer.com/search?new-search=true&query=&content-type=Research&language=En&facet-discipline=%22Psychology%22&featureFlags.show-entitlements=true&sortBy=newestFirst&page=",
    "sociology": "https://link.springer.com/search?new-search=true&query=&content-type=Research&language=En&facet-sub-discipline=%22Sociology%2C+general%22&featureFlags.show-entitlements=true&sortBy=newestFirst&page=",
}

# HTML class to find links
html_class = "app-card-open__link"

# Number of pages to scrape (adjust as needed)
pages_to_scrape = 50
output_dir = "paper_links"

# Extract URLs and save them to files
for key, base_url in base_urls.items():
    print(f"Processing {key}...")
    urls = get_urls(base_url, pages_to_scrape, html_class)
    save_urls_to_file(f"{output_dir}/{key}_links.txt", urls)
    print(f"Saved {len(urls)} URLs to {key}.txt")
