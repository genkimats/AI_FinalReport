import pyperclip

urls = ""
num_urls = 17


astronomy = "https://link.springer.com/search?new-search=true&query=&content-type=Research&facet-sub-discipline=%22Astronomy%2C+Astrophysics+and+Cosmology%22&featureFlags.show-entitlements=true&sortBy=newestFirst&page="
psychology = "https://link.springer.com/search?new-search=true&query=&content-type=Research&language=En&facet-discipline=%22Psychology%22&featureFlags.show-entitlements=true&sortBy=newestFirst&page="
sociology = "https://link.springer.com/search?new-search=true&query=&content-type=Research&language=En&facet-sub-discipline=%22Sociology%2C+general%22&featureFlags.show-entitlements=true&sortBy=newestFirst&page="

# only need to add page num at end
base_url = sociology

for i in range(2, num_urls+2):
    page_url = base_url + str(i) + "\n"
    urls = urls + page_url

pyperclip.copy(urls)
print(pyperclip.paste())