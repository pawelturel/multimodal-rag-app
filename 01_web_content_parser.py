import requests
import pandas as pd
from lxml import etree
from bs4 import BeautifulSoup
import time
import re

# Fetch XML sitemap
sitemap_url = "https://www.deeplearning.ai/sitemap-0.xml"
r = requests.get(sitemap_url)
root = etree.fromstring(r.content)

# Extract relevant URLs
article_urls = [sitemap[0].text for sitemap in root if sitemap[0].text.startswith("https://www.deeplearning.ai/the-batch/issue")]

data = []

# Function to clean article content
def clean_content(content):
    content = re.sub(r"Loading the\s+Elevenlabs\s+Text\s+to\s+Speech\s+AudioNative\s+Player\.\.\.\s*", "", content)
    return content.strip()

# Extract article details
def extract_article_details(url, retries=5, delay=3):
    for attempt in range(retries):
        try:
            article_res = requests.get(url, timeout=30)
            if article_res.status_code == 200:
                soup = BeautifulSoup(article_res.text, "html.parser")
                title = soup.title.text if soup.title else ""

                # Extract content
                content_div = soup.find("div", class_="prose--styled justify-self-center post_postContent__wGZtc")
                content = content_div.get_text(separator=" ") if content_div else ""
                content = clean_content(content)

                # Extract images
                image_urls = [img["src"] for img in content_div.find_all("img")] if content_div else []

                return {"article_url": url, "title": title, "content": content, "image_urls": image_urls}
            else:
                print(f"Retrying to fetch {url}, status code: {article_res.status_code}")
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
        time.sleep(delay)
    return None

# Process each article
for url in article_urls:
    details = extract_article_details(url)
    if details:
        data.append(details)

# Convert to DataFrame
df = pd.DataFrame(data, columns=["article_url", "title", "content", "image_urls"])

print(f"Data Frame has been created with {len(df)} articles.")

from google.cloud import storage

# Google Cloud Storage details
bucket_name = "the-batch-storage"
blob_name = "deeplearning_articles.csv"

def save_csv_to_gcs(df, bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    csv_data = df.to_csv(index=False)
    blob.upload_from_string(csv_data, content_type='text/csv')

# Save to Google Cloud Storage
save_csv_to_gcs(df, bucket_name, blob_name)

print(f"CSV file saved to gs://{bucket_name}/{blob_name}")
