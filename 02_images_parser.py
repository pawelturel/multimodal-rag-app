import os
import pandas as pd
import requests
from google.cloud import storage
from io import BytesIO

# Google Cloud Storage details
BUCKET_NAME = "the-batch-storage"
BLOB_NAME = "deeplearning_articles.csv"
IMAGE_FOLDER = "article_images/"

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def download_csv_from_gcs():
    """Download CSV file from Google Cloud Storage and load it into a DataFrame."""
    blob = bucket.blob(BLOB_NAME)
    content = blob.download_as_text()
    return pd.read_csv(BytesIO(content.encode()), converters={'image_urls': eval})

def upload_image_to_gcs(image_bytes, destination_path):
    """Upload an image to Google Cloud Storage."""
    blob = bucket.blob(destination_path)
    blob.upload_from_string(image_bytes, content_type='image/jpeg')
    return f'gs://{BUCKET_NAME}/{destination_path}'

def process_images(df):
    """Download images and upload them to GCS, storing metadata for retrieval."""
    metadata = []
    
    for _, row in df.iterrows():
        article_url = row['article_url']
        title = row['title']
        image_urls = row['image_urls']
        
        for idx, img_url in enumerate(image_urls):
            try:
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    image_bytes = response.content
                    filename = f"{IMAGE_FOLDER}{title.replace(' ', '_')}_{idx}.jpg"
                    gcs_url = upload_image_to_gcs(image_bytes, filename)
                    
                    metadata.append({
                        "article_url": article_url,
                        "title": title,
                        "image_url": img_url,
                        "image_gcs_path": gcs_url
                    })
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {img_url}: {e}")
        
        print(f"URL processed: {article_url}")
    
    return metadata

def save_metadata_to_gcs(metadata):
    """Save image metadata as a CSV file in Google Cloud Storage."""
    metadata_df = pd.DataFrame(metadata)
    csv_buffer = BytesIO()
    metadata_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    metadata_blob = bucket.blob("article_images_metadata.csv")
    metadata_blob.upload_from_file(csv_buffer, content_type='text/csv')
    print("Metadata uploaded to GCS")

def main():
    df = download_csv_from_gcs()
    metadata = process_images(df)
    save_metadata_to_gcs(metadata)

if __name__ == "__main__":
    main()
