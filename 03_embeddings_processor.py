import os
import pandas as pd
import numpy as np
from google.cloud import storage
import google.auth
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
import vertexai
import io
from PIL import Image

# Constants
BUCKET_NAME = "the-batch-storage"
CSV_BLOB_NAME = "deeplearning_articles.csv"
IMAGES_FOLDER = "article_images/"
PROJECT_ID = "1016376863515"
LOCATION = "us-central1"
MAX_IMAGE_SIZE = 20 * 1024 * 1024

# Initialize Vertex AI and Storage
credentials, _ = google.auth.default()
storage_client = storage.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

print("Libraries imported and environment initialized.")

def resize_image(image_bytes, max_size=MAX_IMAGE_SIZE):
    """Resize image if it exceeds max_size, maintaining aspect ratio."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if len(image_bytes) <= max_size:
            return image_bytes
        
        # Calculate new size to fit within max_size
        img_format = img.format or 'JPEG'
        quality = 95
        while len(image_bytes) > max_size and quality > 5:
            output = io.BytesIO()
            # Reduce size incrementally
            new_width = int(img.width * 0.9)
            new_height = int(img.height * 0.9)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img.save(output, format=img_format, quality=quality)
            image_bytes = output.getvalue()
            output.close()
            quality -= 10
        
        if len(image_bytes) > max_size:
            print(f"Warning: Image still exceeds {max_size} bytes after resizing.")
        return image_bytes
    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        return None

def preprocess_embeddings():
    """Generate and save text and image embeddings to GCS"""
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Load articles
    articles_blob = bucket.blob(CSV_BLOB_NAME)
    articles_data = articles_blob.download_as_string()
    articles_df = pd.read_csv(io.BytesIO(articles_data))
    articles_df['article_url'] = articles_df['article_url'].astype(str).str.strip().str.rstrip('/')
    
    # Load image metadata
    images_blob = bucket.blob("article_images_metadata.csv")
    images_data = images_blob.download_as_string()
    images_metadata_df = pd.read_csv(io.BytesIO(images_data))
    images_metadata_df['article_url'] = images_metadata_df['article_url'].astype(str).str.strip().str.rstrip('/')
    
    # Initialize multimodal model
    multimodal_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    
    # Generate text embeddings
    print("Generating text embeddings...")
    contents = [str(content)[:320] for content in articles_df['content'].tolist()]
    batch_size = 250
    text_embeddings = []
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        print(f"Processing text batch {i // batch_size + 1}: {len(batch)} articles")
        batch_embeddings = [multimodal_model.get_embeddings(contextual_text=text) for text in batch]
        text_embeddings.extend([embedding.text_embedding for embedding in batch_embeddings])
    text_embeddings = np.array(text_embeddings)
    
    # Save text embeddings to GCS
    text_blob = bucket.blob("preprocessed/text_embeddings.npy")
    with io.BytesIO() as f:
        np.save(f, text_embeddings)
        f.seek(0)
        text_blob.upload_from_file(f, content_type="application/octet-stream")
    print(f"Text embeddings saved to gs://{BUCKET_NAME}/preprocessed/text_embeddings.npy")
    
    # Generate image embeddings
    print("Generating image embeddings...")
    image_paths = images_metadata_df['image_gcs_path'].unique()
    batch_size = 100
    image_embeddings = []
    skipped_images = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"Processing image batch {i // batch_size + 1}: {len(batch_paths)} images")
        batch_images = []
        for path in batch_paths:
            try:
                blob = bucket.blob(path.replace(f"gs://{BUCKET_NAME}/", ""))
                image_bytes = blob.download_as_bytes()
                if len(image_bytes) > MAX_IMAGE_SIZE:
                    print(f"Resizing image {path} (size: {len(image_bytes)} bytes)...")
                    image_bytes = resize_image(image_bytes)
                if image_bytes and len(image_bytes) <= MAX_IMAGE_SIZE:
                    batch_images.append(VertexImage(image_bytes))
                else:
                    print(f"Skipping image {path}: Exceeds size limit or resizing failed.")
                    skipped_images.append(path)
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                skipped_images.append(path)
        
        try:
            if batch_images:  # Only process if there are valid images
                batch_embeddings = [multimodal_model.get_embeddings(image=img) for img in batch_images]
                image_embeddings.extend([embedding.image_embedding for embedding in batch_embeddings])
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {str(e)}")
            # Skip failed images from this batch
            valid_embeddings = len(image_embeddings) - len(batch_images)
            skipped_images.extend(batch_paths[valid_embeddings:])
    
    # Create image embeddings DataFrame, excluding skipped images
    valid_paths = [path for path in image_paths if path not in skipped_images]
    image_embeddings_df = pd.DataFrame({
        'image_gcs_path': valid_paths,
        'embedding': image_embeddings
    })
    
    # Save image embeddings to GCS
    print("Saving image embeddings...")
    image_blob = bucket.blob("preprocessed/image_embeddings.npy")
    with io.BytesIO() as f:
        np.save(f, np.array(image_embeddings_df['embedding'].tolist()))
        f.seek(0)
        image_blob.upload_from_file(f, content_type="application/octet-stream")
    image_paths_blob = bucket.blob("preprocessed/image_paths.csv")
    image_embeddings_df[['image_gcs_path']].to_csv("image_paths.csv", index=False)
    image_paths_blob.upload_from_filename("image_paths.csv")
    print(f"Image embeddings saved to gs://{BUCKET_NAME}/preprocessed/image_embeddings.npy")
    print(f"Image paths saved to gs://{BUCKET_NAME}/preprocessed/image_paths.csv")
    
    print(f"Processed {len(text_embeddings)} text embeddings and {len(image_embeddings)} image embeddings.")
    if skipped_images:
        print(f"Skipped {len(skipped_images)} images due to size or processing errors: {', '.join(skipped_images[:5])}...")
    
    return articles_df, images_metadata_df

print("Function to preprocess embeddings defined.")

if __name__ == "__main__":
    articles_df, images_metadata_df = preprocess_embeddings()
    print("Processing embeddings completed.")
