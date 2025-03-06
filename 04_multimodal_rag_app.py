import os
import pandas as pd
import numpy as np
from google.cloud import storage
import google.auth
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.generative_models import GenerativeModel
import vertexai
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import streamlit as st
import subprocess
from datetime import datetime
from collections import deque
import time

# Set the secret path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/GCP_SERVICE_ACCOUNT_KEY"

# Set page config to wide layout
st.set_page_config(layout="wide")

# Use Cloud Run's PORT
port = int(os.getenv("PORT", 8080))

# Constants
BUCKET_NAME = "the-batch-storage"
CSV_BLOB_NAME = "deeplearning_articles.csv"
IMAGES_METADATA_BLOB_NAME = "article_images_metadata.csv"
IMAGES_FOLDER = "article_images/"
PROJECT_ID = "the-batch-multimodal-rag"
LOCATION = "us-central1"
MODEL_NAME = "gemini-pro"
SCRIPTS = [
    "01_web_content_parser.py",
    "02_images_parser.py",
    "03_embeddings_processor.py"
]
LAST_UPDATED_FILE = "last_updated.txt"

# Initialize Google Cloud Storage and Vertex AI
credentials, _ = google.auth.default()
storage_client = storage.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

class MultimodalRAGSystem:
    def __init__(self):
        self.multimodal_model = None
        self.generation_model = None
        self.articles_df = None
        self.images_metadata_df = None
        self.text_embeddings = None
        self.image_embeddings = None
        self.load_data()
        self.load_precomputed_embeddings()

    def load_data(self):
        bucket = storage_client.bucket(BUCKET_NAME)
        articles_blob = bucket.blob(CSV_BLOB_NAME)
        articles_data = articles_blob.download_as_string()
        self.articles_df = pd.read_csv(io.BytesIO(articles_data))
        self.articles_df['article_url'] = self.articles_df['article_url'].astype(str).str.strip().str.rstrip('/')
        images_blob = bucket.blob(IMAGES_METADATA_BLOB_NAME)
        images_data = images_blob.download_as_string()
        self.images_metadata_df = pd.read_csv(io.BytesIO(images_data))
        self.images_metadata_df['article_url'] = self.images_metadata_df['article_url'].astype(str).str.strip().str.rstrip('/')
        if 'image_url' in self.images_metadata_df.columns:
            self.images_metadata_df['image_url'] = self.images_metadata_df['image_url'].astype(str).str.strip()

    def load_precomputed_embeddings(self):
        bucket = storage_client.bucket(BUCKET_NAME)
        text_blob = bucket.blob("preprocessed/text_embeddings.npy")
        with io.BytesIO(text_blob.download_as_bytes()) as f:
            self.text_embeddings = np.load(f)
        print(f"Loaded text embeddings shape: {self.text_embeddings.shape}")
        image_blob = bucket.blob("preprocessed/image_embeddings.npy")
        with io.BytesIO(image_blob.download_as_bytes()) as f:
            image_embeddings = np.load(f)
        image_paths_blob = bucket.blob("preprocessed/image_paths.csv")
        image_paths_data = image_paths_blob.download_as_string()
        image_paths_df = pd.read_csv(io.BytesIO(image_paths_data))
        self.image_embeddings = pd.DataFrame({
            'image_gcs_path': image_paths_df['image_gcs_path'],
            'embedding': list(image_embeddings)
        })
        print(f"Loaded image embeddings shape: {self.image_embeddings.shape}")

    def get_query_embedding(self, query: str) -> np.ndarray:
        if self.multimodal_model is None:
            print("Initializing multimodal model...")
            self.multimodal_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        embedding = self.multimodal_model.get_embeddings(contextual_text=query).text_embedding
        return np.array(embedding)

    def retrieve_relevant_articles_and_images(self, query: str, top_k_articles: int = 5, top_k_images: int = 10) -> tuple:
        query_embedding = self.get_query_embedding(query)

        # Article retrieval (text-based)
        text_similarities = cosine_similarity(query_embedding.reshape(1, -1), self.text_embeddings)[0]
        top_article_indices = np.argsort(text_similarities)[::-1][:top_k_articles]
        article_results = []
        for idx in top_article_indices:
            article = self.articles_df.iloc[idx]
            related_images = self.images_metadata_df[
                self.images_metadata_df['article_url'] == article['article_url']
            ]
            image_paths = related_images['image_gcs_path'].tolist()
            image_urls = related_images['image_url'].tolist() if 'image_url' in related_images.columns else image_paths
            article_results.append({
                'title': article['title'],
                'content': article['content'][:500] + "...",
                'url': article['article_url'],
                'text_similarity': text_similarities[idx],
                'image_paths': image_paths,
                'image_urls': image_urls[:len(image_paths)]
            })

        # Image retrieval (image-based, all images)
        image_embeddings_array = np.array(self.image_embeddings['embedding'].tolist())
        image_similarities = cosine_similarity(query_embedding.reshape(1, -1), image_embeddings_array)[0]
        top_image_indices = np.argsort(image_similarities)[::-1][:top_k_images]
        image_results = []
        for idx in top_image_indices:
            image_path = self.image_embeddings.iloc[idx]['image_gcs_path']
            related_image_data = self.images_metadata_df[
                self.images_metadata_df['image_gcs_path'] == image_path
            ]
            article_url = related_image_data['article_url'].iloc[0] if not related_image_data.empty else "No associated article"
            image_url = related_image_data['image_url'].iloc[0] if not related_image_data.empty and 'image_url' in related_image_data.columns else image_path
            image_results.append({
                'image_path': image_path,
                'similarity': image_similarities[idx],
                'article_url': article_url,
                'image_url': image_url
            })

        return article_results, image_results

    def generate_answer(self, query: str, context: str) -> str:
        if self.generation_model is None:
            print("Initializing Gemini Pro model...")
            self.generation_model = GenerativeModel(MODEL_NAME)
        prompt = f"""
        Question: {query}
        Context: {context}
        Provide a concise answer based on the context:
        """
        try:
            response = self.generation_model.generate_content(prompt)
            return response.text if response else "No response generated."
        except Exception as e:
            return f"Error generating answer with Gemini Pro: {str(e)}. Context:\n{context[:500]}"

    def download_image(self, gcs_path: str) -> Image.Image:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_path = gcs_path.replace(f"gs://{BUCKET_NAME}/", "")
        blob = bucket.blob(blob_path)
        image_data = blob.download_as_bytes()
        return Image.open(io.BytesIO(image_data))

def run_scripts():
    status_placeholder = st.empty()
    output_placeholder = st.empty()  # Define within function
    st.session_state['refreshing'] = True
    st.session_state['process'] = None
    output_lines = deque(maxlen=10)  # Store last 10 lines

    for script in SCRIPTS:
        if not st.session_state['refreshing']:
            status_placeholder.write("Refresh stopped by user.")
            return False
        if not os.path.exists(script):
            status_placeholder.error(f"Script {script} not found in current directory.")
            st.session_state['refreshing'] = False
            return False
        status_placeholder.write(f"Running {script}...")
        try:
            process = subprocess.Popen(
                ["python", "-u", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            st.session_state['process'] = process
            line_counter = 0  # Unique counter for keys
            # Stream stdout line-by-line
            for line in iter(process.stdout.readline, ''):
                if not st.session_state['refreshing']:
                    process.terminate()
                    status_placeholder.write("Refresh stopped by user.")
                    return False
                if line.strip():
                    output_lines.append(line.strip())
                    line_counter += 1
                    output_placeholder.text_area(
                        "Latest outputs:",
                        "\n".join(output_lines),
                        height=200,
                        key=f"output_{script}_{line_counter}",
                        disabled=True
                    )
            process.wait()
            if process.returncode != 0:
                error_output = process.stderr.read()
                status_placeholder.error(f"Error running {script}: {error_output}")
                st.session_state['refreshing'] = False
                return False
            status_placeholder.write(f"Completed {script}")
        except Exception as e:
            status_placeholder.error(f"Exception running {script}: {str(e)}")
            st.session_state['refreshing'] = False
            return False
    # Update last updated timestamp
    with open(LAST_UPDATED_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.session_state['refreshing'] = False
    return True

def get_last_updated():
    if os.path.exists(LAST_UPDATED_FILE):
        with open(LAST_UPDATED_FILE, "r") as f:
            return f.read().strip()
    return "Not available"

def main():
    st.title("Multimodal News Retrieval System from ['The Batch'](https://www.deeplearning.ai/the-batch/)")
    
    # Buttons and Last Updated in a single column section
    last_updated = get_last_updated()
    st.write(f"**Last Updated:** {last_updated}")
    if st.button("Refresh Content"):
        st.session_state['refreshing'] = True
        with st.spinner("Refreshing content from The Batch website..."):
            success = run_scripts()
            if success:
                st.success("Content refreshed successfully!")
                # Reload data after refresh
                st.session_state.rag_system = MultimodalRAGSystem()
            else:
                st.error("Failed to refresh content. Check script errors.")
    # Stop Refresh Button always visible below Refresh Content
    if st.button("Stop Refresh"):
        if 'refreshing' in st.session_state and st.session_state['refreshing']:
            st.session_state['refreshing'] = False
            if 'process' in st.session_state and st.session_state['process']:
                st.session_state['process'].terminate()
                st.session_state['process'] = None
                st.write("Refresh stopped by user.")
    
    # Output section (wider)
    output_cols = st.columns([1, 4])  # 1 part spacer, 4 parts output
    with output_cols[1]:  # Use the wider column
        output_placeholder = st.empty()  # Define here instead of session state
        if 'refreshing' in st.session_state and st.session_state['refreshing']:
            run_scripts()  # Call run_scripts here to update output_placeholder
    
    st.write("Enter a query to retrieve relevant articles and images")
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MultimodalRAGSystem()
    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if query:
            with st.spinner("Retrieving articles and images..."):
                rag_system = st.session_state.rag_system
                article_results, image_results = rag_system.retrieve_relevant_articles_and_images(query)
                context = "\n".join([r['content'] for r in article_results])
                answer = rag_system.generate_answer(query, context)
                
                # Display Answer
                st.subheader("Generated Answer")
                st.write(answer)
                
                # Add extra empty space
                st.write("")
                st.write("")
                
                # Two-column layout for results
                col1, col2 = st.columns(2)
                
                # First Column: Top Articles
                with col1:
                    st.write("### Top Articles (Text-Based Search)")
                    for result in article_results:
                        st.subheader(result['title'])
                        st.write(f"Text Similarity: {result['text_similarity']:.3f}")
                        st.write(result['content'])
                        st.write(f"[Read full article]({result['url']})")
                        if result['image_paths']:
                            st.write("Related Images:")
                            image_cols = st.columns(3)
                            for idx, (img_path, img_url) in enumerate(zip(result['image_paths'], result['image_urls'])):
                                try:
                                    image = rag_system.download_image(img_path)
                                    col_idx = idx % 3
                                    image_cols[col_idx].markdown(f"[View full-size image]({img_url})")
                                    image_cols[col_idx].image(image, caption=f"Image {idx + 1}", width=300)
                                except Exception as e:
                                    st.error(f"Error loading image: {e}")
                
                # Second Column: Top Images
                with col2:
                    st.write("### Top Images (Image-Based Search)")
                    for img_result in image_results:
                        col_left, col_img, col_right = st.columns([1, 2, 1])
                        try:
                            image = rag_system.download_image(img_result['image_path'])
                            link_text = f"[View full-size image]({img_result['image_url']})"
                            if img_result['article_url'] != "No associated article":
                                link_text += f"{' ' * 20}[Source Article]({img_result['article_url']})"
                            col_img.markdown(link_text)
                            col_img.image(image, caption=f"Image (Similarity: {img_result['similarity']:.3f})", width=600)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")

if __name__ == "__main__":
    main()
