# **Multimodal Retrieval-Augmented Generation (RAG) System Documentation**

## **Overview**
This system is designed to retrieve relevant news articles from [The Batch](https://www.deeplearning.ai/the-batch/) using both textual and visual data. The system allows users to input queries, answer questions, and retrieve articles along with associated images using a multimodal approach.

## **System Components**
The system consists of four main scripts:

1. **01_web_content_parser.py** – Fetches articles from The Batch website and saves them to Google Cloud Storage (GCS).
2. **02_images_parser.py** – Downloads and uploads images associated with the articles to GCS, creating a metadata file.
3. **03_embeddings_processor.py** – Processes text and image embeddings using Vertex AI.
4. **04_multimodal_rag_app.py** – Implements the retrieval system, integrates Google’s Gemini LLM for query response generation, and provides a Streamlit-based UI.

---
## **Script Descriptions**

### **1. 01_web_content_parser.py** (Web Content Extraction)
#### **Purpose**
- Extracts article URLs from the website’s XML sitemap.
- Scrapes article content and image URLs.
- Stores the data in a Pandas DataFrame and uploads it as a CSV file to GCS.

#### **Key Functions**
- `clean_content(content)`: Cleans extracted text.
- `extract_article_details(url)`: Fetches the article title, content, and images.
- `save_csv_to_gcs(df, bucket_name, blob_name)`: Saves extracted data to GCS.

#### **Output**
- CSV file: `deeplearning_articles.csv` in `the-batch-storage` bucket.

---
### **2. 02_images_parser.py** (Image Processing)
#### **Purpose**
- Downloads images from article URLs.
- Uploads images to GCS.
- Creates and stores image metadata.

#### **Key Functions**
- `download_csv_from_gcs()`: Loads article data from GCS.
- `upload_image_to_gcs(image_bytes, destination_path)`: Uploads images to GCS.
- `process_images(df)`: Processes images and stores metadata.
- `save_metadata_to_gcs(metadata)`: Saves metadata to GCS.

#### **Output**
- Images stored in GCS under `article_images/`.
- Metadata file `article_images_metadata.csv` in GCS.

---
### **3. 03_embeddings_processor.py** (Embedding Generation)
#### **Purpose**
- Generates text and image embeddings using Vertex AI.
- Saves embeddings for efficient retrieval.

#### **Key Functions**
- `resize_image(image_bytes, max_size)`: Resizes large images.
- `preprocess_embeddings()`: Processes and saves embeddings.

#### **Output**
- `preprocessed/text_embeddings.npy` – Stores text embeddings.
- `preprocessed/image_embeddings.npy` – Stores image embeddings.
- `preprocessed/image_paths.csv` – Stores image paths.

---
### **4. 04_multimodal_rag_app.py** (Retrieval System & UI)
#### **Purpose**
- Loads data and embeddings.
- Allows users to input queries and retrieves relevant articles and images.
- Uses Gemini LLM to generate responses.
- Provides a UI via Streamlit.

#### **Key Functions**
- `load_data()`: Loads article and image metadata.
- `load_precomputed_embeddings()`: Loads embeddings.
- `get_query_embedding(query)`: Generates a query embedding.
- `retrieve_relevant_articles_and_images(query)`: Retrieves top-matching articles and images.
- `generate_answer(query, context)`: Uses Gemini LLM to answer user queries.
- `download_image(gcs_path)`: Downloads images from GCS.
- `run_scripts()`: Refreshes data by running preprocessing scripts.
- `main()`: Streamlit app’s main function.

#### **UI Features**
- Query input for searching articles.
- Displays relevant articles and images.
- Provides a generated answer based on retrieved data.
- Refresh button to update stored content.

---
## **System Workflow**
1. **Web Scraping** (01_web_content_parser.py)
   - Fetches article content and image URLs.
   - Saves to `deeplearning_articles.csv` in GCS.

2. **Image Processing** (02_images_parser.py)
   - Downloads images and uploads them to `article_images/`.
   - Creates `article_images_metadata.csv`.

3. **Embedding Generation** (03_embeddings_processor.py)
   - Generates text and image embeddings using Vertex AI.
   - Saves embeddings for retrieval.

4. **Retrieval & UI** (04_multimodal_rag_app.py)
   - Loads stored data and embeddings.
   - Matches queries with relevant articles/images.
   - Uses Gemini LLM to generate responses.
   - Provides an interactive Streamlit UI.

---
## **Deployment & Testing**
- Run `04_multimodal_rag_app.py` to start the Streamlit UI.
- Use the **Refresh Content** button to update stored data.
- Enter queries to test retrieval and response generation.

## **Future Enhancements**
- Improve article parsing and cleaning.
- Enhance UI with image search and filtering.
- Implement caching for faster retrieval.
