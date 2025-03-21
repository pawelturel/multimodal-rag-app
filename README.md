# **Multimodal Retrieval-Augmented Generation (RAG) System**

## **[APP DEMO](https://www.youtube.com/watch?v=JOkY2_gzk4k)**

## **Overview**
This repository contains the implementation of a **Multimodal Retrieval-Augmented Generation (RAG) System** that retrieves relevant news articles from [The Batch](https://www.deeplearning.ai/the-batch/), incorporating both textual and visual data. The system allows users to input queries, retrieve relevant articles, and display associated images using advanced AI techniques.

## **Features**
- **Automated Web Scraping**: Extracts articles and associated images from The Batch.
- **Multimodal Data Processing**: Handles both text and images for improved retrieval.
- **Efficient Storage & Indexing**: Uses Google Cloud Storage for data storage and Vertex AI for embedding generation.
- **Query-Based Retrieval**: Matches queries with the most relevant articles and images.
- **Generative AI Integration**: Utilizes Gemini LLM for answering user queries.
- **User-Friendly Interface**: Built with Streamlit for interactive searching and visualization.

## **Project Structure**
```plaintext
.
├── 01_web_content_parser.py      # Scrapes articles from The Batch
├── 02_images_parser.py           # Downloads and processes article images
├── 03_embeddings_processor.py    # Generates text and image embeddings
├── 04_multimodal_rag_app.py      # Main Streamlit UI for retrieval and querying
├── requirements.txt              # List of dependencies
├── README.md                     # Project documentation
├── Documentation.md              # Project documentation
├── .gitignore                    # Files to ignore in Git
```

## **Installation & Setup**
### **Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- Google Cloud SDK
- Vertex AI API enabled
- Streamlit

### **Step 1: Clone the Repository**
```sh
git clone https://github.com/your-repo/multimodal-rag.git
cd multimodal-rag
```

### **Step 2: Install Dependencies**
```sh
pip install -r requirements.txt
```

### **Step 3: Set Up Google Cloud**
- Authenticate with Google Cloud:
```sh
gcloud auth application-default login
```
- Ensure you have access to Google Cloud Storage and Vertex AI.

## **Usage**
### **1. Fetch and Store Articles**
```sh
python 01_web_content_parser.py
```
### **2. Process Images**
```sh
python 02_images_parser.py
```
### **3. Generate Embeddings**
```sh
python 03_embeddings_processor.py
```
### **4. Run the Retrieval App**
```sh
streamlit run 04_multimodal_rag_app.py --server.port 8505
```

## **Using the UI**
1. Open your browser and go to `http://localhost:8505` after running the Streamlit app.
2. Enter a search query in the input box.
3. Click the "Search" button to retrieve relevant articles and images.
4. View the generated response from the Gemini LLM based on retrieved content.
5. Browse the top relevant articles and associated images.
6. Use the "Refresh Content" button to update the stored data from The Batch.
7. If needed, stop the refresh process by clicking the "Stop Refresh" button.

## **How It Works**
1. Scrapes articles and images from The Batch.
2. Stores and processes text and images in Google Cloud Storage.
3. Generates embeddings for efficient search.
4. Matches user queries to relevant articles and images.
5. Provides a user-friendly UI for searching and interacting with the data.

## **Future Enhancements**
- Support for additional news sources.
- Improved text and image relevance ranking.
- Enhanced UI with filtering and search optimization.

## **Contributing**
Contributions are welcome! Feel free to open issues or submit pull requests.

## **License**
This project is licensed under the MIT License.
