# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Run Streamlit app
CMD ["streamlit", "run", "multimodal_rag_app.py", "--server.port=8080", "--server.address=0.0.0.0"]