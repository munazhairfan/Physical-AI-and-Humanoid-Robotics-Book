# RAG Chatbot - Physical AI & Humanoid Robotics - Deployment Guide

## Quick Deploy

This RAG chatbot can be deployed in multiple ways. Choose the option that best fits your needs.

### Option 1: Hugging Face Spaces (Easiest)

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Create a new Space with the "Docker" SDK
3. Set your environment variables:
   - `GEMINI_API_KEY`: Your Google Gemini API key
4. Add the following files to your repository:
   - All backend files
   - `Dockerfile`
   - `Procfile`
   - `requirements.txt`
   - `.env.example`
5. The Space will automatically build and deploy using the Dockerfile

### Option 2: Docker (Self-hosting)

1. Clone the repository
2. Create a `.env` file with your API keys (use `.env.example` as template)
3. Build the Docker image:
   ```bash
   docker build -t rag-chatbot .
   ```
4. Run the container:
   ```bash
   docker run -d -p 8000:8000 --env-file .env rag-chatbot
   ```

### Option 3: Direct Python (Development)

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set environment variables:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```
5. Start the server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Environment Variables

- `GEMINI_API_KEY`: Required - Google Gemini API key for LLM and embeddings
- `QDRANT_URL`: Optional - Qdrant cloud instance URL (defaults to in-memory)
- `QDRANT_API_KEY`: Optional - Qdrant cloud instance API key

## API Endpoints

- `POST /chat` - Chat with the RAG bot
- `POST /book_query` - Query the entire book/document collection
- `POST /selected_text` - Process user-selected text
- `POST /ingest` - Ingest documents into the RAG system
- `GET /health` - Health check

## Frontend Integration

The backend provides a REST API that can be integrated with any frontend. The frontend should call the backend API endpoints to interact with the RAG system.

## Notes

- This deployment uses in-memory storage for documents and chat history (not suitable for production)
- For production, connect to a Qdrant cloud instance by setting QDRANT_URL and QDRANT_API_KEY
- The Google Gemini API has usage quotas - monitor your usage at Google AI Studio