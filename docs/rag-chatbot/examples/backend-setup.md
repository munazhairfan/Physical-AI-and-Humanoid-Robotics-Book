---
title: Backend Setup Example
sidebar_label: Backend Setup
---

# Backend Setup Example

This example demonstrates how to set up the RAG Chatbot backend with FastAPI, including all required services and dependencies.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Access to Qdrant vector store (free tier or local instance)
- Access to Neon Postgres database (free tier or local instance)

## Step 1: Create a Virtual Environment

```bash
# Create a new directory for the project
mkdir rag-chatbot-backend
cd rag-chatbot-backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 2: Install Dependencies

Create a `requirements.txt` file:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
qdrant-client[fastembed]==1.8.0
asyncpg==0.29.0
numpy==1.24.3
sentence-transformers==2.2.2
torch==2.0.1
transformers==4.35.0
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Step 3: Create the Project Structure

Create the following directory structure:

```
rag-chatbot-backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── embedding_service.py
│   ├── llm_service.py
│   ├── vectorstore_service.py
│   └── database_service.py
└── config/
    └── settings.py
```

## Step 4: Create Configuration Settings

Create `config/settings.py`:

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Qdrant settings
    qdrant_url: Optional[str] = "http://localhost:6333"  # Use None for local, or your cloud URL
    qdrant_api_key: Optional[str] = None
    collection_name: str = "rag_documents"

    # Neon Postgres settings
    neon_db_url: Optional[str] = "postgresql://localhost:5432/rag_chatbot"  # Use your connection string

    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"  # Free embedding model

    class Config:
        env_file = ".env"

settings = Settings()
```

## Step 5: Create the Embedding Service

Create `app/embedding_service.py`:

```python
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service class for handling text embedding operations.
    Uses open-source embedding models for generating vector representations of text.
    """

    def __init__(self):
        """
        Initialize the embedding service.
        This initializes Sentence Transformers as an alternative to Context7.
        """
        logger.info("Initializing Embedding Service with Sentence Transformers")
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the Sentence Transformers embedding model.
        """
        logger.info("Loading all-MiniLM-L6-v2 embedding model")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
            return model
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers package is required for embedding functionality")

    def embed_text(self, text: str, chunk_size: int = 512) -> List[List[float]]:
        """
        Generate embeddings for the provided text.

        Args:
            text: The input text to embed
            chunk_size: Size of text chunks to process separately

        Returns:
            List of embedding vectors (one per chunk)
        """
        logger.info(f"Embedding text of length {len(text)} with chunk_size {chunk_size}")

        # Split text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        # Generate embeddings for all chunks at once using Sentence Transformers
        embeddings = self.model.encode(chunks).tolist()

        logger.info(f"Generated {len(embeddings)} embeddings for {len(chunks)} text chunks")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        This is typically used for search queries.

        Args:
            query: The query text to embed

        Returns:
            Single embedding vector
        """
        logger.info(f"Embedding query: {query[:50]}...")

        # Generate embedding for the query using Sentence Transformers
        embedding = self.model.encode([query])[0].tolist()

        logger.info("Query embedding generated successfully")
        return embedding

# Global instance of the embedding service
# In a real application, this might be managed by a dependency injection framework
embedding_service = EmbeddingService()

def get_embedding_service():
    """
    Get the global embedding service instance.
    """
    return embedding_service
```

## Step 6: Create the LLM Service

Create `app/llm_service.py`:

```python
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service class for handling language model operations.
    Uses free, open-source LLMs for generating responses based on context.
    """

    def __init__(self):
        """
        Initialize the LLM service.
        In a real implementation, this would initialize the connection to the LLM.
        """
        logger.info("Initializing LLM Service")
        # In a real implementation, this would connect to a free/open-source LLM
        # For example, Hugging Face models like Llama, Mistral, etc.
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the LLM model.
        In a real implementation, this would load a free/open-source model.
        """
        logger.info("Loading LLM model (placeholder)")
        # Placeholder for model loading
        return "placeholder_llm"

    def generate_response(self, query: str, context: Optional[List[Dict]] = None, history: Optional[List[Dict]] = None) -> str:
        """
        Generate a response based on the query, context, and conversation history.

        Args:
            query: The user's query
            context: Retrieved documents or context to inform the response
            history: Conversation history for context

        Returns:
            Generated response string
        """
        logger.info(f"Generating response for query: {query[:50]}...")

        # In a real implementation, this would call the LLM with the query, context, and history
        # For now, we'll simulate the response generation

        # Build a prompt that includes the query, context, and history
        prompt = self._build_prompt(query, context, history)

        # Generate response using the LLM
        response = self._generate_mock_response(prompt, query, context)

        logger.info("Response generated successfully")
        return response

    def _build_prompt(self, query: str, context: Optional[List[Dict]], history: Optional[List[Dict]]) -> str:
        """
        Build a prompt for the LLM that includes context and history.

        Args:
            query: The user's query
            context: Retrieved context documents
            history: Conversation history

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        # Add context if available
        if context:
            prompt_parts.append("Context information:")
            for i, ctx in enumerate(context):
                content = ctx.get('content', '')[:500]  # Limit context length
                prompt_parts.append(f"Document {i+1}: {content}")
            prompt_parts.append("")

        # Add conversation history if available
        if history:
            prompt_parts.append("Conversation history:")
            for turn in history:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")

        # Add the current query
        prompt_parts.append(f"User query: {query}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def _generate_mock_response(self, prompt: str, query: str, context: Optional[List[Dict]]) -> str:
        """
        Generate a mock response that simulates what an LLM would produce.
        This is a placeholder function that simulates LLM behavior.

        Args:
            prompt: The formatted prompt to send to the LLM
            query: The original user query
            context: The context provided to the LLM

        Returns:
            Generated response string
        """
        # In a real implementation, this would call the actual LLM
        # For example: return llm.generate(prompt)

        # Create a response that incorporates the query and context
        if context:
            # If we have context, create a response that references it
            context_snippets = [ctx.get('content', '')[:100] for ctx in context if 'content' in ctx]
            context_preview = " ".join(context_snippets)[:200]
            response = f"Based on the provided context: '{context_preview}...', I can help with your query about '{query[:30]}...'. Here's a relevant response incorporating that information."
        else:
            # If no context, create a more general response
            response = f"I understand you're asking about '{query[:50]}...'. Without specific context, I'm providing a general response based on my training data."

        return response

# Global instance of the LLM service
# In a real application, this might be managed by a dependency injection framework
llm_service = LLMService()

def get_llm_service():
    """
    Get the global LLM service instance.
    """
    return llm_service
```

## Step 7: Create the Vector Store Service

Create `app/vectorstore_service.py`:

```python
from typing import List, Dict, Optional
import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Service class for handling vector store operations with Qdrant.
    Manages document indexing, retrieval, and collection management.
    """

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, collection_name: str = "rag_documents"):
        """
        Initialize the vector store service.

        Args:
            url: Qdrant instance URL (defaults to localhost for development)
            api_key: Qdrant API key (if using cloud instance)
            collection_name: Name of the collection to use for storing documents
        """
        logger.info("Initializing Vector Store Service")

        # Use default local Qdrant for development if no URL provided
        if url is None:
            self.client = QdrantClient(host="localhost", port=6333)
        else:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=False  # Use REST for free tier
            )

        self.collection_name = collection_name

        # Initialize the collection if it doesn't exist
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Ensure the collection exists with appropriate vector parameters.
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                # Create collection with appropriate parameters
                # Using 1536 dimensions for OpenAI embeddings; adjust as needed
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def index_document(self, text: str, doc_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        Index a single document in the vector store.

        Args:
            text: The text content of the document
            doc_id: Optional document ID (will be auto-generated if not provided)
            metadata: Optional metadata to store with the document

        Returns:
            Document ID of the indexed document
        """
        logger.info(f"Indexing document with text length: {len(text)}")

        # Generate a unique ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # In a real implementation, we would get embeddings from the embedding service
        # For now, we'll use mock embeddings
        from .embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        embeddings = embedding_service.embed_text(text, chunk_size=512)

        # Prepare points for insertion
        points = []
        for i, embedding in enumerate(embeddings):
            # Each chunk gets its own point in the vector store
            point_id = f"{doc_id}_chunk_{i}"
            payload = {
                "text": text[i*512:(i+1)*512],  # Store the actual text chunk
                "original_doc_id": doc_id,
                "chunk_index": i,
                "metadata": metadata or {}
            }

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)

        # Insert points into the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Successfully indexed {len(points)} chunks for document {doc_id}")
        return doc_id

    def index_documents(self, documents: List[Dict]) -> List[str]:
        """
        Index multiple documents in the vector store.

        Args:
            documents: List of documents, each with 'text' and optional 'metadata' keys

        Returns:
            List of document IDs for the indexed documents
        """
        logger.info(f"Indexing {len(documents)} documents")

        doc_ids = []
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            doc_id = doc.get('id')

            indexed_id = self.index_document(text, doc_id, metadata)
            doc_ids.append(indexed_id)

        logger.info(f"Successfully indexed {len(documents)} documents")
        return doc_ids

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Perform a similarity search in the vector store.

        Args:
            query_vector: The embedding vector to search for similar documents
            top_k: Number of top results to return

        Returns:
            List of similar documents with their scores and content
        """
        logger.info(f"Performing similarity search with top_k={top_k}")

        # Perform the search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        # Format results
        results = []
        for result in search_results:
            formatted_result = {
                "id": result.id,
                "content": result.payload.get("text", ""),
                "score": result.score,
                "metadata": result.payload.get("metadata", {}),
                "original_doc_id": result.payload.get("original_doc_id", ""),
                "chunk_index": result.payload.get("chunk_index", 0)
            }
            results.append(formatted_result)

        logger.info(f"Search completed, returning {len(results)} results")
        return results

    def delete_document(self, doc_id: str):
        """
        Delete a document and all its chunks from the vector store.

        Args:
            doc_id: The ID of the document to delete
        """
        logger.info(f"Deleting document: {doc_id}")

        # Find all points with this document ID
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="original_doc_id",
                    match=models.MatchValue(value=doc_id)
                )
            ]
        )

        # Delete points matching the filter
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=filter_condition
            )
        )

        logger.info(f"Successfully deleted document: {doc_id}")

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a document by its ID.

        Args:
            doc_id: The ID of the document to retrieve

        Returns:
            Document content and metadata, or None if not found
        """
        logger.info(f"Retrieving document: {doc_id}")

        # Find all points with this document ID
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="original_doc_id",
                    match=models.MatchValue(value=doc_id)
                )
            ]
        )

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_condition,
            limit=1000  # Assuming a document won't have more than 1000 chunks
        )

        if results[0]:
            # Combine all chunks of the document
            chunks = []
            for point in results[0]:
                chunks.append({
                    "text": point.payload.get("text", ""),
                    "chunk_index": point.payload.get("chunk_index", 0)
                })

            # Sort chunks by index to reconstruct the document
            chunks.sort(key=lambda x: x["chunk_index"])

            full_text = "".join(chunk["text"] for chunk in chunks)

            return {
                "id": doc_id,
                "text": full_text,
                "metadata": results[0][0].payload.get("metadata", {})
            }

        logger.info(f"Document {doc_id} not found")
        return None

# Global instance of the vector store service
# In a real application, this might be managed by a dependency injection framework
vectorstore_service = None

def get_vectorstore_service():
    """
    Get the global vector store service instance.
    Initializes the service if it doesn't exist.
    """
    global vectorstore_service
    if vectorstore_service is None:
        from config.settings import settings
        vectorstore_service = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.collection_name
        )
    return vectorstore_service
```

## Step 8: Create the Database Service

Create `app/database_service.py`:

```python
from typing import List, Dict, Optional
import logging
import asyncpg
import uuid
import json
from datetime import datetime
from config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Service class for handling database operations with Neon Postgres.
    Manages document storage, retrieval, and chat history management.
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the database service.

        Args:
            db_url: Database connection URL (defaults to environment variable)
        """
        logger.info("Initializing Database Service")

        # Use default from settings if no URL provided
        if db_url is None:
            db_url = settings.neon_db_url

        self.db_url = db_url
        self.pool = None

    async def initialize(self):
        """
        Initialize the database connection pool and create required tables.
        """
        logger.info("Initializing database connection pool")

        # Create connection pool
        self.pool = await asyncpg.create_pool(
            dsn=self.db_url,
            min_size=1,
            max_size=10,
            command_timeout=60
        )

        # Create required tables
        await self._create_tables()
        logger.info("Database service initialized successfully")

    async def _create_tables(self):
        """
        Create required tables if they don't exist.
        """
        logger.info("Creating database tables if they don't exist")

        # Documents table
        await self.pool.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Chat history table
        await self.pool.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT NOW(),
                context_metadata JSONB
            )
        """)

        logger.info("Database tables created successfully")

    async def store_document(self, content: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a raw document in the database.

        Args:
            content: The text content of the document
            metadata: Optional metadata to store with the document

        Returns:
            Document ID of the stored document
        """
        logger.info(f"Storing document with content length: {len(content)}")

        doc_id = str(uuid.uuid4())

        # Insert document into the database
        await self.pool.execute(
            """
            INSERT INTO documents (id, content, metadata)
            VALUES ($1, $2, $3)
            """,
            doc_id,
            content,
            json.dumps(metadata) if metadata else None
        )

        logger.info(f"Successfully stored document with ID: {doc_id}")
        return doc_id

    async def store_documents(self, documents: List[Dict]) -> List[str]:
        """
        Store multiple documents in the database.

        Args:
            documents: List of documents, each with 'content' and optional 'metadata' keys

        Returns:
            List of document IDs for the stored documents
        """
        logger.info(f"Storing {len(documents)} documents")

        doc_ids = []
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            doc_id = await self.store_document(content, metadata)
            doc_ids.append(doc_id)

        logger.info(f"Successfully stored {len(documents)} documents")
        return doc_ids

    async def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a document by its ID.

        Args:
            doc_id: The ID of the document to retrieve

        Returns:
            Document content and metadata, or None if not found
        """
        logger.info(f"Retrieving document: {doc_id}")

        row = await self.pool.fetchrow(
            """
            SELECT id, content, metadata, created_at, updated_at
            FROM documents
            WHERE id = $1
            """,
            doc_id
        )

        if row:
            return {
                "id": str(row['id']),
                "content": row['content'],
                "metadata": row['metadata'],
                "created_at": row['created_at'],
                "updated_at": row['updated_at']
            }

        logger.info(f"Document {doc_id} not found")
        return None

    async def get_all_documents(self) -> List[Dict]:
        """
        Retrieve all documents from the database.

        Returns:
            List of all documents
        """
        logger.info("Retrieving all documents")

        rows = await self.pool.fetch(
            """
            SELECT id, content, metadata, created_at, updated_at
            FROM documents
            ORDER BY created_at DESC
            """
        )

        documents = []
        for row in rows:
            documents.append({
                "id": str(row['id']),
                "content": row['content'],
                "metadata": row['metadata'],
                "created_at": row['created_at'],
                "updated_at": row['updated_at']
            })

        logger.info(f"Retrieved {len(documents)} documents")
        return documents

    async def update_document(self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Update an existing document.

        Args:
            doc_id: The ID of the document to update
            content: New content (optional, keeps existing if None)
            metadata: New metadata (optional, keeps existing if None)
        """
        logger.info(f"Updating document: {doc_id}")

        # Get current document to preserve unchanged fields
        current_doc = await self.get_document(doc_id)
        if not current_doc:
            logger.warning(f"Document {doc_id} not found for update")
            return

        # Prepare update values
        new_content = content if content is not None else current_doc['content']
        new_metadata = metadata if metadata is not None else current_doc['metadata']

        # Update the document
        await self.pool.execute(
            """
            UPDATE documents
            SET content = $1, metadata = $2, updated_at = NOW()
            WHERE id = $3
            """,
            new_content,
            json.dumps(new_metadata) if new_metadata else None,
            doc_id
        )

        logger.info(f"Successfully updated document: {doc_id}")

    async def delete_document(self, doc_id: str):
        """
        Delete a document from the database.

        Args:
            doc_id: The ID of the document to delete
        """
        logger.info(f"Deleting document: {doc_id}")

        await self.pool.execute(
            """
            DELETE FROM documents
            WHERE id = $1
            """,
            doc_id
        )

        logger.info(f"Successfully deleted document: {doc_id}")

    async def close(self):
        """
        Close the database connection pool.
        """
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

# Global instance of the database service
# In a real application, this might be managed by a dependency injection framework
database_service = None

async def get_database_service():
    """
    Get the global database service instance.
    Initializes the service if it doesn't exist.
    """
    global database_service
    if database_service is None:
        database_service = DatabaseService()
        await database_service.initialize()
    return database_service
```

## Step 9: Create the Main Application

Create `app/main.py`:

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging

# Import the services
from .embedding_service import get_embedding_service
from .llm_service import get_llm_service
from .vectorstore_service import get_vectorstore_service
from .database_service import get_database_service

app = FastAPI(title="RAG Chatbot API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the service instances
embedding_service = get_embedding_service()
llm_service = get_llm_service()
vectorstore_service = get_vectorstore_service()
# Note: database_service is async, so we'll get it in the endpoints that need it

# Request and response models
class EmbedRequest(BaseModel):
    text: str
    chunk_size: Optional[int] = 512

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    chunk_count: int

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    results: List[dict]

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str

class SelectedTextRequest(BaseModel):
    text: str
    context: Optional[str] = ""

class SelectedTextResponse(BaseModel):
    response: str

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """
    Embed text chunks using Sentence Transformers.
    This endpoint takes text input and returns vector embeddings for each chunk.
    """
    try:
        logger.info(f"Embedding text of length {len(request.text)} with chunk size {request.chunk_size}")

        # Use the embedding service to generate embeddings
        embeddings = embedding_service.embed_text(request.text, request.chunk_size)

        response = EmbedResponse(
            embeddings=embeddings,
            chunk_count=len(embeddings)
        )

        logger.info(f"Successfully embedded {len(embeddings)} chunks")
        return response
    except Exception as e:
        logger.error(f"Error in embed_text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the vector store for relevant documents.
    This endpoint takes a query and returns the most similar documents.
    """
    try:
        logger.info(f"Querying for: {request.query[:50]}...")

        # Embed the query using the embedding service
        query_vector = embedding_service.embed_query(request.query)

        # Search the vector store for similar vectors
        results = vectorstore_service.search(query_vector, request.top_k)

        response = QueryResponse(results=results)
        logger.info("Successfully queried documents")
        return response
    except Exception as e:
        logger.error(f"Error in query_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Generate a response to a user message using the LLM.
    This endpoint takes a message and optional conversation history,
    retrieves relevant context, and generates a response.
    """
    try:
        logger.info(f"Chat request with message: {request.message[:50]}...")

        # Query for relevant context based on the message
        query_results = vectorstore_service.search(
            embedding_service.embed_query(request.message),
            top_k=3
        )

        # Generate response using the LLM service
        response_text = llm_service.generate_response(
            query=request.message,
            context=query_results,
            history=request.history
        )

        response = ChatResponse(response=response_text)
        logger.info("Successfully generated chat response")
        return response
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/selected_text", response_model=SelectedTextResponse)
async def process_selected_text(request: SelectedTextRequest):
    """
    Process user-selected text and generate a response based on it.
    This endpoint handles text that a user has highlighted/selected in the frontend.
    """
    try:
        logger.info(f"Processing selected text: {request.text[:50]}...")

        # Embed the selected text
        query_vector = embedding_service.embed_query(request.text)

        # Search for related content in the vector store
        related_results = vectorstore_service.search(query_vector, top_k=3)

        # Generate a response focused on the selected text
        response_text = llm_service.generate_response(
            query=request.text,
            context=related_results,
            history=[]  # No history for selected text processing
        )

        response = SelectedTextResponse(response=response_text)
        logger.info("Successfully processed selected text")
        return response
    except Exception as e:
        logger.error(f"Error in process_selected_text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Selected text processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy", "service": "rag-chatbot-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Step 10: Run the Application

Create a `.env` file to store your configuration (if needed):

```env
QDRANT_URL=http://localhost:6333
NEON_DB_URL=postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require
```

Run the application:

```bash
# Make sure you're in the rag-chatbot-backend directory
cd rag-chatbot-backend

# Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend should now be running at `http://localhost:8000`, with the API documentation available at `http://localhost:8000/docs`.

## Testing the Backend

You can test the backend endpoints using curl or a tool like Postman:

```bash
# Test the health endpoint
curl http://localhost:8000/health

# Test the embed endpoint
curl -X POST "http://localhost:8000/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a test document for embedding."}'

# Test the query endpoint
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is this document about?"}'
```