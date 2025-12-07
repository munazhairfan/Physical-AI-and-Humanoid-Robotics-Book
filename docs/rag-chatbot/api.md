---
title: "RAG Chatbot: API"
description: "API design and implementation for RAG-based chatbot in robotics education"
sidebar_position: 2
slug: /rag-chatbot/api
keywords: [RAG, API, chatbot, robotics, education, backend, FastAPI]
---

# RAG Chatbot: API

## Overview

This section details the API design and implementation for the Retrieval-Augmented Generation (RAG) chatbot system. The API provides endpoints for querying the robotics textbook knowledge base, managing conversation history, and handling various types of robotics-related questions.

## API Architecture

### Tech Stack

The API is built using modern Python web technologies:

```python
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
qdrant-client==1.7.0
sentence-transformers==2.2.2
openai==1.3.5
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
```

### Core Dependencies

```python
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import logging
from datetime import datetime
import uuid

# Local imports
from .embedding import VectorStore, EmbeddingGenerator
from .retriever import ContentRetriever
from .generator import ResponseGenerator
```

## API Endpoints

### 1. Query Endpoint

The main endpoint for asking questions about robotics content:

```python
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or query about robotics content", min_length=1, max_length=1000)
    conversation_id: Optional[str] = Field(None, description="ID of the conversation for context")
    max_results: Optional[int] = Field(5, ge=1, le=10, description="Maximum number of results to retrieve")
    include_context: Optional[bool] = Field(True, description="Whether to include retrieved context in response")

class QueryResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[Dict[str, Any]]
    confidence: float
    timestamp: datetime

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main endpoint for querying the robotics knowledge base
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Retrieve relevant context
        retriever = ContentRetriever()
        context_chunks = await retriever.retrieve(
            query=request.query,
            max_results=request.max_results
        )

        # Generate response using RAG
        generator = ResponseGenerator()
        response = await generator.generate_response(
            query=request.query,
            context_chunks=context_chunks,
            conversation_id=conversation_id
        )

        # Format sources
        sources = [
            {
                "content": chunk["content"][:200] + "...",  # Truncate for brevity
                "metadata": chunk["metadata"],
                "relevance_score": chunk["score"]
            }
            for chunk in context_chunks
        ]

        return QueryResponse(
            response=response,
            conversation_id=conversation_id,
            sources=sources,
            confidence=min(1.0, len(sources) / request.max_results),  # Simple confidence metric
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")
```

### 2. Conversation Management

Endpoints for managing conversation history:

```python
class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Dict[str, str]]  # List of {role, content} pairs
    created_at: datetime
    updated_at: datetime

@app.get("/api/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str):
    """
    Retrieve conversation history
    """
    try:
        # In a real implementation, this would fetch from a database
        # For now, we'll simulate with a simple in-memory store
        conversation = await get_conversation_from_store(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return conversation
    except Exception as e:
        logging.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving conversation")

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation
    """
    try:
        success = await delete_conversation_from_store(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logging.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting conversation")
```

### 3. Content Search

Direct search functionality for exploring content:

```python
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    max_results: Optional[int] = Field(10, ge=1, le=20, description="Maximum number of results")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    query_time: float

@app.post("/api/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Search the robotics knowledge base directly
    """
    try:
        start_time = datetime.utcnow()

        # Create embedding for the search query
        embedder = EmbeddingGenerator()
        query_embedding = embedder.generate_embeddings([request.query])[0]

        # Search vector store
        vector_store = VectorStore()
        results = vector_store.search(
            query_embedding=query_embedding,
            limit=request.max_results
        )

        # Apply filters if provided
        if request.filters:
            results = apply_filters(results, request.filters)

        end_time = datetime.utcnow()
        query_time = (end_time - start_time).total_seconds()

        return SearchResponse(
            results=results,
            total_count=len(results),
            query_time=query_time
        )
    except Exception as e:
        logging.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail="Error performing search")
```

### 4. Content Management

Endpoints for managing the knowledge base:

```python
from fastapi import UploadFile, File
import tempfile
import os

class ContentUploadResponse(BaseModel):
    document_id: str
    chunks_processed: int
    processing_time: float

@app.post("/api/content/upload", response_model=ContentUploadResponse)
async def upload_content(
    file: UploadFile = File(...),
    metadata: Optional[str] = None  # JSON string of metadata
):
    """
    Upload and process new content for the knowledge base
    """
    try:
        start_time = datetime.utcnow()

        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')

        # Parse metadata if provided
        import json
        meta_dict = json.loads(metadata) if metadata else {}
        meta_dict["filename"] = file.filename
        meta_dict["upload_date"] = datetime.utcnow().isoformat()

        # Process and embed the content
        from .embedding import EmbeddingPipeline
        pipeline = EmbeddingPipeline()
        stats = pipeline.process_document(content_str, meta_dict)

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        return ContentUploadResponse(
            document_id=meta_dict.get("doc_id", str(uuid.uuid4())),
            chunks_processed=stats["num_chunks"],
            processing_time=processing_time
        )
    except Exception as e:
        logging.error(f"Error uploading content: {str(e)}")
        raise HTTPException(status_code=500, detail="Error uploading content")

@app.get("/api/content/stats")
async def get_content_stats():
    """
    Get statistics about the knowledge base
    """
    try:
        vector_store = VectorStore()
        stats = vector_store.get_collection_stats()

        return {
            "total_chunks": stats["total_points"],
            "total_documents": stats["total_documents"],  # This would need to be tracked separately
            "last_updated": stats["last_update_time"]
        }
    except Exception as e:
        logging.error(f"Error getting content stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting content stats")
```

## Implementation Classes

### Content Retriever

```python
import asyncio
from typing import List, Dict, Any

class ContentRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = EmbeddingGenerator()

    async def retrieve(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content chunks for a query
        """
        # Generate embedding for query
        query_embedding = self.embedder.generate_embeddings([query])[0]

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=max_results
        )

        return results

    async def retrieve_with_reranking(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank results for better relevance
        """
        # Initial retrieval
        initial_results = await self.retrieve(query, max_results * 2)

        # Simple reranking based on multiple factors
        reranked_results = self._rerank_results(query, initial_results)

        return reranked_results[:max_results]

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results based on multiple relevance factors
        """
        scored_results = []

        for result in results:
            # Calculate multiple relevance scores
            lexical_score = self._calculate_lexical_score(query, result["content"])
            semantic_score = result["score"]  # The original embedding similarity score
            metadata_score = self._calculate_metadata_score(query, result["metadata"])

            # Combine scores with weights
            combined_score = (
                0.3 * lexical_score +
                0.6 * semantic_score +
                0.1 * metadata_score
            )

            result["combined_score"] = combined_score
            scored_results.append(result)

        # Sort by combined score
        return sorted(scored_results, key=lambda x: x["combined_score"], reverse=True)

    def _calculate_lexical_score(self, query: str, content: str) -> float:
        """
        Calculate lexical similarity score
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)

    def _calculate_metadata_score(self, query: str, metadata: Dict) -> float:
        """
        Calculate score based on metadata relevance
        """
        score = 0.0

        # Check if query contains module/section keywords
        query_lower = query.lower()

        if "module" in metadata:
            if metadata["module"].lower() in query_lower:
                score += 0.5

        if "section" in metadata:
            if metadata["section"].lower() in query_lower:
                score += 0.3

        # Check content type relevance
        if "type" in metadata:
            if metadata["type"] in ["code", "example", "equation"]:
                # These might be less relevant for conceptual questions
                score += 0.1 if "how" in query_lower or "implement" in query_lower else 0.0

        return min(1.0, score)
```

### Response Generator

```python
import openai
from typing import List, Dict, Any

class ResponseGenerator:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        # In production, use environment variables for API keys
        # openai.api_key = os.getenv("OPENAI_API_KEY")

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_id: str
    ) -> str:
        """
        Generate a response using the retrieved context
        """
        # Build context from retrieved chunks
        context = self._build_context(context_chunks)

        # Create system message with robotics education context
        system_message = {
            "role": "system",
            "content": (
                "You are an expert robotics educator and assistant. "
                "Use the provided context to answer questions about robotics concepts. "
                "Be precise, educational, and provide code examples when relevant. "
                "If you don't have enough information in the context, say so clearly. "
                "Always maintain a teaching-focused approach."
            )
        }

        # Create user message with query and context
        user_message = {
            "role": "user",
            "content": (
                f"Context: {context}\n\n"
                f"Question: {query}\n\n"
                f"Please provide a comprehensive answer based on the context, "
                f"including any relevant equations, code examples, or conceptual explanations."
            )
        }

        # In a real implementation, you would call the LLM API
        # For this example, we'll simulate the response
        response = await self._simulate_llm_response(query, context)

        return response

    def _build_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a context string from retrieved chunks
        """
        context_parts = []

        for i, chunk in enumerate(context_chunks):
            metadata = chunk["metadata"]
            module = metadata.get("module", "Unknown Module")
            section = metadata.get("section", "Unknown Section")

            context_parts.append(
                f"Source {i+1} ({module} - {section}): {chunk['content']}"
            )

        return "\n\n".join(context_parts)

    async def _simulate_llm_response(self, query: str, context: str) -> str:
        """
        Simulate LLM response for demonstration purposes
        """
        # This is a placeholder - in reality, you'd call an LLM API
        import time
        await asyncio.sleep(0.1)  # Simulate API call delay

        # Simple rule-based response for demonstration
        if "kinematics" in query.lower():
            return (
                "Based on the context provided, forward kinematics is the process of "
                "calculating the position and orientation of the end-effector based on "
                "joint angles. Inverse kinematics is the reverse process - determining "
                "joint angles required to achieve a desired end-effector position. "
                "Both are fundamental concepts in robotics for controlling robot arms."
            )
        elif "control" in query.lower():
            return (
                "Control systems in robotics typically involve feedback mechanisms to "
                "regulate the behavior of robotic systems. PID controllers are commonly "
                "used, with the equation: u(t) = K_p e(t) + K_i ∫e(t)dt + K_d de(t)/dt. "
                "Model Predictive Control (MPC) is another advanced technique that "
                "optimizes control actions over a prediction horizon."
            )
        else:
            return (
                f"I found some relevant information in the robotics textbook:\n\n{context[:500]}...\n\n"
                f"Based on this, the answer to your question '{query}' is that this is "
                f"a simulated response. In a real implementation, an LLM would generate "
                f"a comprehensive answer using the full context."
            )

    async def generate_explanation(self, topic: str, difficulty: str = "intermediate") -> str:
        """
        Generate a structured explanation of a robotics topic
        """
        prompt = (
            f"Provide a {difficulty}-level explanation of {topic} in robotics. "
            f"Include: 1) Definition, 2) Key concepts, 3) Practical applications, "
            f"4) Mathematical formulation if applicable, 5) Code example if relevant."
        )

        # In real implementation: call LLM with this prompt
        return await self._simulate_llm_response(prompt, "")
```

## Error Handling and Validation

### Custom Exceptions

```python
from fastapi import HTTPException, status

class RAGException(Exception):
    """Base exception for RAG-related errors"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class QueryProcessingError(RAGException):
    """Raised when there's an error processing a query"""
    def __init__(self, message: str = "Error processing query"):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

class ContentNotFoundError(RAGException):
    """Raised when requested content is not found"""
    def __init__(self, message: str = "Content not found"):
        super().__init__(message, status.HTTP_404_NOT_FOUND)

class RateLimitError(RAGException):
    """Raised when rate limits are exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status.HTTP_429_TOO_MANY_REQUESTS)
```

### Input Validation

```python
from pydantic import field_validator

class ValidatedQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    conversation_id: Optional[str] = Field(None, min_length=1, max_length=100)
    max_results: int = Field(5, ge=1, le=10)
    include_context: bool = True

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Query must be at least 3 characters long')
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

    @field_validator('conversation_id')
    @classmethod
    def validate_conversation_id(cls, v):
        if v is not None and len(v) < 1:
            raise ValueError('Conversation ID cannot be empty')
        return v
```

## Middleware and Security

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/query")
@limiter.limit("10/minute")
async def rate_limited_query(request: Request, query_request: QueryRequest):
    # Implementation here
    pass
```

## Performance Monitoring

### Request Logging

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Log request details
        logging.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")

        return response

app.add_middleware(RequestLoggingMiddleware)
```

## API Documentation

### Custom API Routes

```python
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/api/docs")
async def api_documentation():
    """
    API documentation endpoint
    """
    return {
        "title": "Robotics RAG Chatbot API",
        "version": "1.0.0",
        "description": "API for querying robotics textbook content using RAG",
        "endpoints": [
            {
                "path": "/api/query",
                "method": "POST",
                "description": "Query the robotics knowledge base"
            },
            {
                "path": "/api/search",
                "method": "POST",
                "description": "Direct search of the knowledge base"
            },
            {
                "path": "/api/conversations/{id}",
                "method": "GET",
                "description": "Get conversation history"
            }
        ]
    }
```

## Complete API Server

```python
import uvicorn
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    """
    # Startup
    logging.info("Starting RAG Chatbot API...")

    # Initialize vector store connection
    vector_store = VectorStore()

    yield

    # Shutdown
    logging.info("Shutting down RAG Chatbot API...")

app = FastAPI(
    title="Robotics RAG Chatbot API",
    description="API for querying robotics textbook content using Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan
)

# Include all the routes defined above...

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Only in development
    )
```

## Summary

This API provides a comprehensive interface for the RAG-based chatbot system in robotics education. It includes endpoints for querying, conversation management, content search, and content management, with proper error handling, validation, and performance monitoring.

Continue with [UI Integration](./ui-integration) to learn about integrating the chatbot into the Docusaurus documentation interface.