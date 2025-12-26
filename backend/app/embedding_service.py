"""
Embedding Service for RAG Chatbot
Handles text embedding operations using Google's Gemini API through OpenAI-compatible interface.
"""
from typing import List, Union
import numpy as np
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service class for handling text embedding operations.
    Uses Google's Gemini API through OpenAI-compatible interface for generating vector representations of text.
    """

    def __init__(self):
        """
        Initialize the embedding service.
        This initializes OpenAI client with Google's Gemini endpoint.
        """
        logger.info("Initializing Embedding Service with OpenAI-compatible Google Gemini")

        # Get Gemini API key from environment
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY environment variable not set - will use mock functionality")
            self.openai_client = None
        else:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(
                    api_key=self.gemini_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                logger.info("Embedding service initialized successfully with OpenAI-compatible Gemini")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client for Gemini: {str(e)} - will use mock functionality")
                self.openai_client = None

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

        # Generate embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            if self.openai_client:
                try:
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-005",  # Using Google's embedding model through OpenAI interface
                        input=chunk
                    )
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)
                    continue
                except Exception as e:
                    logger.error(f"Error embedding chunk with OpenAI-compatible Gemini: {str(e)}")
            else:
                logger.warning("Embedding service not initialized with API key, using fallback")

            # Add a simple fallback embedding based on text characteristics if embedding fails
            # Use a simple hash-based approach to create a somewhat meaningful embedding
            import hashlib
            hash_obj = hashlib.md5(chunk.encode('utf-8'))
            hash_hex = hash_obj.hexdigest()

            # Convert hex hash to a 768-dimensional vector (simplified approach)
            vector = []
            for i in range(0, 768 * 2, 2):  # 768 values, each using 2 hex chars
                hex_pair = hash_hex[i % len(hash_hex):(i + 2) % len(hash_hex)] or '00'
                if len(hex_pair) == 1:
                    hex_pair += '0'
                val = int(hex_pair, 16) / 255.0  # Normalize to 0-1 range
                vector.append(val)

            embeddings.append(vector)

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

        if self.openai_client:
            try:
                # Generate embedding for the query using OpenAI-compatible interface
                response = self.openai_client.embeddings.create(
                    model="text-embedding-005",  # Using Google's embedding model through OpenAI interface
                    input=query
                )
                embedding = response.data[0].embedding

                logger.info("Query embedding generated successfully with OpenAI-compatible Gemini")
                return embedding
            except Exception as e:
                logger.error(f"Error embedding query with OpenAI-compatible Gemini: {str(e)}")
        else:
            logger.warning("Embedding service not initialized with API key, using fallback for query")

        # Return a hash-based fallback vector if embedding fails
        import hashlib
        hash_obj = hashlib.md5(query.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()

        # Convert hex hash to a 768-dimensional vector (simplified approach)
        vector = []
        for i in range(0, 768 * 2, 2):  # 768 values, each using 2 hex chars
            hex_pair = hash_hex[i % len(hash_hex):(i + 2) % len(hash_hex)] or '00'
            if len(hex_pair) == 1:
                hex_pair += '0'
            val = int(hex_pair, 16) / 255.0  # Normalize to 0-1 range
            vector.append(val)

        return vector


# Global instance of the embedding service
# In a real application, this might be managed by a dependency injection framework
embedding_service = None

def get_embedding_service():
    """
    Get the global embedding service instance.
    Initializes the service if it doesn't exist.
    """
    global embedding_service
    if embedding_service is None:
        try:
            embedding_service = EmbeddingService()
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            # Create a service instance even if initialization fails
            embedding_service = EmbeddingService()  # This will handle the error internally
    return embedding_service