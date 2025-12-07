"""
Vector Store Service for RAG Chatbot
Handles operations with the Qdrant vector database for document indexing and retrieval.
"""
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
            # Use in-memory mode for development without requiring a running server
            self.client = QdrantClient(":memory:")
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
                # Using 768 dimensions for Google Gemini embeddings
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
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

        # Determine the correct method based on client capabilities
        # Check if this is an in-memory client (doesn't have search method)
        if not hasattr(self.client, 'search'):
            # Use query_points for in-memory client
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            search_points = search_results.points  # query_points returns QueryResponse
        else:
            # Use search for regular HTTP/gRPC client
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            search_points = search_results  # search returns list of ScoredPoint

        # Format results
        results = []
        for result in search_points:
            formatted_result = {
                "id": result.id,
                "content": result.payload.get("text", "") if result.payload else "",
                "score": result.score,
                "metadata": result.payload.get("metadata", {}) if result.payload else {},
                "original_doc_id": result.payload.get("original_doc_id", "") if result.payload else "",
                "chunk_index": result.payload.get("chunk_index", 0) if result.payload else 0
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
        # Force in-memory mode for development to avoid dependency on external services
        url = None  # This will trigger in-memory mode in VectorStoreService.__init__
        api_key = None
        vectorstore_service = VectorStoreService(url=url, api_key=api_key)
    return vectorstore_service