"""
Improved Vector Store Service for RAG Chatbot
Handles operations with the Qdrant vector database with better chunking and retrieval
"""
from typing import List, Dict, Optional
import logging
import uuid
import re
import html

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Improved Service class for handling vector store operations with Qdrant.
    Uses semantic chunking for better retrieval quality.
    """

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, collection_name: str = "rag_documents"):
        """
        Initialize the vector store service.
        """
        logger.info("Initializing Improved Vector Store Service with semantic chunking")

        # Check for Railway-specific Qdrant configuration first
        import os
        qdrant_host = os.getenv("QDRANT_HOST")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Check if we're in Railway environment with external Qdrant configured
        if qdrant_host:
            # Use external Qdrant for Railway (persistence required)
            qdrant_url = f"https://{qdrant_host}:{qdrant_port}"
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models
                from qdrant_client.http.models import Distance, VectorParams, PointStruct

                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    prefer_grpc=False  # Use REST for compatibility
                )
                logger.info(f"Using external Qdrant for Railway: {qdrant_url}")

                # Set collection name and initialize collection
                self.collection_name = collection_name
                self._ensure_collection_exists()
            except Exception as e:
                logger.error(f"Failed to connect to external Qdrant: {str(e)}")
                logger.info("Falling back to in-memory mode")
                self.client = QdrantClient(":memory:")
                self.collection_name = collection_name
        else:
            # Use local Qdrant for development
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models
                from qdrant_client.http.models import Distance, VectorParams, PointStruct

                # Try to connect to local Qdrant instance
                self.client = QdrantClient(host="localhost", port=6333)
                logger.info("Connected to local Qdrant instance")
            except Exception as e:
                logger.error(f"Failed to connect to local Qdrant: {str(e)}")
                logger.info("Falling back to in-memory mode")
                self.client = QdrantClient(":memory:")

            # Set collection name and initialize collection
            self.collection_name = collection_name
            self._ensure_collection_exists()

    def _remove_markdown_formatting(self, markdown_content: str) -> str:
        """
        Convert markdown content to plain text by removing markdown formatting.
        """
        if not markdown_content:
            return ""

        # Remove HTML tags if any
        text = html.unescape(markdown_content)

        # Remove markdown headers (### Header -> Header)
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

        # Remove bold and italic formatting (**text**, *text*, __text__, _text_)
        text = re.sub(r'\*{2}([^*]+)\*{2}', r'\1', text)  # **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)       # *italic*
        text = re.sub(r'_{2}([^_]+)_{2}', r'\1', text)   # __bold__
        text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_

        # Remove code blocks (```code```)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

        # Remove inline code (`code`)
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Remove links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove images ![alt](url) -> alt
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)

        # Remove blockquotes
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

        # Remove horizontal rules
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

        # Remove reference-style links [text][1] and reference definitions [1]: url
        text = re.sub(r'\[([^\]]+)\]\[[^\]]+\]', r'\1', text)  # [text][1] -> text
        text = re.sub(r'\n\[.+\]:.+\n', '\n', text)  # Remove reference definitions

        # Replace common markdown symbols
        text = re.sub(r'\\', '', text)  # Remove escape characters

        # Remove extra whitespace and normalize
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple blank lines with single
        text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces to single space
        text = text.strip()

        # Clean up any remaining markdown artifacts
        text = re.sub(r'\n\s*-', '\n- ', text)  # Ensure proper list formatting
        text = re.sub(r'\n\s#\s', '\n', text)   # Remove any remaining header markers

        return text

    def _semantic_chunking(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into semantically coherent chunks.
        """
        if not text.strip():
            return []

        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            # If no paragraphs, fall back to sentences
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks

        # Process paragraphs into chunks
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If the paragraph itself is too large, split it
                if len(paragraph) > max_chunk_size:
                    sub_chunks = self._split_large_paragraph(paragraph, max_chunk_size, overlap)
                    chunks.extend(sub_chunks[:-1])  # Add all but the last sub-chunk
                    current_chunk = sub_chunks[-1] if sub_chunks else ""  # Keep the last part
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_large_paragraph(self, paragraph: str, max_chunk_size: int, overlap: int) -> List[str]:
        """
        Split a large paragraph into smaller chunks.
        """
        sentences = re.split(r'[.!?]+\s+', paragraph)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk with some overlap
                if len(sentence) > max_chunk_size:
                    # If sentence is too long, split by words
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) < max_chunk_size:
                            temp_chunk += " " + word if temp_chunk else word
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _ensure_collection_exists(self):
        """
        Ensure that the collection exists in Qdrant.
        Creates the collection if it doesn't exist.
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with appropriate vector size (768 to match Gemini embeddings)
                from qdrant_client.http import models
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            # In memory mode, collection creation is automatic

    def index_document(self, text: str, doc_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        Index a document in the vector store using semantic chunking.
        """
        logger.info(f"Indexing document with semantic chunking, text length: {len(text)}")

        # Generate a unique ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        if self.client is None:
            # Mock implementation when Qdrant client is not available
            logger.warning("Mock vectorstore: Would index document")
            return doc_id

        try:
            from .embedding_service import get_embedding_service
            embedding_service = get_embedding_service()

            # Use semantic chunking instead of fixed-size chunks
            text_chunks = self._semantic_chunking(text, max_chunk_size=1000, overlap=200)
            logger.info(f"Split document into {len(text_chunks)} semantic chunks")

            # Prepare points for insertion
            points = []
            for i, chunk in enumerate(text_chunks):
                # Generate embedding for this chunk
                chunk_embeddings = embedding_service.embed_text(chunk, chunk_size=512)

                for j, embedding in enumerate(chunk_embeddings):
                    # Each chunk gets its own point in the vector store
                    import uuid as uuid_lib
                    from qdrant_client.http.models import PointStruct
                    point_id = str(uuid_lib.uuid4())

                    # Create payload with the chunk and metadata
                    payload = {
                        "text": chunk,
                        "original_doc_id": doc_id,
                        "chunk_index": i,
                        "chunk_part": j,
                        "metadata": metadata or {},
                        "doc_length": len(text),
                        "total_chunks": len(text_chunks)
                    }

                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                    points.append(point)

            # Insert points into the collection
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

            logger.info(f"Successfully indexed document {doc_id} with {len(points)} points from {len(text_chunks)} semantic chunks")
            return doc_id
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            # In case of error, return the doc_id anyway
            return doc_id

    def index_documents(self, documents: List[Dict]) -> List[str]:
        """
        Index multiple documents in the vector store.
        """
        logger.info(f"Indexing {len(documents)} documents")

        indexed_ids = []
        for doc in documents:
            text = doc.get('text', '')
            doc_id = doc.get('doc_id')
            metadata = doc.get('metadata', {})

            indexed_id = self.index_document(text, doc_id, metadata)
            indexed_ids.append(indexed_id)

        logger.info(f"Successfully indexed {len(indexed_ids)} documents")
        return indexed_ids

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Perform a similarity search in the vector store.
        """
        logger.info(f"Performing similarity search with top_k={top_k}")

        if self.client is None:
            # Mock implementation when Qdrant client is not available
            logger.warning("Mock vectorstore: Returning empty search results")
            return []

        try:
            from qdrant_client.http import models
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
                content = result.payload.get("text", "") if result.payload else ""
                # Process content to remove any remaining markdown formatting
                content = self._remove_markdown_formatting(content)

                formatted_result = {
                    "id": result.id,
                    "content": content,
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {}) if result.payload else {},
                    "original_doc_id": result.payload.get("original_doc_id", "") if result.payload else "",
                    "chunk_index": result.payload.get("chunk_index", 0) if result.payload else 0,
                    "chunk_part": result.payload.get("chunk_part", 0) if result.payload else 0,
                    "doc_length": result.payload.get("doc_length", 0) if result.payload else 0,
                    "total_chunks": result.payload.get("total_chunks", 0) if result.payload else 0
                }
                results.append(formatted_result)

            logger.info(f"Search completed, returning {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            # Return empty results in case of error
            return []

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        """
        logger.info(f"Deleting document: {doc_id}")

        if self.client is None:
            # Mock implementation when Qdrant client is not available
            logger.warning("Mock vectorstore: Would delete document")
            return True

        try:
            # Find all points with this original_doc_id and delete them
            from qdrant_client.http import models
            scroll_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="original_doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=1000  # Assuming max 1000 chunks per document
            )

            point_ids = [point.id for point in scroll_results[0]]

            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                logger.info(f"Successfully deleted document {doc_id} with {len(point_ids)} chunks")
                return True
            else:
                logger.warning(f"No points found for document {doc_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a document from the vector store by its ID.
        """
        logger.info(f"Retrieving document: {doc_id}")

        if self.client is None:
            # Mock implementation when Qdrant client is not available
            logger.warning("Mock vectorstore: Would return document")
            return None

        try:
            from qdrant_client.http import models
            scroll_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="original_doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=1000  # Assuming max 1000 chunks per document
            )

            points = scroll_results[0]
            if points:
                # Combine all chunks back into a single document
                sorted_points = sorted(points, key=lambda x: (x.payload.get("chunk_index", 0), x.payload.get("chunk_part", 0)))
                full_text = "".join([point.payload.get("text", "") for point in sorted_points])

                # Take metadata from the first chunk (they should all be the same)
                metadata = sorted_points[0].payload.get("metadata", {})

                return {
                    "id": doc_id,
                    "content": full_text,
                    "metadata": metadata
                }

            logger.info(f"Document {doc_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            return None


# Global instance of the vectorstore service
# In a real application, this might be managed by a dependency injection framework
vectorstore_service = None


def get_vectorstore_service():
    """
    Get the global improved vectorstore service instance.
    Initializes the service if it doesn't exist.
    """
    global vectorstore_service
    if vectorstore_service is None:
        try:
            # Check for environment-specific configuration
            import os
            url = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")

            vectorstore_service = VectorStoreService(url=url, api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize vectorstore service: {str(e)}")
            # Create a service instance even if initialization fails
            vectorstore_service = VectorStoreService()  # This will handle the error internally
    return vectorstore_service