"""
Database Service for RAG Chatbot
Handles operations with the Neon Postgres database for document and chat history storage.
"""
from typing import List, Dict, Optional
import logging
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Service class for handling database operations with Neon Postgres.
    Manages document storage, retrieval, and chat history management.
    """

    def __init__(self):
        self._documents = {}  # type: Dict[str, Dict]
        self._chat_history = {} # type: Dict[str, List[Dict]]
        logger.info("Initializing In-Memory Database Service (DEVELOPMENT ONLY)")


    async def store_document(self, content: str, metadata: Dict) -> str:
        document_id = str(uuid.uuid4())
        self._documents[document_id] = {
            "document_id": document_id,
            "content": content,
            "metadata": metadata
        }
        logger.info(f"Stored in-memory document with ID: {document_id}")
        return document_id


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

    async def store_chat_history(self, session_id: str, user_message: str, bot_response: str, context_metadata: Optional[Dict] = None):
        """
        Store a chat interaction in the database.

        Args:
            session_id: The session ID for the conversation
            user_message: The user's message
            bot_response: The bot's response
            context_metadata: Optional metadata about the context used
        """
        logger.info(f"Storing chat history for session: {session_id}")

        await self.pool.execute(
            """
            INSERT INTO chat_history (session_id, user_message, bot_response, context_metadata)
            VALUES ($1, $2, $3, $4)
            """,
            session_id,
            user_message,
            bot_response,
            json.dumps(context_metadata) if context_metadata else None
        )

        logger.info(f"Successfully stored chat history for session: {session_id}")

    async def get_chat_history(self, session_id: str) -> List[Dict]:
        """
        Retrieve chat history for a specific session.

        Args:
            session_id: The session ID to retrieve history for

        Returns:
            List of chat interactions in the session
        """
        logger.info(f"Retrieving chat history for session: {session_id}")

        rows = await self.pool.fetch(
            """
            SELECT user_message, bot_response, timestamp
            FROM chat_history
            WHERE session_id = $1
            ORDER BY timestamp ASC
            """,
            session_id
        )

        history = []
        for row in rows:
            history.append({
                "user_message": row['user_message'],
                "bot_response": row['bot_response'],
                "timestamp": row['timestamp']
            })

        logger.info(f"Retrieved {len(history)} chat interactions for session: {session_id}")
        return history

    async def get_recent_chat_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent chat sessions.

        Args:
            limit: Number of recent sessions to retrieve

        Returns:
            List of recent session IDs and their metadata
        """
        logger.info(f"Retrieving {limit} most recent chat sessions")

        rows = await self.pool.fetch(
            """
            SELECT DISTINCT session_id, timestamp
            FROM chat_history
            ORDER BY timestamp DESC
            LIMIT $1
            """,
            limit
        )

        sessions = []
        for row in rows:
            sessions.append({
                "session_id": str(row['session_id']),
                "timestamp": row['timestamp']
            })

        logger.info(f"Retrieved {len(sessions)} recent chat sessions")
        return sessions

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
        # In a real implementation, this would come from environment/config
        db_url = None  # Use default from environment
        database_service = DatabaseService(db_url)
        await database_service.initialize()
    return database_service