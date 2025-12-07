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
        return self._documents.get(doc_id)

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
        if session_id not in self._chat_history:
            self._chat_history[session_id] = []
        self._chat_history[session_id].append({
            "user_message": user_message,
            "bot_response": bot_response,
            "timestamp": datetime.now().isoformat(),
            "context_metadata": context_metadata
        })
        logger.info(f"Stored in-memory chat history for session: {session_id}")

    async def get_chat_history(self, session_id: str) -> List[Dict]:
        return self._chat_history.get(session_id, [])

    async def get_recent_chat_sessions(self, limit: int = 10) -> List[Dict]:
        # For in-memory, just return a dummy session or the existing ones
        # This implementation is simplified for rapid deployment
        sessions = []
        for session_id, history in self._chat_history.items():
            if history:
                sessions.append({"session_id": session_id, "last_activity": history[-1]["timestamp"]})
        sessions.sort(key=lambda x: x["last_activity"], reverse=True)
        return sessions[:limit]

    async def close(self):
        # No resources to close for in-memory storage
        logger.info("In-memory database closed (no-op)")

# Global instance of the database service
database_service = None

async def get_database_service() -> DatabaseService:
    global database_service
    if database_service is None:
        database_service = DatabaseService()
    return database_service