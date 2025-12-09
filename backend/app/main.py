from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio

app = FastAPI(title="RAG Chatbot API - Optimized for Railway Deployment")

@app.get("/")
async def root():
    return {"status": "running", "service": "rag-chatbot-api"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances (not initialized yet)
embedding_service = None
llm_service = None
vectorstore_service = None
openai_assistant_service = None

def get_embedding_service():
    global embedding_service
    if embedding_service is None:
        try:
            from .embedding_service import get_embedding_service as _get_embedding_service
            embedding_service = _get_embedding_service()
        except ValueError as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            # Return a mock service that provides basic functionality
            class MockEmbeddingService:
                def embed_text(self, text, chunk_size=512):
                    logger.warning("Mock embedding service: returning dummy embeddings")
                    # Return a simple dummy embedding
                    return [[0.1] * 768]  # 768-dim vector like Gemini embeddings

                def embed_query(self, query):
                    logger.warning("Mock embedding service: returning dummy query embedding")
                    return [0.1] * 768  # 768-dim vector like Gemini embeddings

            embedding_service = MockEmbeddingService()
    return embedding_service

def get_llm_service():
    global llm_service
    if llm_service is None:
        try:
            from .llm_service import get_llm_service as _get_llm_service
            llm_service = _get_llm_service()
        except ValueError as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            # Return a mock service that provides basic functionality
            class MockLLMService:
                def generate_response(self, query, context=None, history=None):
                    logger.warning("Mock LLM service: returning mock response")
                    return f"I'm sorry, but I couldn't process your query '{query}' because the LLM service is not properly configured. Please set the GEMINI_API_KEY environment variable."

            llm_service = MockLLMService()
    return llm_service

def get_vectorstore_service():
    global vectorstore_service
    if vectorstore_service is None:
        try:
            from .vectorstore_service import get_vectorstore_service as _get_vectorstore_service
            vectorstore_service = _get_vectorstore_service()
        except Exception as e:
            logger.error(f"Failed to initialize vectorstore service: {str(e)}")
            # Return a mock service that provides basic functionality
            class MockVectorStoreService:
                def index_document(self, text, doc_id=None, metadata=None):
                    logger.warning("Mock vectorstore service: would index document")
                    return doc_id or "mock-doc-id"

                def search(self, query_vector, top_k=5):
                    logger.warning("Mock vectorstore service: returning empty search results")
                    return []

                def delete_document(self, doc_id):
                    logger.warning(f"Mock vectorstore service: would delete document {doc_id}")

                def get_document(self, doc_id):
                    logger.warning(f"Mock vectorstore service: would return document {doc_id}")
                    return None

            vectorstore_service = MockVectorStoreService()
    return vectorstore_service

def get_openai_assistant_service():
    global openai_assistant_service
    if openai_assistant_service is None:
        try:
            from .openai_assistant_service import get_openai_assistant_service as _get_openai_assistant_service
            openai_assistant_service = _get_openai_assistant_service()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI assistant service: {str(e)}")
            # Return the mock service that's already implemented
            from .openai_assistant_service import MockGeminiAssistantService
            openai_assistant_service = MockGeminiAssistantService()
    return openai_assistant_service

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

class DocumentIngestRequest(BaseModel):
    content: str
    title: Optional[str] = None
    metadata: Optional[dict] = {}

class DocumentIngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    message: str

class DocumentQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_full_book: Optional[bool] = True

class OpenAIAssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    use_rag_context: Optional[bool] = True
    query_for_rag: Optional[str] = ""

class OpenAIAssistantResponse(BaseModel):
    response: str
    thread_id: str
    assistant_id: Optional[str] = None

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """
    Embed text chunks using Sentence Transformers.
    This endpoint takes text input and returns vector embeddings for each chunk.
    """
    try:
        logger.info(f"Embedding text of length {len(request.text)} with chunk size {request.chunk_size}")

        # Use the embedding service to generate embeddings
        embedding_service = get_embedding_service()
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
        embedding_service = get_embedding_service()
        query_vector = embedding_service.embed_query(request.query)

        # Search the vector store for similar vectors
        vectorstore_service = get_vectorstore_service()
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

        # Get services (they will be initialized on first access)
        embedding_service = get_embedding_service()
        llm_service = get_llm_service()
        vectorstore_service = get_vectorstore_service()

        # Query for relevant context based on the message
        query_vector = embedding_service.embed_query(request.message)
        query_results = vectorstore_service.search(query_vector, top_k=3)

        # Generate response using the LLM service
        response_text = llm_service.generate_response(
            query=request.message,
            context=query_results,
            history=request.history
        )

        # Store the conversation in the database (if needed) - make this optional
        try:
            db_service = await get_database_service()
            import uuid
            session_id = str(uuid.uuid4())
            await db_service.store_chat_history(session_id, request.message, response_text)
        except Exception as db_error:
            logger.warning(f"Database operation failed: {str(db_error)} - continuing without history storage")

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

        # Get services
        embedding_service = get_embedding_service()
        llm_service = get_llm_service()
        vectorstore_service = get_vectorstore_service()

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

@app.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document(request: DocumentIngestRequest):
    """
    Ingest a document into the RAG system.
    This endpoint takes document content, chunks it, embeds it, and stores it in the vector store.
    """
    try:
        logger.info(f"Ingesting document with title: {request.title or 'Untitled'}")

        # Get services
        vectorstore_service = get_vectorstore_service()

        # Get the database service
        db_service = await get_database_service()

        # Store the raw document in the database
        doc_id = await db_service.store_document(
            content=request.content,
            metadata={**(request.metadata or {}), "title": request.title or "Untitled"}
        )

        # Index the document in the vector store
        indexed_doc_id = vectorstore_service.index_document(
            text=request.content,
            doc_id=doc_id,
            metadata={**(request.metadata or {}), "title": request.title or "Untitled"}
        )

        # For now, we're using a simple approach where we just embed the entire content
        # In a real implementation, we would chunk the document appropriately
        chunks = [request.content[i:i+512] for i in range(0, len(request.content), 512)]

        logger.info(f"Successfully ingested document {indexed_doc_id} with {len(chunks)} chunks")

        return DocumentIngestResponse(
            document_id=indexed_doc_id,
            chunks_indexed=len(chunks),
            message=f"Successfully ingested document with {len(chunks)} chunks"
        )
    except Exception as e:
        logger.error(f"Error in ingest_document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@app.post("/ingest_file", response_model=DocumentIngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a document file into the RAG system.
    This endpoint accepts a file upload, reads its content, and processes it.
    """
    try:
        logger.info(f"Ingesting file: {file.filename}")

        # Read the file content
        content = await file.read()
        content_str = content.decode('utf-8')

        # Get services
        vectorstore_service = get_vectorstore_service()

        # Get the database service
        db_service = await get_database_service()

        # Store the raw document in the database
        doc_id = await db_service.store_document(
            content=content_str,
            metadata={"filename": file.filename, "file_type": file.content_type}
        )

        # Index the document in the vector store
        indexed_doc_id = vectorstore_service.index_document(
            text=content_str,
            doc_id=doc_id,
            metadata={"filename": file.filename, "file_type": file.content_type}
        )

        # Simple chunking approach
        chunks = [content_str[i:i+512] for i in range(0, len(content_str), 512)]

        logger.info(f"Successfully ingested file {file.filename} with {len(chunks)} chunks")

        return DocumentIngestResponse(
            document_id=indexed_doc_id,
            chunks_indexed=len(chunks),
            message=f"Successfully ingested file {file.filename} with {len(chunks)} chunks"
        )
    except Exception as e:
        logger.error(f"Error in ingest_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")

@app.post("/book_query", response_model=ChatResponse)
async def query_book(request: DocumentQueryRequest):
    """
    Query the entire book or document collection.
    This endpoint searches through all ingested documents to answer questions.
    """
    try:
        logger.info(f"Querying book with: {request.query[:50]}...")

        # Get services
        embedding_service = get_embedding_service()
        llm_service = get_llm_service()
        vectorstore_service = get_vectorstore_service()

        # Embed the query using the embedding service
        query_vector = embedding_service.embed_query(request.query)

        # Search the vector store for similar vectors across all documents
        results = vectorstore_service.search(query_vector, request.top_k)

        # Generate response using the LLM service with the retrieved context
        response_text = llm_service.generate_response(
            query=request.query,
            context=results,
            history=[]
        )

        logger.info("Successfully generated book query response")
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error in query_book: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Book query failed: {str(e)}")

@app.post("/openai_assistant", response_model=OpenAIAssistantResponse)
async def openai_assistant_chat(request: OpenAIAssistantRequest):
    """
    Chat with the OpenAI assistant, optionally using RAG context.
    This endpoint provides OpenAI Agent-like functionality integrated with our RAG system.
    """
    try:
        logger.info(f"OpenAI Assistant request: {request.message[:50]}...")

        # Use the OpenAI assistant service
        assistant_service = get_openai_assistant_service()

        # Get other services
        embedding_service = get_embedding_service()
        vectorstore_service = get_vectorstore_service()

        # If we should use RAG context, retrieve it first
        rag_context = ""
        if request.use_rag_context:
            query_for_rag = request.query_for_rag or request.message
            # Embed the query using the embedding service
            query_vector = embedding_service.embed_query(query_for_rag)
            # Search the vector store for similar vectors
            results = vectorstore_service.search(query_vector, top_k=3)
            # Format the context from the results
            rag_context = "\n\nRelevant context from documents:\n"
            for result in results:
                rag_context += f"- {result['content'][:200]}...\n"

        # Create or use existing thread
        thread_id = request.thread_id or assistant_service.create_thread()
        if not thread_id:
            raise HTTPException(status_code=500, detail="Failed to create or access thread")

        success = assistant_service.add_message_to_thread(thread_id, request.message, "user")
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add message to thread")

        # Create an assistant if one doesn't exist or if service is available
        assistant_id = None
        if assistant_service.is_available():
            assistant_id = assistant_service.create_rag_assistant()
            if not assistant_id:
                raise HTTPException(status_code=500, detail="Failed to create assistant")
        else:
            # If OpenAI is not available, return a mock response
            response_text = f"OpenAI Assistant is not available (API key not set or package not installed). Original message: {request.message}"
            if rag_context:
                response_text += f"\n\nRAG Context was available but OpenAI is not accessible."

            return OpenAIAssistantResponse(
                response=response_text,
                thread_id=thread_id,
                assistant_id=assistant_id
            )

        # Run the assistant to get a response
        response_text = assistant_service.run_assistant(thread_id, assistant_id)
        if not response_text:
            raise HTTPException(status_code=500, detail="Failed to get response from assistant")

        logger.info("Successfully got response from OpenAI Assistant")
        return OpenAIAssistantResponse(
            response=response_text,
            thread_id=thread_id,
            assistant_id=assistant_id
        )
    except Exception as e:
        logger.error(f"Error in openai_assistant_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI Assistant chat failed: {str(e)}")

@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Retrieve chat history for a specific session.
    """
    try:
        logger.info(f"Retrieving chat history for session: {session_id}")

        db_service = await get_database_service()
        history = await db_service.get_chat_history(session_id)

        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error in get_chat_history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@app.get("/recent_sessions")
async def get_recent_sessions(limit: int = 10):
    """
    Retrieve recent chat sessions.
    """
    try:
        logger.info(f"Retrieving {limit} recent chat sessions")

        db_service = await get_database_service()
        sessions = await db_service.get_recent_chat_sessions(limit)

        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error in get_recent_sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recent sessions: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy", "service": "rag-chatbot-api"}

# Import here to avoid circular imports - make it more robust
async def get_database_service():
    try:
        from .database_service import get_database_service as _get_database_service
        return await _get_database_service()
    except ImportError as e:
        logger.error(f"Failed to import database service: {str(e)}")
        # Return a mock service that provides basic functionality without database
        class MockDatabaseService:
            async def store_chat_history(self, session_id, user_message, bot_response, context_metadata=None):
                logger.warning(f"Mock DB: Would store chat history for session {session_id}")
                pass

            async def get_chat_history(self, session_id):
                logger.warning(f"Mock DB: Would return chat history for session {session_id}")
                return []

            async def get_recent_chat_sessions(self, limit=10):
                logger.warning("Mock DB: Would return recent chat sessions")
                return []

        return MockDatabaseService()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=8000)