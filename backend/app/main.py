from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, HTTPException, status
import re
import html
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio
import os
import secrets
import string
from datetime import timedelta
from dotenv import load_dotenv

def markdown_to_text(markdown_content: str) -> str:
    """
    Convert markdown content to plain text by removing markdown formatting.

    Args:
        markdown_content: Raw markdown content

    Returns:
        Plain text content with markdown formatting removed
    """
    if not markdown_content:
        return ""

    # Remove HTML tags if any
    text = html.unescape(markdown_content)

    # Remove common document metadata lines (like 'id:', 'sidebar_position:', etc.)
    text = re.sub(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*.*$', '', text, flags=re.MULTILINE)

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

    # Remove links [text](url) -> text (including the problematic format)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url) -> text
    text = re.sub(r'\*\*\[([^\]]+)\]\([^)]+\)\*\*', r'\1', text)  # **[text](url)** -> text

    # Remove the specific problematic format in your data
    text = re.sub(r'\*\*\[([^\]]+)\]\([^)]+\)\s*=\s*id:\s*[^\s]+\s+sidebar_position:\s*\d+', '', text)

    # Remove images ![alt](url) -> alt
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)

    # Remove blockquotes
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove reference-style links [text][1] and reference definitions [1]: url
    text = re.sub(r'\[([^\]]+)\]\[[^\]]+\]', r'\1', text)  # [text][1] -> text
    text = re.sub(r'\n\[.+\]:.+\n', '\n', text)  # Remove reference definitions

    # Remove YAML frontmatter if present
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)

    # Replace common markdown symbols
    text = re.sub(r'\\', '', text)  # Remove escape characters

    # Remove extra whitespace and normalize
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple blank lines with single
    text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces to single space
    text = text.strip()

    # Clean up any remaining markdown artifacts
    text = re.sub(r'\n\s*-', '\n- ', text)  # Ensure proper list formatting
    text = re.sub(r'\n\s#\s', '\n', text)   # Remove any remaining header markers

    # Clean up any remaining document artifacts
    text = re.sub(r'\.\s*\.\s*\.', '', text)  # Remove ellipsis artifacts
    text = re.sub(r'\s*=\s*id:[^,\n]*,', '', text)  # Remove id assignments
    text = re.sub(r'\s*sidebar_position:\s*\d+', '', text)  # Remove sidebar positions

    return text

# Load environment variables from .env file
load_dotenv()
from .auth_service import (
    authenticate_user,
    register_user,
    get_current_user_from_token,
    generate_oauth_state,
    validate_oauth_state,
    create_access_token as auth_create_access_token,
    get_user_by_email
)
from .database_models import get_db
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import RedirectResponse

# Define Token model
class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

app = FastAPI(title="RAG Chatbot API - Optimized for Railway Deployment")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Add CORS middleware to allow requests from the frontend
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,  # Allow credentials to be sent with requests
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add TrustedHostMiddleware for Railway deployment
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Railway-specific hosts will be allowed
)

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
            # Check for Railway-specific Qdrant configuration first (no external service detection)
            import os
            qdrant_host = os.getenv("QDRANT_HOST")
            qdrant_port = os.getenv("QDRANT_PORT", "6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            from .vectorstore_service import VectorStoreService

            if qdrant_host:
                # Use external Qdrant for Railway (persistence required)
                qdrant_url = f"https://{qdrant_host}:{qdrant_port}"
                vectorstore_service = VectorStoreService(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
            else:
                # Use default VectorStoreService which handles local configuration
                vectorstore_service = VectorStoreService()

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

# Import database service at the top level
from .database_service import get_database_service

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

        # Convert markdown to plain text before indexing
        plain_text_content = markdown_to_text(request.content)

        # Store the raw document in the database
        doc_id = await db_service.store_document(
            content=plain_text_content,  # Store the cleaned content
            metadata={**(request.metadata or {}), "title": request.title or "Untitled"}
        )

        # Index the document in the vector store
        indexed_doc_id = vectorstore_service.index_document(
            text=plain_text_content,
            doc_id=doc_id,
            metadata={**(request.metadata or {}), "title": request.title or "Untitled"}
        )

        # For now, we're using a simple approach where we just embed the entire content
        # In a real implementation, we would chunk the document appropriately
        chunks = [plain_text_content[i:i+512] for i in range(0, len(plain_text_content), 512)]

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

        # Convert markdown to plain text before indexing (if it's a text/markdown file)
        plain_text_content = markdown_to_text(content_str)

        # Store the raw document in the database
        doc_id = await db_service.store_document(
            content=plain_text_content,
            metadata={"filename": file.filename, "file_type": file.content_type}
        )

        # Index the document in the vector store
        indexed_doc_id = vectorstore_service.index_document(
            text=plain_text_content,
            doc_id=doc_id,
            metadata={"filename": file.filename, "file_type": file.content_type}
        )

        # Simple chunking approach
        chunks = [plain_text_content[i:i+512] for i in range(0, len(plain_text_content), 512)]

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

# Authentication routes
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

@app.post("/register", response_model=Token)
async def register_user_endpoint(request: RegisterRequest):
    """
    Register a new user.
    """
    from .database_models import get_db
    from sqlalchemy.orm import Session

    db: Session = next(get_db())
    try:
        from .auth_service import register_user
        user = register_user(db=db, email=request.email, password=request.password, name=request.name)

        # Create access token
        from datetime import timedelta
        from .auth_service import create_access_token
        access_token_expires = timedelta(minutes=30)  # 30 minutes expiry
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "is_verified": user.is_verified
            }
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    finally:
        db.close()

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/login", response_model=Token)
async def login_user_endpoint(request: LoginRequest):
    """
    Login a user and return an access token.
    """
    from .database_models import get_db
    from sqlalchemy.orm import Session

    db: Session = next(get_db())
    try:
        from .auth_service import authenticate_user
        user = authenticate_user(db=db, email=request.email, password=request.password)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password. Please try again.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create access token
        from datetime import timedelta
        from .auth_service import create_access_token
        access_token_expires = timedelta(minutes=30)  # 30 minutes expiry
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "is_verified": user.is_verified
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )
    finally:
        db.close()

@app.post("/logout")
async def logout_user_endpoint(token: str = Depends(oauth2_scheme)):
    """
    Logout a user (invalidate token in a real implementation).
    """
    # In a real implementation, you would add the token to a blacklist
    return {"message": "Successfully logged out"}

@app.get("/me")
async def get_current_user_endpoint(token: str = Depends(oauth2_scheme)):
    """
    Get the current user's information.
    """
    from .auth_service import get_current_user_from_token
    user = get_current_user_from_token(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"user": {"id": user.id, "email": user.email, "name": user.name}}

# OAuth routes (redirect to external providers)
@app.get("/auth/google")
async def login_with_google():
    """
    Redirect to Google OAuth login.
    """
    from .auth_service import generate_oauth_state
    state = generate_oauth_state()
    # In a real implementation, redirect to Google OAuth
    # This is a placeholder - you'd need to implement the full OAuth flow
    google_client_id = os.getenv('GOOGLE_CLIENT_ID')
    if not google_client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth is not configured. Please set GOOGLE_CLIENT_ID environment variable."
        )
    # Use the correct redirect URI based on environment
    base_url = os.getenv('OAUTH_REDIRECT_URI', 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app')
    redirect_uri = f"{base_url}/auth/google/callback"
    redirect_url = f"https://accounts.google.com/o/oauth2/auth?response_type=code&client_id={google_client_id}&redirect_uri={redirect_uri}&scope=openid email profile&state={state}"
    return RedirectResponse(url=redirect_url)

@app.get("/auth/github")
async def login_with_github():
    """
    Redirect to GitHub OAuth login.
    """
    from .auth_service import generate_oauth_state
    state = generate_oauth_state()
    # In a real implementation, redirect to GitHub OAuth
    # This is a placeholder - you'd need to implement the full OAuth flow
    github_client_id = os.getenv('GITHUB_CLIENT_ID')
    if not github_client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GitHub OAuth is not configured. Please set GITHUB_CLIENT_ID environment variable."
        )
    # Use the correct redirect URI based on environment
    base_url = os.getenv('OAUTH_REDIRECT_URI', 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app')
    redirect_uri = f"{base_url}/auth/github/callback"
    redirect_url = f"https://github.com/login/oauth/authorize?client_id={github_client_id}&redirect_uri={redirect_uri}&scope=user:email&state={state}"
    return RedirectResponse(url=redirect_url)


# OAuth callback routes to handle responses from providers
@app.get("/auth/google/callback")
async def google_callback(code: str = None, state: str = None, db: Session = Depends(get_db)):
    """
    Handle Google OAuth callback.
    """
    from .auth_service import validate_oauth_state, register_user, get_user_by_email, create_access_token as auth_create_access_token
    from urllib.parse import urlencode
    import requests

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code from Google"
        )

    if state and not validate_oauth_state(state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state parameter"
        )

    try:
        # Exchange authorization code for access token
        google_client_id = os.getenv('GOOGLE_CLIENT_ID')
        google_client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        base_url = os.getenv('OAUTH_REDIRECT_URI', 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app')
        redirect_uri = f"{base_url}/auth/google/callback"

        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "code": code,
            "client_id": google_client_id,
            "client_secret": google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        }

        token_response = requests.post(token_url, data=token_data)
        token_json = token_response.json()

        if "access_token" not in token_json:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to obtain access token from Google"
            )

        access_token = token_json["access_token"]

        # Get user info from Google
        user_info_response = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        user_info = user_info_response.json()

        if "email" not in user_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get user info from Google"
            )

        # Check if user exists, create if not
        email = user_info["email"]
        name = user_info.get("name", email.split("@")[0])  # Use email prefix as name if not provided

        user = get_user_by_email(db, email)
        if not user:
            # Generate a secure temporary password for OAuth users (ensure it's under 72 bytes)
            temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
            # Ensure password doesn't exceed 72 bytes for bcrypt compatibility
            if len(temp_password.encode('utf-8')) > 72:
                temp_password = temp_password[:72]
            user = register_user(db, email, temp_password, name)

        # Create JWT token
        token_data = {
            "sub": user.email,
            "user_id": user.id,
            "name": user.name
        }
        access_token = auth_create_access_token(data=token_data)

        # Redirect to frontend with token
        frontend_url = os.getenv('FRONTEND_URL', 'https://physical-ai-and-humanoid-robotics-b-rosy.vercel.app')
        redirect_params = urlencode({"token": access_token, "user": user.name})
        redirect_url = f"{frontend_url}/?auth=success&{redirect_params}"

        return RedirectResponse(url=redirect_url)

    except Exception as e:
        print(f"Google OAuth error: {str(e)}")
        error_msg = str(e)
        # Check if this is a password length issue
        if "password cannot be longer than 72 bytes" in error_msg:
            error_msg = "password cannot be longer than 72 bytes, truncate manually if necessary (e.g. my_password[:72])"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google OAuth failed: {error_msg}"
        )


@app.get("/auth/github/callback")
async def github_callback(code: str = None, state: str = None, db: Session = Depends(get_db)):
    """
    Handle GitHub OAuth callback.
    """
    from .auth_service import validate_oauth_state, register_user, get_user_by_email, create_access_token as auth_create_access_token
    from urllib.parse import urlencode
    import requests

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code from GitHub"
        )

    if state and not validate_oauth_state(state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state parameter"
        )

    try:
        # Exchange authorization code for access token
        github_client_id = os.getenv('GITHUB_CLIENT_ID')
        github_client_secret = os.getenv('GITHUB_CLIENT_SECRET')
        base_url = os.getenv('OAUTH_REDIRECT_URI', 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app')
        redirect_uri = f"{base_url}/auth/github/callback"

        token_url = "https://github.com/login/oauth/access_token"
        token_data = {
            "code": code,
            "client_id": github_client_id,
            "client_secret": github_client_secret,
            "redirect_uri": redirect_uri
        }

        token_response = requests.post(
            token_url,
            data=token_data,
            headers={"Accept": "application/json"}
        )
        token_json = token_response.json()

        if "access_token" not in token_json:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to obtain access token from GitHub"
            )

        access_token = token_json["access_token"]

        # Get user info from GitHub
        user_info_response = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {access_token}"}
        )
        user_info = user_info_response.json()

        # Get user email from GitHub (separate API call needed)
        emails_response = requests.get(
            "https://api.github.com/user/emails",
            headers={"Authorization": f"token {access_token}"}
        )
        emails = emails_response.json()
        email = None
        for email_obj in emails:
            if email_obj.get("primary") and email_obj.get("verified"):
                email = email_obj["email"]
                break
        if not email:
            email = f"github_{user_info['id']}@example.com"  # Fallback email

        # Check if user exists, create if not
        name = user_info.get("name") or user_info.get("login", email.split("@")[0])

        user = get_user_by_email(db, email)
        if not user:
            # Generate a secure temporary password for OAuth users (ensure it's under 72 bytes)
            temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
            # Ensure password doesn't exceed 72 bytes for bcrypt compatibility
            if len(temp_password.encode('utf-8')) > 72:
                temp_password = temp_password[:72]
            user = register_user(db, email, temp_password, name)

        # Create JWT token
        token_data = {
            "sub": user.email,
            "user_id": user.id,
            "name": user.name
        }
        access_token = auth_create_access_token(data=token_data)

        # Redirect to frontend with token
        frontend_url = os.getenv('FRONTEND_URL', 'https://physical-ai-and-humanoid-robotics-b-rosy.vercel.app')
        redirect_params = urlencode({"token": access_token, "user": user.name})
        redirect_url = f"{frontend_url}/?auth=success&{redirect_params}"

        return RedirectResponse(url=redirect_url)

    except Exception as e:
        print(f"GitHub OAuth error: {str(e)}")
        error_msg = str(e)
        # Check if this is a password length issue
        if "password cannot be longer than 72 bytes" in error_msg:
            error_msg = "password cannot be longer than 72 bytes, truncate manually if necessary (e.g. my_password[:72])"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitHub OAuth failed: {error_msg}"
        )


@app.post("/verify-email")
async def verify_email_endpoint(token: str):
    """
    Verify a user's email using the verification token.
    """
    from .database_models import get_db
    from sqlalchemy.orm import Session
    from .auth_service import verify_user_email

    db: Session = next(get_db())
    try:
        user = verify_user_email(db=db, token=token)
        return {
            "message": "Email verified successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "is_verified": user.is_verified
            }
        }
    except Exception as e:
        logger.error(f"Email verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)