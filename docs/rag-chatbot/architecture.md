---
title: RAG Chatbot Architecture
sidebar_label: Architecture
---

# RAG Chatbot Architecture

## System Overview

The RAG Chatbot system is designed with a modular architecture that separates concerns into distinct layers. The system consists of a frontend user interface, a backend API, a vector store for document retrieval, and a database for persistent storage.

## Architecture Components

### 1. Frontend Layer (Docusaurus + React)

The frontend layer is built using Docusaurus, which provides a static site generation framework. It includes:

- **Chat Interface**: A React component that allows users to interact with the chatbot
- **Text Selection Capture**: Functionality to capture user-selected text and send it to the backend for processing
- **API Communication Layer**: Handles communication with the backend API endpoints

### 2. Backend Layer (FastAPI)

The backend layer is built using FastAPI, a modern, fast web framework for building APIs with Python. It includes:

- **API Endpoints**:
  - `/embed`: For processing and embedding text chunks
  - `/query`: For performing similarity searches in the vector store
  - `/chat`: For generating responses using the LLM
  - `/selected_text`: For processing user-selected text with RAG
- **Embedding Service**: Uses Sentence Transformers (all-MiniLM-L6-v2) for generating text embeddings as an alternative to Context7
- **LLM Service**: Interfaces with a free, open-source LLM for response generation
- **Vector Store Service**: Communicates with Qdrant for vector storage and retrieval
- **Database Service**: Interacts with Neon Postgres for storing documents and chat history

### 3. Vector Store (Qdrant)

The vector store layer uses Qdrant, a vector similarity search engine:

- **Collections**: Organizes embedded text chunks
- **Search Engine**: Performs similarity searches using vector embeddings
- **Indexing**: Efficiently indexes and retrieves vector representations of text

### 4. Database (Neon Postgres)

The database layer uses Neon serverless Postgres:

- **Document Storage**: Stores raw documents and metadata
- **Chat History**: Maintains conversation history for context
- **User Data**: Stores user-specific information if needed

## Data Flow

### Standard Query Flow

1. User submits a query through the frontend
2. Frontend sends the query to the backend's `/query` endpoint
3. Backend embeds the query using the embedding service
4. Backend searches for similar vectors in Qdrant
5. Backend retrieves relevant documents from the vector store
6. Backend sends the query and retrieved documents to the LLM service
7. LLM generates a response based on the context
8. Backend returns the response to the frontend
9. Frontend displays the response to the user

### Selected Text Flow

1. User selects text in the frontend
2. Frontend captures the selected text and sends it to the backend's `/selected_text` endpoint
3. Backend processes the selected text using RAG techniques
4. Backend may embed the selected text and search for related content
5. Backend generates a response focused on the selected text
6. Backend returns the response to the frontend
7. Frontend displays the response to the user

## MCP Server Support

The architecture is designed to leverage MCP (Managed Compute Platform) servers for:

- Scalable API endpoint hosting
- Distributed vector search capabilities
- Load balancing for high availability
- Auto-scaling based on demand