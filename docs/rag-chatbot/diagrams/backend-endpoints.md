---
title: Backend API Endpoints Diagram
sidebar_label: Backend Endpoints
---

# Backend API Endpoints Diagram

```mermaid
graph TD
    subgraph "FastAPI Backend"
        A["/embed - Text Embedding"]
        B["/query - Document Retrieval"]
        C["/chat - Chat Generation"]
        D["/selected_text - Selected Text Processing"]
        E["/health - Health Check"]
    end

    subgraph "Services"
        F["Embedding Service"]
        G["Vector Store Service"]
        H["LLM Service"]
        I["Database Service"]
    end

    subgraph "External"
        J["Context7"]
        K["Qdrant Vector Store"]
        L["Neon Postgres DB"]
        M["Open Source LLM"]
    end

    A --> F
    B --> F
    B --> G
    C --> H
    D --> F
    D --> G
    D --> H

    F --> J
    G --> K
    H --> M
    I --> L

    A -.-> J
    B -.-> K
    C -.-> M
    D -.-> J
    D -.-> K
    D -.-> M
```

## API Endpoints Description

### `/embed`
- **Method**: POST
- **Purpose**: Generate vector embeddings for text chunks using Context7
- **Input**: Text content and chunk size
- **Output**: Embedding vectors

### `/query`
- **Method**: POST
- **Purpose**: Search for relevant documents in the vector store
- **Input**: Query text and number of results
- **Output**: List of relevant documents with scores

### `/chat`
- **Method**: POST
- **Purpose**: Generate responses using the LLM with context
- **Input**: User message and conversation history
- **Output**: Generated response

### `/selected_text`
- **Method**: POST
- **Purpose**: Process user-selected text with RAG
- **Input**: Selected text and optional context
- **Output**: Response based on selected text

### `/health`
- **Method**: GET
- **Purpose**: Check API health status
- **Output**: Health status information