---
title: Embed Document Example
sidebar_label: Embed Document
---

# Embed Document Example

This example demonstrates how to embed a document using the RAG Chatbot backend, including chunking the document and generating vector embeddings.

## Overview

The embedding process involves:
1. Reading the source document
2. Chunking the document into smaller pieces
3. Generating vector embeddings for each chunk
4. Storing the embeddings in the vector store

## Prerequisites

- Backend API running (typically at `http://localhost:8000`)
- Access to the `/embed` endpoint
- Document to embed (text format)

## Step 1: Prepare Your Document

First, prepare the document you want to embed. This can be any text content:

```python
# Example document content
document_content = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.

Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.

Modern machine learning techniques are at the heart of AI. Problems for AI applications include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects.
"""
```

## Step 2: Chunk the Document

The document needs to be split into smaller chunks for processing:

```python
def chunk_document(text: str, chunk_size: int = 512) -> list:
    """
    Split a document into chunks of specified size.

    Args:
        text: The document text to chunk
        chunk_size: Size of each chunk in characters

    Returns:
        List of text chunks
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# Example usage
document_chunks = chunk_document(document_content, chunk_size=512)
print(f"Document split into {len(document_chunks)} chunks")
```

## Step 3: Embed the Document Using the API

Here's how to embed your document using the backend API:

### Python Example

```python
import requests
import json

def embed_document(api_url: str, document_text: str, chunk_size: int = 512):
    """
    Embed a document using the RAG Chatbot backend API.

    Args:
        api_url: Base URL of the backend API
        document_text: Text content of the document to embed
        chunk_size: Size of text chunks to process separately

    Returns:
        Response from the embedding API
    """
    # Prepare the request payload
    payload = {
        "text": document_text,
        "chunk_size": chunk_size
    }

    # Make the API request
    response = requests.post(
        f"{api_url}/embed",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Successfully embedded {result['chunk_count']} chunks")
        print(f"Total embeddings generated: {len(result['embeddings'])}")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Example usage
api_url = "http://localhost:8000"  # Replace with your backend URL
embeddings_result = embed_document(api_url, document_content, chunk_size=512)
```

### JavaScript/TypeScript Example

```javascript
async function embedDocument(apiUrl, documentText, chunkSize = 512) {
    try {
        const response = await fetch(`${apiUrl}/embed`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: documentText,
                chunk_size: chunkSize
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log(`Successfully embedded ${result.chunk_count} chunks`);
        console.log(`Total embeddings generated: ${result.embeddings.length}`);
        return result;
    } catch (error) {
        console.error('Error embedding document:', error);
        throw error;
    }
}

// Example usage
const apiUrl = "http://localhost:8000"; // Replace with your backend URL
const documentContent = `Your document text here...`;
embedDocument(apiUrl, documentContent, 512);
```

## Step 4: Store in Vector Store

After embedding, you'll typically want to store the embeddings in your vector store (Qdrant). Here's how to do it using the vector store service:

```python
from app.vectorstore_service import get_vectorstore_service

def store_embedded_document(document_text: str, metadata: dict = None):
    """
    Store an embedded document in the vector store.

    Args:
        document_text: The text content of the document
        metadata: Optional metadata to store with the document

    Returns:
        Document ID of the stored document
    """
    # Get the vector store service instance
    vectorstore_service = get_vectorstore_service()

    # Index the document (this handles chunking and embedding internally)
    doc_id = vectorstore_service.index_document(
        text=document_text,
        metadata=metadata
    )

    print(f"Document stored with ID: {doc_id}")
    return doc_id

# Example usage
metadata = {
    "title": "Introduction to AI",
    "author": "John Doe",
    "source": "example_document.txt",
    "tags": ["AI", "Machine Learning", "NLP"]
}

stored_doc_id = store_embedded_document(document_content, metadata)
```

## Step 5: Complete Example Script

Here's a complete script that combines all the steps:

```python
import requests
import json
from app.vectorstore_service import get_vectorstore_service

def complete_document_embedding_process(api_url: str, document_text: str, metadata: dict = None):
    """
    Complete process: embed and store a document.

    Args:
        api_url: Base URL of the backend API
        document_text: Text content of the document to embed
        metadata: Optional metadata to store with the document

    Returns:
        Document ID of the stored document
    """
    print("Starting document embedding process...")

    # Step 1: Embed the document via API
    print("Step 1: Embedding document via API...")
    payload = {
        "text": document_text,
        "chunk_size": 512
    }

    response = requests.post(
        f"{api_url}/embed",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        print(f"Embedding failed: {response.status_code} - {response.text}")
        return None

    embedding_result = response.json()
    print(f"Successfully embedded {embedding_result['chunk_count']} chunks")

    # Step 2: Store in vector store (if you want to do it via the service directly)
    # Alternatively, you can let the vector store service handle embedding internally
    print("Step 2: Storing in vector store...")
    vectorstore_service = get_vectorstore_service()

    doc_id = vectorstore_service.index_document(
        text=document_text,
        metadata=metadata
    )

    print(f"Document stored successfully with ID: {doc_id}")
    return doc_id

# Example usage
if __name__ == "__main__":
    api_url = "http://localhost:8000"  # Replace with your backend URL

    sample_document = """
    Machine learning (ML) is a field of inquiry devoted to understanding and building
    methods that 'learn', that is, methods that leverage data to improve performance
    on some set of tasks. It is seen as a part of artificial intelligence. Machine
    learning algorithms build a model based on sample data, known as training data,
    in order to make predictions or decisions without being explicitly programmed
    to do so.

    Machine learning algorithms are used in a wide variety of applications, such as
    in medicine, email filtering, speech recognition, and computer vision, where it
    is difficult or unfeasible to develop conventional algorithms to perform the
    needed tasks.
    """

    metadata = {
        "title": "Machine Learning Overview",
        "author": "Jane Smith",
        "source": "ml_introduction.txt",
        "tags": ["ML", "AI", "Algorithms"]
    }

    doc_id = complete_document_embedding_process(api_url, sample_document, metadata)
    if doc_id:
        print(f"Process completed successfully! Document ID: {doc_id}")
    else:
        print("Process failed!")
```

## API Response Format

The `/embed` endpoint returns a JSON response in this format:

```json
{
  "embeddings": [
    [0.1, 0.2, 0.3, ...],  // First chunk embedding (1536 dimensions)
    [0.4, 0.5, 0.6, ...],  // Second chunk embedding
    ...
  ],
  "chunk_count": 3
}
```

Each embedding is a vector (list of floating-point numbers) representing the semantic meaning of the text chunk. The length of each vector depends on the embedding model used (typically 384, 768, or 1536 dimensions).

## Error Handling

Always implement proper error handling when working with the embedding API:

```python
def embed_document_with_error_handling(api_url: str, document_text: str):
    """
    Embed a document with comprehensive error handling.
    """
    try:
        payload = {
            "text": document_text,
            "chunk_size": 512
        }

        response = requests.post(
            f"{api_url}/embed",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30  # 30 second timeout
        )

        # Check if the request was successful
        response.raise_for_status()

        result = response.json()

        # Validate the response structure
        if 'embeddings' not in result or 'chunk_count' not in result:
            raise ValueError("Invalid response format from embedding API")

        print(f"Successfully embedded {result['chunk_count']} chunks")
        return result

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        print(f"Response content: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Connection error: Could not reach the embedding API")
    except requests.exceptions.Timeout:
        print("Request timed out: The embedding API took too long to respond")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None
```

This example demonstrates the complete process of embedding a document using the RAG Chatbot system, from preparing the document to storing the embeddings in the vector store.