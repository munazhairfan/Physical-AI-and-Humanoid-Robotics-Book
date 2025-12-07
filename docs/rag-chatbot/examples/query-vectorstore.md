---
title: Query Vector Store Example
sidebar_label: Query Vector Store
---

# Query Vector Store Example

This example demonstrates how to query the vector store using the RAG Chatbot backend to find similar documents based on a query.

## Overview

The vector store query process involves:
1. Converting a query string to an embedding vector
2. Performing similarity search in the vector store
3. Retrieving the most relevant documents

## Prerequisites

- Backend API running (typically at `http://localhost:8000`)
- Documents already embedded and stored in the vector store
- Access to the `/query` endpoint

## Step 1: Prepare Your Query

Start with a query string that represents what you're looking for:

```python
# Example query
query_text = "What is artificial intelligence and how does it work?"
```

## Step 2: Query the Vector Store Using the API

Here's how to query the vector store using the backend API:

### Python Example

```python
import requests
import json

def query_vector_store(api_url: str, query: str, top_k: int = 5):
    """
    Query the vector store using the RAG Chatbot backend API.

    Args:
        api_url: Base URL of the backend API
        query: The search query text
        top_k: Number of top results to return

    Returns:
        Response from the query API
    """
    # Prepare the request payload
    payload = {
        "query": query,
        "top_k": top_k
    }

    # Make the API request
    response = requests.post(
        f"{api_url}/query",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Found {len(result['results'])} similar documents")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Example usage
api_url = "http://localhost:8000"  # Replace with your backend URL
query_results = query_vector_store(api_url, query_text, top_k=3)

if query_results:
    for i, result in enumerate(query_results['results']):
        print(f"Result {i+1}:")
        print(f"  Content: {result['content'][:100]}...")
        print(f"  Score: {result['score']}")
        print(f"  Document ID: {result['original_doc_id']}")
        print()
```

### JavaScript/TypeScript Example

```javascript
async function queryVectorStore(apiUrl, query, topK = 5) {
    try {
        const response = await fetch(`${apiUrl}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: topK
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log(`Found ${result.results.length} similar documents`);
        return result;
    } catch (error) {
        console.error('Error querying vector store:', error);
        throw error;
    }
}

// Example usage
const apiUrl = "http://localhost:8000"; // Replace with your backend URL
const queryText = "What is artificial intelligence and how does it work?";
queryVectorStore(apiUrl, queryText, 3)
    .then(results => {
        results.results.forEach((result, index) => {
            console.log(`Result ${index + 1}:`);
            console.log(`  Content: ${result.content.substring(0, 100)}...`);
            console.log(`  Score: ${result.score}`);
            console.log(`  Document ID: ${result.original_doc_id}`);
            console.log();
        });
    });
```

## Step 3: Direct Vector Store Service Query

You can also query the vector store directly using the service:

```python
from app.embedding_service import get_embedding_service
from app.vectorstore_service import get_vectorstore_service

def direct_vector_store_query(query_text: str, top_k: int = 5):
    """
    Query the vector store directly using the service.

    Args:
        query_text: The search query text
        top_k: Number of top results to return

    Returns:
        List of similar documents
    """
    # Get service instances
    embedding_service = get_embedding_service()
    vectorstore_service = get_vectorstore_service()

    # Embed the query
    query_vector = embedding_service.embed_query(query_text)

    # Search the vector store
    results = vectorstore_service.search(query_vector, top_k=top_k)

    print(f"Found {len(results)} similar documents")
    return results

# Example usage
direct_results = direct_vector_store_query(query_text, top_k=3)
for i, result in enumerate(direct_results):
    print(f"Direct Result {i+1}:")
    print(f"  Content: {result['content'][:100]}...")
    print(f"  Score: {result['score']}")
    print(f"  Document ID: {result['original_doc_id']}")
    print()
```

## Step 4: Complete Query Example Script

Here's a complete script that demonstrates various querying approaches:

```python
import requests
import json
from app.embedding_service import get_embedding_service
from app.vectorstore_service import get_vectorstore_service

class VectorStoreQueryExample:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url

    def api_query(self, query: str, top_k: int = 5):
        """Query using the backend API."""
        payload = {
            "query": query,
            "top_k": top_k
        }

        response = requests.post(
            f"{self.api_url}/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"API query failed: {response.status_code} - {response.text}")
            return None

    def direct_service_query(self, query: str, top_k: int = 5):
        """Query using the vector store service directly."""
        embedding_service = get_embedding_service()
        vectorstore_service = get_vectorstore_service()

        query_vector = embedding_service.embed_query(query)
        results = vectorstore_service.search(query_vector, top_k=top_k)

        return {"results": results}

    def compare_queries(self, query: str, top_k: int = 3):
        """Compare results from API and direct service queries."""
        print(f"Query: {query}")
        print("=" * 50)

        # API query
        print("API Query Results:")
        api_results = self.api_query(query, top_k)
        if api_results:
            for i, result in enumerate(api_results['results']):
                print(f"  {i+1}. Score: {result['score']:.3f}")
                print(f"     Content: {result['content'][:80]}...")
                print()

        # Direct service query
        print("Direct Service Query Results:")
        service_results = self.direct_service_query(query, top_k)
        for i, result in enumerate(service_results['results']):
            print(f"  {i+1}. Score: {result['score']:.3f}")
            print(f"     Content: {result['content'][:80]}...")
            print()

# Example usage
if __name__ == "__main__":
    example = VectorStoreQueryExample("http://localhost:8000")  # Replace with your backend URL

    # Example queries
    queries = [
        "What is machine learning?",
        "How does natural language processing work?",
        "Explain neural networks in AI"
    ]

    for query in queries:
        example.compare_queries(query)
        print("\n" + "="*70 + "\n")
```

## Step 5: Advanced Query Techniques

### Query with Filters

You can also perform more advanced queries with filters:

```python
def filtered_query(query_text: str, filters: dict = None, top_k: int = 5):
    """
    Perform a filtered query on the vector store.

    Args:
        query_text: The search query text
        filters: Dictionary of filters to apply
        top_k: Number of top results to return

    Returns:
        List of filtered similar documents
    """
    embedding_service = get_embedding_service()
    vectorstore_service = get_vectorstore_service()

    # Embed the query
    query_vector = embedding_service.embed_query(query_text)

    # In a real implementation, you would apply filters during the search
    # This is a simplified example that filters results after retrieval
    all_results = vectorstore_service.search(query_vector, top_k=10)  # Get more results to filter

    if filters:
        filtered_results = []
        for result in all_results:
            include_result = True
            for key, value in filters.items():
                if key in result.get('metadata', {}):
                    if result['metadata'][key] != value:
                        include_result = False
                        break
                else:
                    include_result = False
                    break

            if include_result:
                filtered_results.append(result)

        return filtered_results[:top_k]
    else:
        return all_results[:top_k]

# Example with filters
filters = {"tags": ["AI", "Machine Learning"]}  # Example filter
filtered_results = filtered_query("AI concepts", filters, top_k=3)
```

## API Response Format

The `/query` endpoint returns a JSON response in this format:

```json
{
  "results": [
    {
      "id": "chunk-123",
      "content": "The content of the most similar document chunk...",
      "score": 0.85,
      "metadata": {
        "title": "AI Introduction",
        "author": "John Doe"
      },
      "original_doc_id": "doc-456",
      "chunk_index": 0
    },
    {
      "id": "chunk-789",
      "content": "The content of the second most similar document chunk...",
      "score": 0.72,
      "metadata": {
        "title": "Machine Learning Basics",
        "author": "Jane Smith"
      },
      "original_doc_id": "doc-101",
      "chunk_index": 1
    }
  ]
}
```

The `score` indicates similarity (higher scores are more similar), and `content` contains the actual text of the matching document chunk.

## Error Handling

Always implement proper error handling when querying the vector store:

```python
def query_with_error_handling(api_url: str, query: str, top_k: int = 5):
    """
    Query the vector store with comprehensive error handling.
    """
    try:
        payload = {
            "query": query,
            "top_k": top_k
        }

        response = requests.post(
            f"{api_url}/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30  # 30 second timeout
        )

        # Check if the request was successful
        response.raise_for_status()

        result = response.json()

        # Validate the response structure
        if 'results' not in result:
            raise ValueError("Invalid response format from query API")

        print(f"Successfully retrieved {len(result['results'])} results")
        return result

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        print(f"Response content: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Connection error: Could not reach the query API")
    except requests.exceptions.Timeout:
        print("Request timed out: The query API took too long to respond")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None
```

This example demonstrates how to query the vector store effectively using the RAG Chatbot system, including both API-based and direct service-based approaches, with proper error handling and result processing.