---
title: Setting up Qdrant Vector Store
sidebar_label: Qdrant Setup
---

# Setting up Qdrant Free Tier

## Overview

Qdrant is a vector similarity search engine that provides a production-ready solution for storing and searching vector embeddings. It offers a free tier that is suitable for development and small-scale production applications.

## Prerequisites

- An internet connection
- A Qdrant Cloud account (or access to a Qdrant instance)

## Creating a Qdrant Cloud Account

1. **Visit Qdrant Cloud**: Go to [cloud.qdrant.io](https://cloud.qdrant.io) to access the Qdrant Cloud platform.

2. **Sign Up**: Click on the "Sign Up" button and create an account using your email address or social login.

3. **Verify Email**: Check your email for a verification message from Qdrant and click the verification link.

## Setting Up a Free Collection

1. **Create a New Cluster**:
   - After logging in, click on "Create Cluster" or "New Project"
   - Select the "Free" tier option
   - Choose a region closest to your users for optimal performance
   - Give your cluster a descriptive name (e.g., `rag-chatbot-cluster`)

2. **Configure Cluster Settings**:
   - For the free tier, you'll have limited resources (typically up to 100MB storage)
   - Accept the default settings for the free tier
   - Click "Create" or "Launch"

3. **Wait for Provisioning**:
   - The cluster creation may take a few minutes
   - Wait until the status shows as "Active" or "Ready"

## Getting Connection Details

1. **Access Cluster Dashboard**:
   - Once your cluster is ready, click on it to access the dashboard
   - Note down the cluster endpoint URL (e.g., `https://your-cluster-name.us-east.qdrant.io`)

2. **Create API Key**:
   - Navigate to the "API Keys" or "Security" section
   - Click "Create API Key" or "Generate Key"
   - Give the key a name (e.g., `rag-chatbot-key`)
   - Select appropriate permissions (typically read/write for our use case)
   - Copy and securely store the generated API key

## Configuring Your Application

### Environment Variables

Set up the following environment variables in your application:

```bash
QDRANT_URL=https://your-cluster-name.us-east.qdrant.io
QDRANT_API_KEY=your-api-key-here
```

### Collection Setup

1. **Create a Collection**:
   - You can create collections programmatically in your application
   - Or use the Qdrant Cloud dashboard to create a collection manually
   - Recommended collection name: `rag_documents`
   - For text embeddings, use a vector size of 1536 (for OpenAI embeddings) or 384/768 (for sentence transformer models)
   - Choose an appropriate distance metric (typically `Cosine` for text embeddings)

2. **Collection Parameters Example**:
   ```json
   {
     "name": "rag_documents",
     "vector_size": 1536,
     "distance": "Cosine"
   }
   ```

## Testing the Connection

You can test your Qdrant connection using the following Python code:

```python
from qdrant_client import QdrantClient

# Initialize the client
client = QdrantClient(
    url="https://your-cluster-name.us-east.qdrant.io",
    api_key="your-api-key-here",
    prefer_grpc=False  # Free tier typically uses REST
)

# Test the connection
try:
    collections = client.get_collections()
    print("Connection successful!")
    print(f"Available collections: {collections}")
except Exception as e:
    print(f"Connection failed: {e}")
```

## Free Tier Limitations

Be aware of the free tier limitations:

- **Storage**: Limited to 100MB (varies by provider)
- **Operations**: Limited requests per minute/hour
- **Collections**: Limited number of collections
- **Performance**: Shared resources with other users

## Best Practices for Free Tier

1. **Monitor Usage**: Regularly check your usage in the Qdrant Cloud dashboard
2. **Optimize Embeddings**: Use efficient embedding models to reduce vector size
3. **Batch Operations**: Perform batch operations when possible to reduce API calls
4. **Clean Up**: Regularly remove unnecessary data to stay within limits

## Troubleshooting

- **Connection Issues**: Verify your URL and API key are correct
- **Rate Limits**: Implement exponential backoff in your application
- **Authentication Errors**: Ensure your API key has the correct permissions

## Next Steps

Once you have Qdrant set up, you can proceed to integrate it with your application by implementing the vector store service that will handle document indexing and similarity search operations.