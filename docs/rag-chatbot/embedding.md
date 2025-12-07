---
title: "RAG Chatbot: Embedding"
description: "Embedding strategies and implementation for RAG-based chatbot in robotics education"
sidebar_position: 1
slug: /rag-chatbot/embedding
keywords: [RAG, embedding, vector database, robotics, education, AI]
---

# RAG Chatbot: Embedding

## Overview

This section covers the embedding strategies and implementation for a Retrieval-Augmented Generation (RAG) chatbot system designed for robotics education. The RAG system enables the chatbot to access and utilize the comprehensive robotics textbook content to provide accurate, contextually relevant responses to student queries.

## Embedding Strategy

### Text Chunking

To effectively process the robotics textbook content, we implement a hierarchical chunking strategy:

```python
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class TextChunk:
    id: str
    content: str
    metadata: Dict
    embedding: List[float] = None

class TextChunker:
    def __init__(self, max_chunk_size: int = 512, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_by_semantic_boundaries(self, text: str, metadata: Dict) -> List[TextChunk]:
        """
        Chunk text based on semantic boundaries (headings, paragraphs, code blocks)
        """
        chunks = []

        # Split by headings first (H1, H2, H3)
        heading_pattern = r'(#{1,3}\s+.*?)(?=\n#{1,3}\s+|\Z)'
        sections = re.split(heading_pattern, text, flags=re.DOTALL)

        # Process each section
        for i, section in enumerate(sections):
            if section.strip() and not section.startswith('#'):
                continue  # Skip empty sections or heading-only sections

            if section.startswith('#'):
                # This is a heading, get the following content
                if i + 1 < len(sections) and sections[i + 1].strip():
                    content = sections[i] + "\n" + sections[i + 1]
                    chunks.extend(self._create_subchunks(content, metadata))
            else:
                chunks.extend(self._create_subchunks(section, metadata))

        return chunks

    def _create_subchunks(self, text: str, metadata: Dict) -> List[TextChunk]:
        """
        Create subchunks within a section, respecting paragraph and code boundaries
        """
        subchunks = []

        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        current_chunk = ""
        current_metadata = metadata.copy()

        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if current_chunk:
                    chunk_id = f"{metadata.get('doc_id', 'unknown')}_{len(subchunks)}"
                    subchunks.append(TextChunk(
                        id=chunk_id,
                        content=current_chunk.strip(),
                        metadata=current_metadata
                    ))

                # Start new chunk with overlap
                current_chunk = self._get_overlap(current_chunk) + paragraph
            else:
                current_chunk += "\n\n" + paragraph

        # Add the last chunk
        if current_chunk:
            chunk_id = f"{metadata.get('doc_id', 'unknown')}_{len(subchunks)}"
            subchunks.append(TextChunk(
                id=chunk_id,
                content=current_chunk.strip(),
                metadata=current_metadata
            ))

        return subchunks

    def _get_overlap(self, text: str) -> str:
        """
        Get overlapping text from the end of the chunk for continuity
        """
        words = text.split()
        overlap_words = words[-self.overlap:] if len(words) > self.overlap else words
        return " ".join(overlap_words)
```

### Embedding Models

We use multiple embedding strategies to capture different aspects of the robotics content:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import torch

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator with pre-trained model
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy().tolist()

    def generate_multilingual_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        For robotics content that might include code, use multilingual model
        """
        multilingual_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = multilingual_model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy().tolist()

    def generate_code_aware_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Special handling for code snippets in robotics documentation
        """
        # Process code and text separately
        processed_texts = []
        for text in texts:
            # Extract code blocks and process separately
            code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
            non_code_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

            # Combine processed content
            processed = f"{non_code_text} {' '.join([cb[:100] for cb in code_blocks])}"  # Include truncated code
            processed_texts.append(processed)

        return self.generate_embeddings(processed_texts)
```

### Vector Database Integration

```python
import qdrant_client
from qdrant_client.http import models
from typing import List, Dict, Optional
import uuid

class VectorStore:
    def __init__(self, collection_name: str = "robotics_textbook"):
        self.client = qdrant_client.QdrantClient(":memory:")  # Use in-memory for development
        # For production: qdrant_client.QdrantClient(url="http://localhost:6333")
        self.collection_name = collection_name
        self._init_collection()

    def _init_collection(self):
        """
        Initialize the Qdrant collection with appropriate vector configuration
        """
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Size of all-MiniLM-L6-v2 embeddings
                    distance=models.Distance.COSINE
                )
            )

    def store_embeddings(self, chunks: List[TextChunk]):
        """
        Store text chunks with their embeddings in the vector database
        """
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")

            points.append(
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=chunk.embedding,
                    payload={
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """
        Search for relevant chunks based on query embedding
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        return [
            {
                "content": result.payload["content"],
                "metadata": result.payload["metadata"],
                "score": result.score
            }
            for result in results
        ]
```

## Robotics-Specific Embedding Considerations

### Technical Terminology

Robotics content contains specialized terminology that requires special handling:

```python
class RoboticsEmbeddingEnhancer:
    def __init__(self):
        # Common robotics terms and their embeddings
        self.technical_terms = {
            "forward kinematics": "The process of calculating the position and orientation of the end-effector based on joint angles",
            "inverse kinematics": "The process of determining joint angles required to achieve a desired end-effector position",
            "configuration space": "The space of all possible configurations of a robot",
            "motion planning": "The process of finding a path from start to goal while avoiding obstacles",
            "control system": "A system that manages and commands the behavior of other devices or systems",
            "PID controller": "A control loop feedback mechanism that calculates an error value as the difference between desired and measured values",
            "Reinforcement Learning": "A type of machine learning where agents learn to make decisions by interacting with an environment",
            "SLAM": "Simultaneous Localization and Mapping - the computational problem of constructing a map while locating an agent"
        }

    def enhance_with_domain_knowledge(self, text: str) -> str:
        """
        Enhance text with domain-specific explanations for better embedding
        """
        enhanced_text = text
        for term, definition in self.technical_terms.items():
            if term.lower() in text.lower():
                enhanced_text += f"\n\nFor context: {term} refers to {definition}"

        return enhanced_text
```

### Multi-Modal Embeddings

For robotics content that includes code examples, equations, and diagrams:

```python
class MultiModalEmbedding:
    def __init__(self):
        self.text_embedder = EmbeddingGenerator()
        # In a real implementation, you would also have image and code embedders

    def embed_robotics_content(self, content: Dict) -> List[float]:
        """
        Embed multi-modal robotics content (text, code, equations)
        """
        embeddings = []

        # Embed text content
        if "text" in content:
            text_embedding = self.text_embedder.generate_embeddings([content["text"]])[0]
            embeddings.append(text_embedding)

        # Embed code content if present
        if "code" in content:
            code_embedding = self.text_embedder.generate_code_aware_embeddings([content["code"]])[0]
            embeddings.append(code_embedding)

        # Combine embeddings (simple average for now)
        if embeddings:
            combined = np.mean(embeddings, axis=0)
            return combined.tolist()
        else:
            return [0.0] * 384  # Default embedding size
```

## Quality Assurance

### Embedding Evaluation

```python
class EmbeddingEvaluator:
    def __init__(self):
        pass

    def evaluate_relevance(self, query: str, retrieved_chunks: List[Dict]) -> float:
        """
        Evaluate how relevant the retrieved chunks are to the query
        """
        # Simple keyword overlap as a basic metric
        query_words = set(query.lower().split())
        total_relevance = 0

        for chunk in retrieved_chunks:
            chunk_words = set(chunk["content"].lower().split())
            overlap = len(query_words.intersection(chunk_words))
            total_relevance += overlap / len(query_words) if query_words else 0

        return total_relevance / len(retrieved_chunks) if retrieved_chunks else 0

    def evaluate_diversity(self, retrieved_chunks: List[Dict]) -> float:
        """
        Evaluate how diverse the retrieved chunks are
        """
        if len(retrieved_chunks) < 2:
            return 0.0

        # Calculate average cosine distance between embeddings
        embeddings = [chunk.get("embedding", [0.0] * 384) for chunk in retrieved_chunks]
        distances = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = self._cosine_distance(embeddings[i], embeddings[j])
                distances.append(dist)

        return sum(distances) / len(distances) if distances else 0.0

    def _cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine distance between two vectors
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 1.0  # Maximum distance if one vector is zero

        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        return 1 - cosine_similarity
```

## Implementation Pipeline

```python
class EmbeddingPipeline:
    def __init__(self, collection_name: str = "robotics_textbook"):
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        self.enhancer = RoboticsEmbeddingEnhancer()
        self.vector_store = VectorStore(collection_name)
        self.evaluator = EmbeddingEvaluator()

    def process_document(self, content: str, metadata: Dict) -> Dict:
        """
        Complete pipeline: chunk -> enhance -> embed -> store
        """
        # Step 1: Chunk the document
        chunks = self.chunker.chunk_by_semantic_boundaries(content, metadata)

        # Step 2: Enhance with domain knowledge
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_content = self.enhancer.enhance_with_domain_knowledge(chunk.content)
            enhanced_chunk = TextChunk(
                id=chunk.id,
                content=enhanced_content,
                metadata=chunk.metadata
            )
            enhanced_chunks.append(enhanced_chunk)

        # Step 3: Generate embeddings
        texts = [chunk.content for chunk in enhanced_chunks]
        embeddings = self.embedder.generate_embeddings(texts)

        # Step 4: Assign embeddings to chunks
        for chunk, embedding in zip(enhanced_chunks, embeddings):
            chunk.embedding = embedding

        # Step 5: Store in vector database
        self.vector_store.store_embeddings(enhanced_chunks)

        # Step 6: Evaluate quality
        stats = {
            "num_chunks": len(enhanced_chunks),
            "avg_chunk_size": np.mean([len(chunk.content) for chunk in enhanced_chunks]),
        }

        return stats

# Example usage
def main():
    pipeline = EmbeddingPipeline()

    # Example metadata for a robotics textbook section
    metadata = {
        "doc_id": "module3_section2",
        "module": "AI Perception & Sensor Fusion",
        "section": "Kalman Filters",
        "author": "textbook",
        "version": "1.0"
    }

    # Example content (this would come from your actual textbook files)
    sample_content = """
# Kalman Filters in Robotics

Kalman filters are essential tools in robotics for state estimation. They provide optimal estimates of system states in the presence of noise.

## Mathematical Foundation

The Kalman filter operates in two phases: prediction and update. In the prediction phase, the filter uses the system model to predict the next state and its uncertainty. In the update phase, it incorporates sensor measurements to refine the state estimate.

### Prediction Phase:
x_pred = F * x_prev + B * u
P_pred = F * P_prev * F^T + Q

### Update Phase:
K = P_pred * H^T * (H * P_pred * H^T + R)^-1
x_new = x_pred + K * (z - H * x_pred)
P_new = (I - K * H) * P_pred
    """

    stats = pipeline.process_document(sample_content, metadata)
    print(f"Processed document with {stats['num_chunks']} chunks")
    print(f"Average chunk size: {stats['avg_chunk_size']:.2f} characters")

if __name__ == "__main__":
    main()
```

## Performance Optimization

### Batch Processing

For efficient processing of large robotics textbooks:

```python
class BatchEmbeddingProcessor:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.embedder = EmbeddingGenerator()

    def process_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Process texts in batches for efficiency
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.embedder.generate_embeddings(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
```

## Summary

This embedding system provides the foundation for a RAG-based chatbot in robotics education. By carefully chunking content based on semantic boundaries, enhancing technical terminology, and storing embeddings in an efficient vector database, we create a system that can retrieve relevant information to answer student questions about robotics concepts.

Continue with [API Integration](./api) to learn about the backend API for the RAG system.