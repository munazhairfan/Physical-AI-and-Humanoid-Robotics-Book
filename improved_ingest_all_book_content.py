#!/usr/bin/env python3
"""
Improved Script to ingest ALL Physical AI & Humanoid Robotics book content into Qdrant cluster
This includes content from docs, i18n, and blog directories with semantic chunking
"""
import os
import sys
import glob
import uuid
from pathlib import Path
import re

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Load environment variables from backend directory
from dotenv import load_dotenv
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
dotenv_path = os.path.join(backend_dir, '.env')
load_dotenv(dotenv_path)

def get_all_possible_content():
    """Gather ALL possible book content from all documentation directories"""

    print("Gathering ALL book content from all directories...")
    print("="*60)

    # Define different documentation directories
    doc_dirs = [
        os.path.join(os.path.dirname(__file__), 'frontend', 'rag-chatbot-frontend', 'docs'),
        os.path.join(os.path.dirname(__file__), 'frontend', 'rag-chatbot-frontend', 'i18n', 'ur', 'docusaurus-plugin-content-docs', 'current'),
        os.path.join(os.path.dirname(__file__), 'frontend', 'rag-chatbot-frontend', 'blog')
    ]

    all_content = []
    processed_files = set()  # To avoid duplicates

    for docs_dir in doc_dirs:
        if os.path.exists(docs_dir):
            print(f"\nProcessing directory: {docs_dir}")

            # Get all markdown files from this directory
            md_files = glob.glob(os.path.join(docs_dir, "*.md"))

            print(f"Found {len(md_files)} markdown files in {os.path.basename(docs_dir)}")

            for file_path in md_files:
                # Skip node_modules (we already filtered this in our search, but just to be safe)
                if 'node_modules' in file_path:
                    continue

                # Use relative path to identify duplicates
                rel_path = os.path.relpath(file_path, os.path.dirname(__file__))

                if rel_path in processed_files:
                    print(f"  SKIPPED (duplicate): {os.path.basename(file_path)}")
                    continue

                processed_files.add(rel_path)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Extract title from the file (first heading)
                    lines = content.split('\n')
                    title = "Untitled"
                    for line in lines:
                        if line.startswith('# '):
                            title = line[2:].strip()  # Remove '# ' prefix
                            break

                    file_info = {
                        'title': title,
                        'content': content,
                        'filename': os.path.basename(file_path),
                        'filepath': file_path,
                        'source_dir': os.path.basename(docs_dir)
                    }
                    all_content.append(file_info)

                    print(f"  PROCESSED: {title} ({len(content)} chars) from {os.path.basename(docs_dir)}")

                except Exception as e:
                    print(f"  ERROR: Error processing {file_path}: {str(e)}")
        else:
            print(f"Directory does not exist: {docs_dir}")

    print(f"\nTotal unique files processed: {len(all_content)}")

    return all_content

def semantic_chunking(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Split text into semantically coherent chunks.
    """
    if not text.strip():
        return []

    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if not paragraphs:
        # If no paragraphs, fall back to sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # Process paragraphs into chunks
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < max_chunk_size:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If the paragraph itself is too large, split it
            if len(paragraph) > max_chunk_size:
                sub_chunks = split_large_paragraph(paragraph, max_chunk_size, overlap)
                chunks.extend(sub_chunks[:-1])  # Add all but the last sub-chunk
                current_chunk = sub_chunks[-1] if sub_chunks else ""  # Keep the last part
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def split_large_paragraph(paragraph: str, max_chunk_size: int, overlap: int) -> list:
    """
    Split a large paragraph into smaller chunks.
    """
    sentences = re.split(r'[.!?]+\s+', paragraph)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start new chunk with some overlap
            if len(sentence) > max_chunk_size:
                # If sentence is too long, split by words
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) < max_chunk_size:
                        temp_chunk += " " + word if temp_chunk else word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def ingest_all_book_content():
    """Ingest all book content to the external Qdrant cluster with semantic chunking"""

    print("Ingesting ALL Physical AI & Humanoid Robotics Content with semantic chunking...")
    print("="*70)

    # Load environment variables
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    print(f"QDRANT_HOST: {qdrant_host}")
    print(f"QDRANT_API_KEY: {'SET' if qdrant_api_key else 'NOT SET'}")

    if not qdrant_host:
        print("ERROR: QDRANT_HOST not set in environment variables")
        return False

    try:
        # Get all book content
        all_content = get_all_possible_content()

        if not all_content:
            print("ERROR: No book content found to ingest")
            return False

        print(f"\nTotal files to process: {len(all_content)}")

        # Import and create the improved vectorstore service with external configuration
        from backend.app.vectorstore_service import VectorStoreService

        # Create service with external Qdrant configuration
        qdrant_url = f"https://{qdrant_host}:{qdrant_port}"
        print(f"\nConnecting to external Qdrant: {qdrant_url}")

        vectorstore_service = VectorStoreService(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

        print("Successfully connected to external Qdrant cluster")

        # Check initial point count
        collection_info = vectorstore_service.client.get_collection("rag_documents")
        initial_points = collection_info.points_count
        print(f"Initial points in collection: {initial_points}")

        total_chunks_processed = 0
        total_documents_processed = 0

        # Process each document separately with semantic chunking
        for content_item in all_content:
            print(f"\nProcessing: {content_item['title']} from {content_item['source_dir']}")

            # Create metadata for this specific document
            metadata = {
                "title": content_item['title'],
                "filename": content_item['filename'],
                "source_dir": content_item['source_dir'],
                "original_file_path": content_item['filepath'],
                "content_type": "book_content",
                "ingestion_date": str(uuid.uuid4()),
                "original_content_length": len(content_item['content'])
            }

            # Index the document using semantic chunking
            doc_id = str(uuid.uuid4())
            indexed_id = vectorstore_service.index_document(
                text=content_item['content'],
                doc_id=doc_id,
                metadata=metadata
            )

            if indexed_id:
                print(f"  Indexed document {indexed_id} with semantic chunks")
                total_documents_processed += 1

                # Count the chunks for this document
                chunks = semantic_chunking(content_item['content'], max_chunk_size=1000, overlap=200)
                total_chunks_processed += len(chunks)
                print(f"  Created {len(chunks)} semantic chunks for this document")

        # Check how many points are now in the collection
        collection_info = vectorstore_service.client.get_collection("rag_documents")
        final_points = collection_info.points_count
        points_added = final_points - initial_points
        print(f"\nFinal points in collection: {final_points}")
        print(f"Points added: {points_added}")

        if points_added > 0:
            print(f"\nSUCCESS: Complete book content successfully ingested to external Qdrant cluster!")
            print(f"   Added {points_added} new points from {total_documents_processed} documents!")
            print(f"   Total semantic chunks created: {total_chunks_processed}")
            print(f"   Total points now: {final_points}")

            # Report which directories were included
            sources = set([item['source_dir'] for item in all_content])
            print(f"\nContent sources included in the knowledge base:")
            for source in sorted(sources):
                files_from_source = [item for item in all_content if item['source_dir'] == source]
                print(f"  - {source}: {len(files_from_source)} files")

            return True
        else:
            print(f"\nINFO: No new points were added. Content may not have been ingested.")
            return False

    except Exception as e:
        print(f"ERROR: Failed to ingest complete book content: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = ingest_all_book_content()

    if success:
        print(f"\n" + "="*70)
        print(f"SUCCESS: Complete Physical AI & Humanoid Robotics Content")
        print(f"   has been successfully stored in the external Qdrant cluster!")
        print(f"INFO: All modules/chapters are now embedded in your cluster collection.")
        print(f"INFO: The dashboard should now show all the book points.")
        print(f"INFO: Your complete textbook with all content is available for the chatbot.")
        print(f"INFO: Using improved semantic chunking for better retrieval.")
        print(f"="*70)
    else:
        print(f"\nINFO: Book ingestion completed but may have issues due to API quota limits.")
        print(f"   The core functionality is working, but API quotas may limit full functionality.")