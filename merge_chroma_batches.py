#!/usr/bin/env python
# merge_chroma_batches.py
# Script to merge all batch ChromaDBs into a single optimized database

import os
import shutil
import logging
import time
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BATCH_DB_DIR = "chroma_db"
MERGED_DB_DIR = "merged_chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_BATCH_SIZE = 40000  # ChromaDB has a limit of ~41K documents per batch

def main():
    start_time = time.time()
    logger.info(f"Starting ChromaDB batch merge process")
    
    # Create embedding function
    logger.info(f"Initializing embedding function with model: {MODEL_NAME}")
    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Check if the source directory exists
    if not os.path.exists(BATCH_DB_DIR):
        logger.error(f"Batch ChromaDB directory {BATCH_DB_DIR} does not exist")
        return
    
    # Get all batch directories
    batch_dirs = [d for d in os.listdir(BATCH_DB_DIR) if d.startswith("batch_")]
    
    if not batch_dirs:
        logger.error(f"No batch directories found in {BATCH_DB_DIR}")
        return
    
    # Sort batch directories numerically
    batch_dirs.sort(key=lambda x: int(x.split("_")[1]))
    logger.info(f"Found {len(batch_dirs)} batch directories: {batch_dirs}")
    
    # Create the destination directory if it doesn't exist
    if os.path.exists(MERGED_DB_DIR):
        logger.warning(f"Merged DB directory {MERGED_DB_DIR} already exists. It will be overwritten.")
        shutil.rmtree(MERGED_DB_DIR)
    
    os.makedirs(MERGED_DB_DIR, exist_ok=True)
    
    # Create new merged ChromaDB
    logger.info(f"Creating new merged ChromaDB at {MERGED_DB_DIR}")
    merged_db = Chroma(persist_directory=MERGED_DB_DIR, embedding_function=embedding_function)
    
    # Track all documents
    all_docs = []
    total_docs = 0
    
    # Process each batch directory
    for batch_idx, batch_dir in enumerate(batch_dirs, 1):
        batch_start_time = time.time()
        full_batch_path = os.path.join(BATCH_DB_DIR, batch_dir)
        
        logger.info(f"Processing batch {batch_idx}/{len(batch_dirs)}: {batch_dir}")
        
        try:
            # Load the existing batch with LangChain's Chroma wrapper
            batch_db = Chroma(persist_directory=full_batch_path, embedding_function=embedding_function)
            
            # Use langchain's get() method to get all documents as Document objects
            batch_docs = batch_db.get(include=["documents", "metadatas", "embeddings"])
            
            if not batch_docs["documents"]:
                logger.warning(f"No documents found in batch {batch_dir}")
                continue
                
            doc_count = len(batch_docs["documents"])
            logger.info(f"Found {doc_count} documents in batch {batch_dir}")
            
            # Convert to Document objects
            batch_documents = []
            for i, doc_content in enumerate(batch_docs["documents"]):
                if doc_content:
                    metadata = batch_docs["metadatas"][i] if batch_docs["metadatas"] else {}
                    doc = Document(page_content=doc_content, metadata=metadata)
                    batch_documents.append(doc)
            
            # Add these documents to our cumulative list
            all_docs.extend(batch_documents)
            total_docs += len(batch_documents)
            
            # If we have accumulated enough documents, add them to the merged DB
            if len(all_docs) >= MAX_BATCH_SIZE:
                _add_documents_in_batches(merged_db, all_docs, MAX_BATCH_SIZE)
                all_docs = []  # Clear the list after adding to DB
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Processed batch {batch_idx}/{len(batch_dirs)} with {len(batch_documents)} documents in {batch_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_dir}: {str(e)}")
    
    # Add any remaining documents to the merged DB
    if all_docs:
        logger.info(f"Adding final batch of {len(all_docs)} documents to merged ChromaDB")
        _add_documents_in_batches(merged_db, all_docs, MAX_BATCH_SIZE)
    
    total_time = time.time() - start_time
    logger.info(f"Merge completed in {total_time:.2f} seconds")
    logger.info(f"Successfully processed {total_docs} documents from {len(batch_dirs)} batches")

def _add_documents_in_batches(db, documents, batch_size):
    """Add documents to the database in batches to avoid exceeding ChromaDB limits."""
    total = len(documents)
    logger.info(f"Adding {total} documents to ChromaDB in batches of {batch_size}")
    
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        batch = documents[i:end_idx]
        logger.info(f"Adding batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}: documents {i} to {end_idx-1}")
        db.add_documents(batch)
        logger.info(f"Successfully added batch of {len(batch)} documents")
    
    # Persist after adding all batches
    logger.info("Persisting ChromaDB to disk...")
    db.persist()
    logger.info("ChromaDB persisted successfully")

if __name__ == "__main__":
    main() 