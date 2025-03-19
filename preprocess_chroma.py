#!/usr/bin/env python
# preprocess_chroma.py
# Script to enhance and optimize the existing Chroma database with advanced preprocessing

import os
import logging
import time
import json
import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import argparse
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata

# Optional NLP libraries for advanced processing
try:
    import spacy
    import nltk
    from nltk.corpus import stopwords
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_DIR = "merged_chroma_db"
ENHANCED_DB_DIR = "enhanced_chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o"
# Reduce concurrency to avoid rate limits
MAX_CONCURRENT_TASKS = 2
MAX_DOCS_PER_SUMMARY = 20
# Add rate limiting configuration
MIN_RETRY_DELAY = 1  # Start with 1-second delay
MAX_RETRY_DELAY = 60  # Up to 60-second delay
MAX_RETRY_ATTEMPTS = 10  # Maximum retry attempts
# Add delay between document processing to avoid rate limits
BATCH_PROCESSING_DELAY = 3  # seconds

# Load environment variables
load_dotenv()

class RateLimitError(Exception):
    """Exception for rate limit errors."""
    pass

class DocumentSummary(BaseModel):
    """Schema for document summary extraction."""
    summary: str = Field(description="A comprehensive summary of the document content")
    key_entities: List[str] = Field(description="List of important named entities mentioned in the document")
    document_type: str = Field(description="Classification of document type (e.g., memo, report, interview, telegram)")
    date_mentioned: Optional[str] = Field(description="Any dates mentioned in the document")
    locations_mentioned: List[str] = Field(description="Locations mentioned in the document")
    people_mentioned: List[str] = Field(description="People mentioned in the document")
    organizations_mentioned: List[str] = Field(description="Organizations mentioned in the document")
    importance_score: int = Field(description="Estimated importance of this document on a scale of 1-10")

class RateLimitedChatOpenAI(ChatOpenAI):
    """Rate-limited wrapper for ChatOpenAI with exponential backoff."""
    
    @retry(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=MIN_RETRY_DELAY, max=MAX_RETRY_DELAY),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=lambda retry_state: logger.info(f"Rate limit hit. Retrying in {retry_state.next_action.sleep} seconds...")
    )
    def invoke(self, *args, **kwargs):
        try:
            # Add a small random delay before each call to reduce likelihood of rate limits
            time.sleep(random.uniform(0.1, 0.5))
            return super().invoke(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "rate limit" in error_str.lower() or "429" in error_str:
                logger.warning(f"Rate limit error: {error_str}")
                raise RateLimitError(error_str)
            raise e

def setup_nlp_tools():
    """Initialize NLP tools if available."""
    if not ADVANCED_NLP_AVAILABLE:
        logger.warning("Advanced NLP libraries not available. Some features will be disabled.")
        return None, None
    
    # Initialize spaCy for named entity recognition
    try:
        logger.info("Loading spaCy NER model...")
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        nlp = None
    
    # Initialize NLTK resources
    try:
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
    
    return nlp

def extract_metadata_with_llm(doc: Document, llm):
    """Use LLM to extract metadata from a document."""
    parser = PydanticOutputParser(pydantic_object=DocumentSummary)
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert analyst specializing in declassified intelligence documents.
    
    Analyze the following document excerpt and extract key metadata.
    
    DOCUMENT CONTENT:
    {document_content}
    
    Extract the following information in a structured format:
    1. A concise yet comprehensive summary (2-3 sentences)
    2. Key named entities mentioned
    3. Document type classification
    4. Any dates mentioned
    5. Locations mentioned
    6. People mentioned
    7. Organizations mentioned
    8. Importance score (1-10 scale) based on historical significance
    
    {format_instructions}
    """)
    
    try:
        format_instructions = parser.get_format_instructions()
    except AttributeError:
        # Fallback for newer versions of LangChain/Pydantic
        format_instructions = """
        Return a JSON object with the following keys:
        summary: string
        key_entities: list of strings
        document_type: string
        date_mentioned: string or null
        locations_mentioned: list of strings
        people_mentioned: list of strings
        organizations_mentioned: list of strings
        importance_score: integer (1-10)
        """
    
    _input = {
        "document_content": doc.page_content,
        "format_instructions": format_instructions
    }
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke(_input)
        return result.dict()
    except Exception as e:
        # Check if it's a rate limit error
        error_str = str(e)
        if "rate limit" in error_str.lower() or "429" in error_str:
            logger.warning(f"Rate limit error in extract_metadata_with_llm: {error_str}")
            # The wrapper will handle retries, but we need to propagate the error
            raise
        
        logger.error(f"Error extracting metadata with LLM: {e}")
        # Provide a basic fallback
        return {
            "summary": "Error generating summary",
            "key_entities": [],
            "document_type": "unknown",
            "date_mentioned": None,
            "locations_mentioned": [],
            "people_mentioned": [],
            "organizations_mentioned": [],
            "importance_score": 5
        }

def extract_entities_with_spacy(text: str, nlp):
    """Extract named entities using spaCy."""
    if not nlp:
        return {
            "locations": [],
            "people": [],
            "organizations": [],
            "dates": [],
            "miscellaneous": []
        }
    
    try:
        doc = nlp(text)
        entities = {
            "locations": [],
            "people": [],
            "organizations": [],
            "dates": [],
            "miscellaneous": []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                if ent.text not in entities["locations"]:
                    entities["locations"].append(ent.text)
            elif ent.label_ == "PERSON":
                if ent.text not in entities["people"]:
                    entities["people"].append(ent.text)
            elif ent.label_ == "ORG":
                if ent.text not in entities["organizations"]:
                    entities["organizations"].append(ent.text)
            elif ent.label_ == "DATE":
                if ent.text not in entities["dates"]:
                    entities["dates"].append(ent.text)
            else:
                if ent.text not in entities["miscellaneous"]:
                    entities["miscellaneous"].append(ent.text)
        
        return entities
    except Exception as e:
        logger.error(f"Error extracting entities with spaCy: {e}")
        return {
            "locations": [],
            "people": [],
            "organizations": [],
            "dates": [],
            "miscellaneous": []
        }

def create_document_summaries(documents: List[Document], batch_size: int = 10):
    """Create document summaries and extract metadata using LLM."""
    logger.info(f"Generating document summaries and metadata for {len(documents)} documents")
    
    # Initialize the LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Cannot generate summaries.")
        return []
    
    # Use the rate-limited version
    llm = RateLimitedChatOpenAI(
        api_key=api_key,
        model=GPT_MODEL,
        temperature=0,
        max_tokens=1000
    )
    
    enhanced_docs = []
    nlp = setup_nlp_tools()
    
    # Process documents sequentially with small batches to better manage rate limits
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_enhanced_docs = []
        
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
            futures = []
            
            for doc in batch:
                futures.append(executor.submit(enhance_document, doc, llm, nlp))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
                try:
                    enhanced_doc = future.result()
                    batch_enhanced_docs.append(enhanced_doc)
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
        
        enhanced_docs.extend(batch_enhanced_docs)
        
        # Add delay between batches to avoid hitting rate limits
        if i + batch_size < len(documents):
            delay = BATCH_PROCESSING_DELAY * (1 + random.uniform(-0.2, 0.2))  # Add some randomness
            logger.info(f"Pausing for {delay:.2f} seconds before processing next batch to avoid rate limits...")
            time.sleep(delay)
    
    logger.info(f"Successfully generated metadata for {len(enhanced_docs)} documents")
    return enhanced_docs

def enhance_document(doc: Document, llm, nlp):
    """Enhance a single document with summaries and metadata."""
    try:
        # Extract metadata with LLM
        llm_metadata = extract_metadata_with_llm(doc, llm)
        
        # Extract entities with spaCy if available
        if nlp:
            spacy_entities = extract_entities_with_spacy(doc.page_content, nlp)
        else:
            spacy_entities = None
        
        # Combine metadata
        enhanced_metadata = doc.metadata.copy() if doc.metadata else {}
        
        # Handle the llm_metadata which might be a dict or an object
        if llm_metadata:
            if isinstance(llm_metadata, dict):
                metadata_dict = llm_metadata
            else:
                # Try to convert to dict - for newer Pydantic versions
                try:
                    metadata_dict = llm_metadata.dict()
                except AttributeError:
                    try:
                        metadata_dict = llm_metadata.model_dump()
                    except AttributeError:
                        # Fallback to manual extraction
                        metadata_dict = {
                            "summary": getattr(llm_metadata, "summary", ""),
                            "document_type": getattr(llm_metadata, "document_type", "unknown"),
                            "importance_score": getattr(llm_metadata, "importance_score", 5),
                            "people_mentioned": getattr(llm_metadata, "people_mentioned", []),
                            "organizations_mentioned": getattr(llm_metadata, "organizations_mentioned", []),
                            "locations_mentioned": getattr(llm_metadata, "locations_mentioned", []),
                            "date_mentioned": getattr(llm_metadata, "date_mentioned", None),
                            "key_entities": getattr(llm_metadata, "key_entities", [])
                        }
            
            enhanced_metadata.update({
                "summary": metadata_dict.get("summary", ""),
                "document_type": metadata_dict.get("document_type", "unknown"),
                "importance_score": metadata_dict.get("importance_score", 5),
                "people_mentioned": metadata_dict.get("people_mentioned", []),
                "organizations_mentioned": metadata_dict.get("organizations_mentioned", []),
                "locations_mentioned": metadata_dict.get("locations_mentioned", []),
                "date_mentioned": metadata_dict.get("date_mentioned", None),
                "key_entities": metadata_dict.get("key_entities", []),
                "nlp_processed": True
            })
        
        # Add spaCy entities if available
        if spacy_entities:
            enhanced_metadata.update({
                "spacy_locations": spacy_entities.get("locations", []),
                "spacy_people": spacy_entities.get("people", []),
                "spacy_organizations": spacy_entities.get("organizations", []),
                "spacy_dates": spacy_entities.get("dates", []),
                "spacy_miscellaneous": spacy_entities.get("miscellaneous", [])
            })
        
        # Create a new document with enhanced metadata
        enhanced_doc = Document(
            page_content=doc.page_content,
            metadata=enhanced_metadata
        )
        
        return enhanced_doc
    except Exception as e:
        logger.error(f"Error enhancing document: {e}")
        # Return a document with basic metadata
        basic_metadata = doc.metadata.copy() if doc.metadata else {}
        basic_metadata.update({
            "nlp_processed": False,
            "processing_error": str(e)
        })
        return Document(page_content=doc.page_content, metadata=basic_metadata)

def create_hierarchical_chunks(documents: List[Document]):
    """Create hierarchical chunks (small and large) from documents."""
    logger.info(f"Creating hierarchical chunks from {len(documents)} documents")
    
    # Create small chunks for specific retrieval
    small_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create larger chunks for context
    large_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    small_chunks = []
    large_chunks = []
    summary_chunks = []
    
    for doc in tqdm(documents, desc="Creating hierarchical chunks"):
        # Get document ID or create one
        doc_id = doc.metadata.get("source", "unknown")
        
        # Create small chunks
        small_docs = small_splitter.split_documents([doc])
        for i, small_doc in enumerate(small_docs):
            small_doc.metadata.update({
                "chunk_type": "small",
                "chunk_id": f"{doc_id}_small_{i}",
                "parent_id": doc_id
            })
            small_chunks.append(small_doc)
        
        # Create large chunks
        large_docs = large_splitter.split_documents([doc])
        for i, large_doc in enumerate(large_docs):
            large_doc.metadata.update({
                "chunk_type": "large",
                "chunk_id": f"{doc_id}_large_{i}",
                "parent_id": doc_id
            })
            large_chunks.append(large_doc)
        
        # Create summary chunk if summary exists
        summary = doc.metadata.get("summary", "")
        if summary:
            summary_doc = Document(
                page_content=summary,
                metadata={
                    "chunk_type": "summary",
                    "chunk_id": f"{doc_id}_summary",
                    "parent_id": doc_id,
                    "document_type": doc.metadata.get("document_type", "unknown"),
                    "importance_score": doc.metadata.get("importance_score", 5),
                    "people_mentioned": doc.metadata.get("people_mentioned", []),
                    "organizations_mentioned": doc.metadata.get("organizations_mentioned", []),
                    "locations_mentioned": doc.metadata.get("locations_mentioned", []),
                    "date_mentioned": doc.metadata.get("date_mentioned", None)
                }
            )
            summary_chunks.append(summary_doc)
    
    logger.info(f"Created {len(small_chunks)} small chunks, {len(large_chunks)} large chunks, and {len(summary_chunks)} summary chunks")
    return small_chunks, large_chunks, summary_chunks

def filter_metadata_for_chroma(documents: List[Document]) -> List[Document]:
    """Filter complex metadata that Chroma cannot store."""
    filtered_docs = []
    for doc in documents:
        # Create filtered metadata dictionary that only contains simple types
        filtered_metadata = {}
        if doc.metadata:
            for key, value in doc.metadata.items():
                # Skip None values
                if value is None:
                    # Replace None with empty string
                    filtered_metadata[key] = ""
                # Convert lists to strings if needed
                elif isinstance(value, list):
                    filtered_metadata[key] = ", ".join([str(item) for item in value])
                # Include only simple types
                elif isinstance(value, (str, int, float, bool)):
                    filtered_metadata[key] = value
                # Convert any other types to string
                else:
                    filtered_metadata[key] = str(value)
        
        # Create a new document with filtered metadata
        filtered_doc = Document(
            page_content=doc.page_content,
            metadata=filtered_metadata
        )
        filtered_docs.append(filtered_doc)
    return filtered_docs

def create_enhanced_chroma_db(chunks: List[Document], embedding_function, db_path: str, collection_name: str = "documents"):
    """Create a Chroma database with the given documents and embedding function."""
    logger.info(f"Creating Chroma DB at {db_path} with {len(chunks)} chunks")
    
    # Ensure the directory exists
    os.makedirs(db_path, exist_ok=True)
    
    # Filter metadata to ensure it's compatible with Chroma
    filtered_chunks = filter_metadata_for_chroma(chunks)
    
    # Create DB
    try:
        db = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=embedding_function,
            persist_directory=db_path,
            collection_name=collection_name
        )
        logger.info(f"Successfully created Chroma DB with {len(filtered_chunks)} chunks")
        db.persist()
        return db
    except Exception as e:
        logger.error(f"Error creating Chroma DB: {e}")
        return None

def process_existing_chroma(
    source_db_dir: str = CHROMA_DB_DIR,
    target_db_dir: str = ENHANCED_DB_DIR,
    batch_size: int = 100,
    sample_size: int = 0,
    max_batches: int = 0,
    concurrent_tasks: int = None,
    batch_delay: float = None,
    skip_processed: bool = False
):
    """Process existing Chroma database to create enhanced version with hierarchical chunks and metadata."""
    start_time = time.time()
    logger.info(f"Starting enhanced preprocessing of ChromaDB from {source_db_dir}")
    
    # Update global rate limiting settings if provided
    global MAX_CONCURRENT_TASKS, BATCH_PROCESSING_DELAY
    if concurrent_tasks is not None:
        MAX_CONCURRENT_TASKS = concurrent_tasks
        logger.info(f"Using {MAX_CONCURRENT_TASKS} concurrent tasks")
    if batch_delay is not None:
        BATCH_PROCESSING_DELAY = batch_delay
        logger.info(f"Using {BATCH_PROCESSING_DELAY}s delay between batches")
    
    # Initialize embedding functions
    logger.info(f"Initializing embedding functions")
    hf_embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Try to use OpenAI embeddings if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            logger.info(f"Initializing OpenAI embedding function with model {OPENAI_EMBEDDING_MODEL}")
            openai_embedding_function = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=api_key
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {e}. Falling back to HuggingFace.")
            openai_embedding_function = None
    else:
        logger.warning("OpenAI API key not found. Using HuggingFace embeddings only.")
        openai_embedding_function = None
    
    # Check for already processed documents if skip_processed is True
    processed_doc_ids = set()
    if skip_processed and os.path.exists(target_db_dir):
        try:
            logger.info("Checking for already processed documents...")
            metadata_file = os.path.join(target_db_dir, "metadata", "document_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    for doc_meta in metadata:
                        if 'source' in doc_meta:
                            processed_doc_ids.add(doc_meta['source'])
                        elif 'id' in doc_meta:
                            processed_doc_ids.add(doc_meta['id'])
                logger.info(f"Found {len(processed_doc_ids)} already processed documents that will be skipped")
            else:
                logger.warning("No metadata file found in target directory. Will process all documents.")
        except Exception as e:
            logger.error(f"Error loading processed document IDs: {e}. Will process all documents.")
    
    # Load source database
    try:
        logger.info(f"Loading source ChromaDB from {source_db_dir}")
        source_db = Chroma(
            persist_directory=source_db_dir,
            embedding_function=hf_embedding_function
        )
        
        # Get all documents
        logger.info("Retrieving all documents from the source database")
        results = source_db.get(include=["documents", "metadatas"])
        documents = []
        
        for i, doc_content in enumerate(results["documents"]):
            if doc_content:
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                
                # Get document ID for skip check
                doc_id = metadata.get('source', metadata.get('id', f"doc_{i}"))
                
                # Skip if already processed
                if skip_processed and doc_id in processed_doc_ids:
                    continue
                    
                doc = Document(page_content=doc_content, metadata=metadata)
                documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} documents from source database to process")
        if skip_processed and processed_doc_ids:
            logger.info(f"Skipped {len(processed_doc_ids)} already processed documents")
        
        # Apply sample size limit if specified
        if sample_size > 0 and sample_size < len(documents):
            logger.info(f"Using sample of {sample_size} documents for processing")
            random.seed(42)  # For reproducibility
            documents = random.sample(documents, sample_size)
        
        # Process documents in batches
        all_enhanced_docs = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        # Apply max_batches limit if specified
        if max_batches > 0 and max_batches < total_batches:
            logger.info(f"Processing limited to {max_batches} batches (out of {total_batches})")
            total_batches = max_batches
            documents = documents[:max_batches * batch_size]
        
        for i in range(0, len(documents), batch_size):
            # Check if we've reached the max_batches limit
            if max_batches > 0 and i // batch_size >= max_batches:
                logger.info(f"Reached maximum batch limit of {max_batches}")
                break
                
            batch = documents[i:i+batch_size]
            batch_num = i//batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} documents")
            
            # Create document summaries and extract metadata
            try:
                enhanced_docs = create_document_summaries(batch, batch_size=min(5, batch_size))  # Use smaller sub-batches
                all_enhanced_docs.extend(enhanced_docs)
                
                # Add a longer pause between main batches
                if batch_num < total_batches:
                    inter_batch_delay = BATCH_PROCESSING_DELAY * 2 * (1 + random.uniform(-0.1, 0.3))
                    logger.info(f"Batch {batch_num} complete. Pausing for {inter_batch_delay:.2f} seconds to respect rate limits...")
                    time.sleep(inter_batch_delay)
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Continue with next batch after a longer pause to recover from rate limits
                recovery_delay = BATCH_PROCESSING_DELAY * 4
                logger.info(f"Pausing for {recovery_delay} seconds to recover from error...")
                time.sleep(recovery_delay)
        
        # Check if we have any enhanced documents
        if not all_enhanced_docs:
            logger.error("No documents were successfully enhanced")
            return {
                "status": "error",
                "error": "No documents were successfully enhanced"
            }
        
        # Create hierarchical chunks
        try:
            logger.info("Creating hierarchical chunks")
            small_chunks, large_chunks, summary_chunks = create_hierarchical_chunks(all_enhanced_docs)
        except Exception as e:
            logger.error(f"Error creating hierarchical chunks: {e}")
            return {
                "status": "error",
                "error": f"Failed to create hierarchical chunks: {str(e)}"
            }
        
        # Create databases with different embeddings
        if openai_embedding_function:
            try:
                logger.info("Creating databases with OpenAI embeddings")
                db_dir_openai = os.path.join(target_db_dir, "openai")
                
                # Create databases with different chunk types
                create_enhanced_chroma_db(small_chunks, openai_embedding_function, os.path.join(db_dir_openai, "small_chunks"), "small_chunks")
                # Add delay between DB creation operations
                time.sleep(BATCH_PROCESSING_DELAY)
                
                create_enhanced_chroma_db(large_chunks, openai_embedding_function, os.path.join(db_dir_openai, "large_chunks"), "large_chunks")
                time.sleep(BATCH_PROCESSING_DELAY)
                
                create_enhanced_chroma_db(summary_chunks, openai_embedding_function, os.path.join(db_dir_openai, "summary_chunks"), "summary_chunks")
                time.sleep(BATCH_PROCESSING_DELAY)
                
                create_enhanced_chroma_db(all_enhanced_docs, openai_embedding_function, os.path.join(db_dir_openai, "full_docs"), "full_docs")
                time.sleep(BATCH_PROCESSING_DELAY * 2)  # Longer delay before switching to HuggingFace
            except Exception as e:
                logger.error(f"Error creating OpenAI embedding databases: {e}")
                # Continue to create HuggingFace databases
        
        # Always create HuggingFace embedding databases as fallback
        try:
            logger.info("Creating databases with HuggingFace embeddings")
            db_dir_hf = os.path.join(target_db_dir, "huggingface")
            
            create_enhanced_chroma_db(small_chunks, hf_embedding_function, os.path.join(db_dir_hf, "small_chunks"), "small_chunks")
            time.sleep(BATCH_PROCESSING_DELAY)
            
            create_enhanced_chroma_db(large_chunks, hf_embedding_function, os.path.join(db_dir_hf, "large_chunks"), "large_chunks")
            time.sleep(BATCH_PROCESSING_DELAY)
            
            create_enhanced_chroma_db(summary_chunks, hf_embedding_function, os.path.join(db_dir_hf, "summary_chunks"), "summary_chunks")
            time.sleep(BATCH_PROCESSING_DELAY)
            
            create_enhanced_chroma_db(all_enhanced_docs, hf_embedding_function, os.path.join(db_dir_hf, "full_docs"), "full_docs")
        except Exception as e:
            logger.error(f"Error creating HuggingFace embedding databases: {e}")
            return {
                "status": "error",
                "error": f"Failed to create vector databases: {str(e)}"
            }
        
        # Save document metadata as JSON for easy access
        try:
            logger.info("Saving document metadata to JSON")
            metadata_dir = os.path.join(target_db_dir, "metadata")
            os.makedirs(metadata_dir, exist_ok=True)
            
            metadata_file = os.path.join(metadata_dir, "document_metadata.json")
            doc_metadata = []
            
            for doc in all_enhanced_docs:
                doc_meta = doc.metadata.copy()
                doc_meta["content_preview"] = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                doc_metadata.append(doc_meta)
            
            with open(metadata_file, "w") as f:
                json.dump(doc_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            # Continue as this is not critical
        
        total_time = time.time() - start_time
        logger.info(f"Enhanced preprocessing completed in {total_time:.2f} seconds")
        
        return {
            "status": "success",
            "document_count": len(documents),
            "enhanced_document_count": len(all_enhanced_docs),
            "small_chunks_count": len(small_chunks),
            "large_chunks_count": len(large_chunks),
            "summary_chunks_count": len(summary_chunks),
            "processing_time": total_time
        }
        
    except Exception as e:
        logger.error(f"Error processing ChromaDB: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    """Main execution function"""
    # Make globals modifiable
    global MAX_CONCURRENT_TASKS, BATCH_PROCESSING_DELAY
    
    parser = argparse.ArgumentParser(description="Enhanced preprocessing for Chroma database")
    parser.add_argument("--source", default=CHROMA_DB_DIR, help="Source Chroma DB directory")
    parser.add_argument("--target", default=ENHANCED_DB_DIR, help="Target directory for enhanced DB")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--sample-size", type=int, default=0, help="Sample size for processing")
    parser.add_argument("--max-batches", type=int, default=0, help="Maximum number of batches to process")
    parser.add_argument("--concurrent-tasks", type=int, default=MAX_CONCURRENT_TASKS, 
                        help=f"Number of concurrent tasks (default: {MAX_CONCURRENT_TASKS})")
    parser.add_argument("--batch-delay", type=float, default=BATCH_PROCESSING_DELAY,
                        help=f"Delay between batches in seconds (default: {BATCH_PROCESSING_DELAY})")
    parser.add_argument("--skip-processed", action="store_true", help="Skip already processed documents")
    args = parser.parse_args()
    
    # Update global settings if provided via command line
    if args.concurrent_tasks:
        MAX_CONCURRENT_TASKS = args.concurrent_tasks
    if args.batch_delay:
        BATCH_PROCESSING_DELAY = args.batch_delay
    
    result = process_existing_chroma(
        source_db_dir=args.source,
        target_db_dir=args.target,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        max_batches=args.max_batches,
        skip_processed=args.skip_processed
    )
    
    if result["status"] == "success":
        logger.info(f"Successfully processed {result['document_count']} documents in {result['processing_time']:.2f} seconds")
        logger.info(f"Created {result['small_chunks_count']} small chunks, {result['large_chunks_count']} large chunks, and {result['summary_chunks_count']} summary chunks")
    else:
        logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 