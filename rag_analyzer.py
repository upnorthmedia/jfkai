import os
import fitz
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import logging
import time
from langchain.schema import Document
import tempfile
import concurrent.futures
import math
from threading import Lock
# LLM Whisperer imports
from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Optional: For RAG with an LLM ---
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings  # Or other embeddings
from langchain.vectorstores import Chroma
# Updated import path
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    logger.info("Successfully imported PyMuPDFLoader from langchain_community")
except ImportError:
    try:
        from langchain.document_loaders import PyMuPDFLoader
        logger.info("Successfully imported PyMuPDFLoader from langchain")
    except ImportError:
        logger.error("Failed to import PyMuPDFLoader from either path")

from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
PDF_DIR = "pdfs"
CHROMA_DB_DIR = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"  # A good, fast sentence transformer model.
CHUNK_SIZE = 1000 #1000
CHUNK_OVERLAP = 200 #200
# LLM Whisperer Configuration
USE_LLM_WHISPERER = True  # Set to True to use LLM Whisperer
LLM_WHISPERER_MODE = "form"  # Options: 'native-text', 'low-cost', 'form'
# Multithreading Configuration
MAX_THREADS = 10  # Maximum number of concurrent threads
BATCH_SIZE = 100  # Number of PDFs to process in each batch

# --- Step 1: Text Extraction using LLM Whisperer ---
def extract_text_with_llm_whisperer(pdf_path):
    """Extract text from a PDF using LLM Whisperer."""
    logger.info(f"Extracting text with LLM Whisperer from: {pdf_path}")
    
    try:
        # Initialize the LLM Whisperer client
        llm_whisperer_client = LLMWhispererClientV2()
        
        # Process the PDF with LLM Whisperer in high quality mode
        start_time = time.time()
        
        # Process with wait_for_completion=True to get the result synchronously
        result = llm_whisperer_client.whisper(
            file_path=pdf_path,
            mode=LLM_WHISPERER_MODE,  # Using form mode which supports OCR
            output_mode="line-printer",  # For better LLM consumption
            wait_for_completion=True,
            wait_timeout=300  # 5 minutes timeout
        )
        
        processing_time = time.time() - start_time
        
        # Extract the text from the result
        if result["status"] == "processed" and "extraction" in result and "result_text" in result["extraction"]:
            text_content = result["extraction"]["result_text"]
            logger.info(f"LLM Whisperer extraction successful in {processing_time:.2f} seconds. Extracted {len(text_content)} characters.")
            return text_content
        else:
            logger.warning(f"LLM Whisperer extraction failed or returned empty result: {result}")
            return ""
            
    except LLMWhispererClientException as e:
        logger.error(f"LLM Whisperer extraction failed: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error during LLM Whisperer extraction: {str(e)}")
        return ""

def process_single_pdf(pdf_file, pdf_dir, text_splitter):
    """Process a single PDF file and return the resulting document chunks."""
    try:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        # Extract text using LLM Whisperer
        text_content = extract_text_with_llm_whisperer(pdf_path)
        
        # Check if we got any text content
        if text_content.strip():
            # Create a document with the extracted text
            page_document = Document(page_content=text_content, metadata={"source": pdf_file})
            
            # Split the document into chunks
            chunks = text_splitter.split_documents([page_document])
            logging.info(f"Split {pdf_file} into {len(chunks)} chunks")
            return chunks
        else:
            logging.warning(f"No text content extracted from {pdf_file}")
            return []
                
    except Exception as e:
        logging.error(f"Error processing {pdf_file}: {str(e)}")
        return []

def load_and_chunk_documents_concurrent(pdf_files, pdf_dir, chunk_size, chunk_overlap, max_threads):
    """
    Load PDF documents from a list of files and split them into chunks using multithreading.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    
    # Create a thread-safe progress bar
    progress_lock = Lock()
    progress_counter = {"count": 0, "total": len(pdf_files)}
    
    def update_progress(_):
        with progress_lock:
            progress_counter["count"] += 1
            count = progress_counter["count"]
            total = progress_counter["total"]
            percent = (count / total) * 100
            avg_time = (time.time() - start_time) / max(1, count)
            eta_minutes = (total - count) * avg_time / 60
            print(f"\rProcessing: {percent:.1f}% ({count}/{total}) | ETA: {eta_minutes:.1f}m", end="", flush=True)
    
    # Process files in parallel
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_pdf, pdf_file, pdf_dir, text_splitter): pdf_file 
            for pdf_file in pdf_files
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            pdf_file = future_to_file[future]
            try:
                chunks = future.result()
                documents.extend(chunks)
                update_progress(pdf_file)
            except Exception as e:
                logger.error(f"Unhandled exception processing {pdf_file}: {str(e)}")
    
    # Final newline after progress updates
    print()
    
    # Final logging with total time
    total_time = time.time() - start_time
    logger.info(f"Processed {len(pdf_files)} PDFs in {total_time/60:.2f} minutes.")
    logger.info(f"Total documents after chunking: {len(documents)}")
    
    return documents

def load_and_chunk_documents(pdf_dir, chunk_size, chunk_overlap, batch_size=None, max_threads=1):
    """
    Load PDF documents from a directory and split them into chunks.
    If batch_size is provided, will only load that many PDFs.
    """
    # Check if the PDF directory exists
    if not os.path.exists(pdf_dir):
        logging.error(f"Directory {pdf_dir} does not exist.")
        return []
    
    # Get all PDF files in the directory
    all_pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not all_pdf_files:
        logging.warning(f"No PDF files found in {pdf_dir}.")
        return []
    
    # If batch_size is provided, limit to that many files
    if batch_size and batch_size < len(all_pdf_files):
        pdf_files = all_pdf_files[:batch_size]
        logging.info(f"Processing batch of {len(pdf_files)} PDF files from {len(all_pdf_files)} total.")
    else:
        pdf_files = all_pdf_files
        logging.info(f"Processing all {len(pdf_files)} PDF files.")
    
    # Use concurrent processing if max_threads > 1
    if max_threads > 1:
        return load_and_chunk_documents_concurrent(pdf_files, pdf_dir, chunk_size, chunk_overlap, max_threads)
    
    # Otherwise, use the original sequential processing method
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Use tqdm for progress display with estimated completion time
    start_time = time.time()
    processed_files = 0
    
    progress_bar = tqdm(pdf_files, desc="Processing PDFs")
    for pdf_file in progress_bar:
        try:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            
            # Update progress bar with ETA
            processed_files += 1
            elapsed_time = time.time() - start_time
            if processed_files > 1:  # Only calculate ETA after processing at least one file
                avg_time_per_file = elapsed_time / processed_files
                eta = avg_time_per_file * (len(pdf_files) - processed_files)
                progress_bar.set_postfix({
                    'ETA': f"{eta/60:.1f}m", 
                    'Processed': f"{processed_files}/{len(pdf_files)}"
                })
            
            # Extract text using LLM Whisperer
            text_content = extract_text_with_llm_whisperer(pdf_path)
            
            # Check if we got any text content
            if text_content.strip():
                # Create a document with the extracted text
                page_document = Document(page_content=text_content, metadata={"source": pdf_file})
                
                # Split the document into chunks
                chunks = text_splitter.split_documents([page_document])
                documents.extend(chunks)
                logging.info(f"Split {pdf_file} into {len(chunks)} chunks")
            else:
                logging.warning(f"No text content extracted from {pdf_file}")
                
        except Exception as e:
            logging.error(f"Error processing {pdf_file}: {str(e)}")
    
    # Final logging with total time
    total_time = time.time() - start_time
    logging.info(f"Total documents after chunking: {len(documents)}")
    logging.info(f"Processing completed in {total_time/60:.2f} minutes. Processed {processed_files} files.")
    
    return documents


# --- Step 2: Create Embeddings and Store in ChromaDB ---

def create_chroma_db(documents, model_name, chroma_db_dir, batch_id=None):
    """Creates or updates a ChromaDB with document embeddings."""
    try:
        logger.info(f"Creating embedding function using model: {model_name}")
        embedding_function = HuggingFaceEmbeddings(model_name=model_name) #for use with Langchain
        
        # Create ChromaDB directory if it doesn't exist
        os.makedirs(chroma_db_dir, exist_ok=True)
        logger.info(f"ChromaDB directory: {chroma_db_dir}")
        
        # If processing in batches, use a separate directory for this batch
        if batch_id is not None:
            batch_db_dir = os.path.join(chroma_db_dir, f"batch_{batch_id}")
            os.makedirs(batch_db_dir, exist_ok=True)
            logger.info(f"Batch ChromaDB directory: {batch_db_dir}")
        else:
            batch_db_dir = chroma_db_dir
        
        # Create ChromaDB client and collection
        logger.info(f"Creating ChromaDB from {len(documents)} documents...")
        db = Chroma.from_documents(documents, embedding_function, persist_directory=batch_db_dir)
        logger.info("Persisting ChromaDB to disk...")
        db.persist() #saves to disk.
        logger.info("ChromaDB created/updated successfully.")
        return db #return the db object for later use.

    except Exception as e:
        logger.error(f"Error with ChromaDB: {e}")
        return None

def process_batch(batch_id, pdf_files, pdf_dir, model_name, chroma_db_dir, chunk_size, chunk_overlap, max_threads):
    """Process a batch of PDFs from extraction to ChromaDB storage."""
    logger.info(f"=== Starting Batch {batch_id} with {len(pdf_files)} files ===")
    
    # 1. Load and chunk documents with multithreading
    start_time = time.time()
    documents = load_and_chunk_documents_concurrent(pdf_files, pdf_dir, chunk_size, chunk_overlap, max_threads)
    
    if not documents:
        logger.warning(f"No documents loaded in batch {batch_id}. Skipping.")
        return None
    
    # 2. Create/update ChromaDB for this batch
    db = create_chroma_db(documents, model_name, chroma_db_dir, batch_id)
    
    batch_time = time.time() - start_time
    logger.info(f"=== Batch {batch_id} completed in {batch_time/60:.2f} minutes ===")
    logger.info(f"=== Processed {len(pdf_files)} files into {len(documents)} document chunks ===")
    
    return db

# --- Step 3: (Optional) RAG with Langchain and OpenAI ---

def setup_rag_chain(db):
    """Sets up a RetrievalQA chain for RAG using Langchain and OpenAI."""
    try:
        logger.info("Loading environment variables...")
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return None
        
        logger.info("API key loaded successfully (key not logged for security)")
        logger.info("Initializing OpenAI LLM...")
        llm = OpenAI(temperature=0, openai_api_key=api_key)

        logger.info("Setting up RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" is a good, simple chain type for RAG.
            retriever=db.as_retriever(),
            return_source_documents=True, # Return source documents with answers
        )

        logger.info("RetrievalQA chain created successfully")
        return qa_chain

    except Exception as e:
        logger.error(f"Error setting up RAG chain: {e}")
        return None


# --- Step 4: Querying (either vector search or RAG) ---
def run_query(db, qa_chain=None):
    while True:
        query = input("Enter your query (or 'q' to quit): ")
        if query.lower() == 'q':
            break

        logger.info(f"Running query: {query}")
        
        if qa_chain:  # Use RAG if available
            try:
                logger.info("Using RAG chain for query")
                result = qa_chain({"query": query})
                print("\nAnswer (RAG):")
                print(result["result"])
                print("\nSource Documents:")
                for doc in result["source_documents"]:
                    print(f"  - {doc.metadata['source']}")

            except Exception as e:
                logger.error(f"RAG failed: {e}")
                logger.info("Falling back to Vector Search")
                #Fallback to vector search if RAG failed.
                docs = db.similarity_search_with_score(query)
                print("\nResults (Vector Search):")
                for doc, score in docs:
                    print(f"  - Score: {score:.4f}, Source: {doc.metadata['source']}") #, Page: {doc.metadata['page'] + 1}

        else:  # Use Vector Search
            logger.info("Using Vector Search for query")
            docs = db.similarity_search_with_score(query)
            print("\nResults (Vector Search):")
            for doc, score in docs:
                print(f"  - Score: {score:.4f}, Source: {doc.metadata['source']}") #, Page: {doc.metadata['page'] + 1}

# --- Main Execution ---

if __name__ == "__main__":
    logger.info("=== Starting RAG Analyzer ===")
    
    # Load environment variables
    load_dotenv()
    
    # Check if LLM Whisperer API key is set
    if USE_LLM_WHISPERER and not os.getenv("LLMWHISPERER_API_KEY"):
        logger.warning("LLMWHISPERER_API_KEY not found in environment variables. Using environment variable LLM_WHISPER_API_KEY instead.")
        # Set environment variable for LLM Whisperer from LLM_WHISPER_API_KEY
        os.environ["LLMWHISPERER_API_KEY"] = os.getenv("LLM_WHISPER_API_KEY", "")
        
        if not os.getenv("LLMWHISPERER_API_KEY"):
            logger.error("LLM_WHISPER_API_KEY not found in environment variables. Exiting.")
            print("Error: LLM_WHISPER_API_KEY not found in environment variables. Please set it in the .env file.")
            exit()
    
    # Check for PDF directory
    if not os.path.exists(PDF_DIR):
        logger.error(f"PDF directory does not exist: {PDF_DIR}")
        print(f"Error: PDF directory '{PDF_DIR}' does not exist. Please create it and add PDF files.")
        exit()
    
    # Get all PDF files
    all_pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    if not all_pdf_files:
        logger.error(f"No PDF files found in: {PDF_DIR}")
        print(f"Error: No PDF files found in '{PDF_DIR}'. Please add PDF files.")
        exit()
    
    logger.info(f"Found {len(all_pdf_files)} PDF files in {PDF_DIR}")
    
    # Create batches of files
    num_batches = math.ceil(len(all_pdf_files) / BATCH_SIZE)
    logger.info(f"Processing in {num_batches} batches of up to {BATCH_SIZE} files each")
    
    # Process each batch
    batch_dbs = []
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(all_pdf_files))
        batch_files = all_pdf_files[start_idx:end_idx]
        
        # Process this batch
        batch_db = process_batch(
            batch_id=i+1,
            pdf_files=batch_files,
            pdf_dir=PDF_DIR,
            model_name=MODEL_NAME,
            chroma_db_dir=CHROMA_DB_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            max_threads=MAX_THREADS
        )
        
        if batch_db:
            batch_dbs.append(batch_db)
    
    # Use the latest batch DB for querying (typically the last one)
    if batch_dbs:
        db = batch_dbs[-1]
        
        # 3. (Optional) Setup RAG chain
        logger.info("Setting up RAG chain")
        qa_chain = setup_rag_chain(db)
        
        if not qa_chain:
            logger.warning("RAG chain setup failed. Will use vector search only.")
            print("Note: RAG functionality not available. Using vector search only.")
        
        # 4. Run queries
        logger.info("Starting query interface")
        run_query(db, qa_chain)
    else:
        logger.error("No batches were successfully processed")
        print("Error: No batches were successfully processed. Check logs for details.")
    
    logger.info("=== RAG Analyzer finished ===")