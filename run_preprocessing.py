#!/usr/bin/env python
# run_preprocessing.py
# Script to run the preprocessing pipeline

import argparse
import os
import logging
import sys
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if prerequisites are met."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OpenAI API key not found. Some features will be limited.")
    
    # Check if original database exists
    if not os.path.exists("merged_chroma_db"):
        logger.error("Original merged_chroma_db not found. Please run merge_chroma_batches.py first.")
        return False
    
    # Check for required packages
    try:
        import preprocess_chroma
        logger.info("Preprocessing module found.")
        
        # Check for tenacity package
        try:
            import tenacity
            logger.info("Rate limiting dependencies found.")
        except ImportError:
            logger.warning("Rate limiting dependencies (tenacity) not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tenacity"])
            logger.info("Successfully installed tenacity.")
            
    except ImportError:
        logger.error("preprocess_chroma.py not found. Please make sure it's in the current directory.")
        return False
    
    return True

def main():
    """Main function to run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline for JFK Files Explorer")
    parser.add_argument("--source", default="merged_chroma_db", help="Source Chroma DB directory")
    parser.add_argument("--target", default="enhanced_chroma_db", help="Target directory for enhanced DB")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of documents to process in each batch")
    parser.add_argument("--skip-check", action="store_true", help="Skip prerequisites check")
    parser.add_argument("--force", action="store_true", help="Force overwrite if target directory exists")
    parser.add_argument("--sample", type=int, default=0, help="Process only a sample of documents (for testing)")
    parser.add_argument("--max-batches", type=int, default=0, help="Maximum number of batches to process (0 for all)")
    # Add rate limiting parameters
    parser.add_argument("--concurrent-tasks", type=int, default=2, 
                        help="Number of concurrent tasks (default: 2)")
    parser.add_argument("--batch-delay", type=float, default=3,
                        help="Delay between batches in seconds (default: 3)")
    # Add skip processed option
    parser.add_argument("--skip-processed", action="store_true", 
                        help="Skip documents that have already been processed (useful for resuming)")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check prerequisites unless skipped
    if not args.skip_check and not check_prerequisites():
        logger.error("Prerequisites check failed. Exiting.")
        return 1
    
    # Check if target directory exists
    if os.path.exists(args.target) and not args.force and not args.skip_processed:
        logger.error(f"Target directory {args.target} already exists. Use --force to overwrite or --skip-processed to add only new documents.")
        return 1
    
    # Import the preprocessing module
    try:
        from preprocess_chroma import process_existing_chroma
    except ImportError as e:
        logger.error(f"Failed to import preprocess_chroma module: {e}")
        return 1
    
    # Display a warning about rate limits when processing large batches
    if args.batch_size > 5 and args.concurrent_tasks > 1 and args.batch_delay < 2:
        logger.warning("""
===================== RATE LIMIT WARNING =====================
You are processing relatively large batches with high concurrency and low delay.
This may cause OpenAI API rate limit errors.

Consider:
1. Reducing batch size (--batch-size)
2. Reducing concurrent tasks (--concurrent-tasks)
3. Increasing delay between batches (--batch-delay)

For testing, use '--sample 5 --batch-size 2 --concurrent-tasks 1' 
to process a small sample safely.
==============================================================
        """)
        
        # Ask for confirmation if not in a script
        if sys.stdin.isatty():  # Check if running in an interactive terminal
            confirmation = input("Do you want to continue with these settings? (y/n): ")
            if confirmation.lower() != 'y':
                logger.info("Operation cancelled by user.")
                return 0
    
    # Start preprocessing
    logger.info("Starting preprocessing pipeline...")
    start_time = time.time()
    
    try:
        # Create processing options
        process_opts = {
            "source_db_dir": args.source,
            "target_db_dir": args.target,
            "batch_size": args.batch_size
        }
        
        # Add sample size if specified
        if args.sample > 0:
            process_opts["sample_size"] = args.sample
            logger.info(f"Processing sample of {args.sample} documents")
        
        # Add max batches if specified
        if args.max_batches > 0:
            process_opts["max_batches"] = args.max_batches
            logger.info(f"Processing maximum of {args.max_batches} batches")
        
        # Add rate limiting options
        process_opts["concurrent_tasks"] = args.concurrent_tasks
        process_opts["batch_delay"] = args.batch_delay
        logger.info(f"Using {args.concurrent_tasks} concurrent tasks with {args.batch_delay}s delay between batches")
        
        # Add skip processed option
        if args.skip_processed:
            process_opts["skip_processed"] = True
            logger.info("Skipping already processed documents")
        
        result = process_existing_chroma(**process_opts)
        
        if result["status"] == "success":
            elapsed_time = time.time() - start_time
            logger.info(f"Preprocessing completed successfully in {elapsed_time:.2f} seconds.")
            logger.info(f"Processed {result['document_count']} documents.")
            logger.info(f"Created {result.get('small_chunks_count', 0)} small chunks.")
            logger.info(f"Created {result.get('large_chunks_count', 0)} large chunks.")
            logger.info(f"Created {result.get('summary_chunks_count', 0)} summary chunks.")
            logger.info(f"Enhanced database is available at {args.target}")
            
            # Print next steps
            print("\n===== NEXT STEPS =====")
            print(f"1. Run the enhanced app with: streamlit run enhanced_app.py")
            print(f"2. Or test the retrieval with: python advanced_retrieval.py")
            print("====================\n")
            
            return 0
        else:
            logger.error(f"Preprocessing failed: {result.get('error', 'Unknown error')}")
            return 1
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 