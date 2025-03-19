#!/usr/bin/env python
# advanced_retrieval.py
# Module implementing advanced retrieval strategies for the JFK Files Explorer

import os
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema.retriever import BaseRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ENHANCED_DB_DIR = "enhanced_chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o"
DEFAULT_TOP_K = 10

# Load environment variables
load_dotenv()

class HybridRetriever:
    """Advanced retriever combining multiple strategies for better document retrieval."""
    
    def __init__(
        self,
        db_dir: str = ENHANCED_DB_DIR,
        hf_model_name: str = MODEL_NAME,
        openai_model: str = OPENAI_EMBEDDING_MODEL,
        use_openai: bool = True,
        gpt_model: str = GPT_MODEL
    ):
        """Initialize the hybrid retriever."""
        self.db_dir = db_dir
        self.hf_model_name = hf_model_name
        self.openai_model = openai_model
        self.use_openai = use_openai and os.getenv("OPENAI_API_KEY") is not None
        self.gpt_model = gpt_model
        
        # Initialize embeddings
        self.initialize_embeddings()
        
        # Load databases
        self.load_databases()
        
        # Initialize LLM for query understanding
        self.initialize_llm()
    
    def initialize_embeddings(self):
        """Initialize embedding functions."""
        logger.info(f"Initializing embedding functions")
        self.hf_embeddings = HuggingFaceEmbeddings(model_name=self.hf_model_name)
        
        if self.use_openai:
            try:
                logger.info(f"Initializing OpenAI embedding function with model {self.openai_model}")
                self.openai_embeddings = OpenAIEmbeddings(
                    model=self.openai_model,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
            except Exception as e:
                logger.error(f"Error initializing OpenAI embeddings: {e}. Falling back to HuggingFace.")
                self.openai_embeddings = None
                self.use_openai = False
        else:
            logger.warning("OpenAI embeddings disabled or API key not found. Using HuggingFace only.")
            self.openai_embeddings = None
    
    def load_databases(self):
        """Load Chroma databases for different types of chunks."""
        self.databases = {}
        
        # Determine which embedding provider to use
        provider = "openai" if self.use_openai and self.openai_embeddings else "huggingface"
        embedding_func = self.openai_embeddings if self.use_openai and self.openai_embeddings else self.hf_embeddings
        
        logger.info(f"Loading databases using {provider} embeddings")
        db_base_dir = os.path.join(self.db_dir, provider)
        
        # Check if enhanced database exists
        if not os.path.exists(db_base_dir):
            logger.warning(f"Enhanced database directory {db_base_dir} not found. Falling back to original database.")
            # If enhanced DB doesn't exist, use the original merged DB
            try:
                self.db_original = Chroma(
                    persist_directory="merged_chroma_db",
                    embedding_function=self.hf_embeddings
                )
                logger.info("Loaded original merged ChromaDB")
                return
            except Exception as e:
                logger.error(f"Error loading original database: {e}")
                return
        
        # Load collections
        for collection_name in ["small_chunks", "large_chunks", "summary_chunks", "full_docs"]:
            collection_path = os.path.join(db_base_dir, collection_name)
            
            if os.path.exists(collection_path):
                try:
                    logger.info(f"Loading {collection_name} from {collection_path}")
                    db = Chroma(
                        persist_directory=collection_path,
                        embedding_function=embedding_func,
                        collection_name=collection_name
                    )
                    self.databases[collection_name] = db
                    logger.info(f"Successfully loaded {collection_name}")
                except Exception as e:
                    logger.error(f"Error loading {collection_name}: {e}")
            else:
                logger.warning(f"Collection {collection_name} not found at {collection_path}")
        
        # Load metadata if available
        metadata_path = os.path.join(self.db_dir, "metadata", "document_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded document metadata from {metadata_path}")
            except Exception as e:
                logger.error(f"Error loading document metadata: {e}")
                self.metadata = None
        else:
            logger.warning(f"Document metadata not found at {metadata_path}")
            self.metadata = None
    
    def initialize_llm(self):
        """Initialize LLM for query understanding and reranking."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found. LLM-based query enhancement will be disabled.")
            self.llm = None
            return
        
        try:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=self.gpt_model,
                temperature=0,
                max_tokens=1000
            )
            logger.info(f"Initialized LLM with model {self.gpt_model}")
            
            # Create query enhancement chain
            query_enhance_prompt = ChatPromptTemplate.from_template("""
            You are an expert in analyzing intelligence documents.
            
            Based on the following query, generate two outputs:
            1. An enhanced version of the query that is more specific and includes relevant terms that might appear in intelligence documents
            2. A list of 3-5 key entities or terms that are likely to be mentioned in relevant documents
            
            Original Query: {query}
            
            Enhanced Query:
            """)
            
            self.query_enhancement_chain = query_enhance_prompt | self.llm | StrOutputParser()
            
            # Setup contextual compression for relevant information extraction
            self.contextual_compressor = LLMChainExtractor.from_llm(self.llm)
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.llm = None
    
    def enhance_query(self, query: str) -> str:
        """Enhance the query with LLM to improve retrieval."""
        if not self.llm:
            return query
        
        try:
            enhanced_query = self.query_enhancement_chain.invoke({"query": query})
            logger.info(f"Enhanced query: {enhanced_query}")
            return enhanced_query
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, use_all_retrievers: bool = True) -> Dict[str, Any]:
        """Retrieve documents using a hybrid approach."""
        start_time = time.time()
        
        # Check if we have databases loaded
        if not hasattr(self, "databases") or not self.databases:
            logger.warning("No enhanced databases available. Falling back to original if available.")
            if hasattr(self, "db_original") and self.db_original:
                try:
                    results = self.db_original.similarity_search_with_score(query, k=top_k)
                    docs = [doc for doc, _ in results]
                    return {
                        "documents": docs,
                        "retrieval_time": time.time() - start_time,
                        "enhanced_query": query,
                        "retriever_used": "original"
                    }
                except Exception as e:
                    logger.error(f"Error retrieving from original database: {e}")
                    return {
                        "documents": [],
                        "retrieval_time": time.time() - start_time,
                        "error": str(e)
                    }
            else:
                logger.error("No databases available for retrieval")
                return {
                    "documents": [],
                    "retrieval_time": time.time() - start_time,
                    "error": "No databases available for retrieval"
                }
        
        # Enhance query if LLM is available
        enhanced_query = self.enhance_query(query)
        
        # Multi-stage retrieval
        all_docs = []
        retrieval_metadata = {}
        
        # Retrieve from summary chunks first (high precision)
        if "summary_chunks" in self.databases and use_all_retrievers:
            try:
                logger.info(f"Retrieving from summary chunks with k={top_k}")
                summary_results = self.databases["summary_chunks"].similarity_search_with_score(
                    enhanced_query, k=top_k
                )
                
                summary_docs = [doc for doc, score in summary_results]
                summary_scores = [score for _, score in summary_results]
                
                retrieval_metadata["summary_retrieval"] = {
                    "count": len(summary_docs),
                    "avg_score": sum(summary_scores) / len(summary_scores) if summary_scores else 0
                }
                
                all_docs.extend(summary_docs)
                logger.info(f"Retrieved {len(summary_docs)} documents from summary chunks")
            except Exception as e:
                logger.error(f"Error retrieving from summary chunks: {e}")
        
        # Retrieve from small chunks (high precision for specific facts)
        if "small_chunks" in self.databases:
            try:
                logger.info(f"Retrieving from small chunks with k={top_k}")
                small_results = self.databases["small_chunks"].similarity_search_with_score(
                    enhanced_query, k=top_k
                )
                
                small_docs = [doc for doc, score in small_results]
                small_scores = [score for _, score in small_results]
                
                retrieval_metadata["small_chunk_retrieval"] = {
                    "count": len(small_docs),
                    "avg_score": sum(small_scores) / len(small_scores) if small_scores else 0
                }
                
                all_docs.extend(small_docs)
                logger.info(f"Retrieved {len(small_docs)} documents from small chunks")
            except Exception as e:
                logger.error(f"Error retrieving from small chunks: {e}")
        
        # Retrieve from large chunks (more context)
        if "large_chunks" in self.databases and use_all_retrievers:
            try:
                logger.info(f"Retrieving from large chunks with k={top_k // 2}")
                large_results = self.databases["large_chunks"].similarity_search_with_score(
                    enhanced_query, k=top_k // 2  # Using fewer large chunks as they contain more context
                )
                
                large_docs = [doc for doc, score in large_results]
                large_scores = [score for _, score in large_results]
                
                retrieval_metadata["large_chunk_retrieval"] = {
                    "count": len(large_docs),
                    "avg_score": sum(large_scores) / len(large_scores) if large_scores else 0
                }
                
                all_docs.extend(large_docs)
                logger.info(f"Retrieved {len(large_docs)} documents from large chunks")
            except Exception as e:
                logger.error(f"Error retrieving from large chunks: {e}")
        
        # Apply contextual compression if LLM is available
        filtered_docs = all_docs
        if self.llm and self.contextual_compressor and use_all_retrievers:
            try:
                logger.info("Applying contextual compression to retrieved documents")
                # Create a proper retriever object to use with ContextualCompressionRetriever
                
                class SimpleRetriever(BaseRetriever):
                    """A simple retriever that returns the provided documents."""
                    
                    def __init__(self, documents):
                        self._documents = documents
                        super().__init__()
                        
                    def _get_relevant_documents(self, query):
                        return self._documents
                
                simple_retriever = SimpleRetriever(all_docs)
                
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=self.contextual_compressor,
                    base_retriever=simple_retriever
                )
                
                filtered_docs = compression_retriever.get_relevant_documents(query)
                logger.info(f"Compressed {len(all_docs)} documents to {len(filtered_docs)} relevant ones")
                
                retrieval_metadata["compression"] = {
                    "original_count": len(all_docs),
                    "compressed_count": len(filtered_docs)
                }
            except Exception as e:
                logger.error(f"Error applying contextual compression: {e}")
        
        # Remove duplicate chunks based on parent_id
        seen_parents = set()
        deduplicated_docs = []
        
        for doc in filtered_docs:
            parent_id = doc.metadata.get("parent_id", None)
            chunk_id = doc.metadata.get("chunk_id", None)
            
            # If we have a chunk_id and haven't seen the parent document, add it
            if chunk_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                deduplicated_docs.append(doc)
            # If we don't have a chunk_id or parent_id, just add the document
            elif not chunk_id:
                deduplicated_docs.append(doc)
        
        retrieval_time = time.time() - start_time
        logger.info(f"Total retrieval time: {retrieval_time:.2f} seconds")
        
        return {
            "documents": deduplicated_docs[:top_k],  # Return only top_k documents
            "full_results": filtered_docs,  # Include all results for reference
            "retrieval_time": retrieval_time,
            "enhanced_query": enhanced_query,
            "retrieval_metadata": retrieval_metadata
        }
    
    def retrieve_and_generate(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """Retrieve documents and generate a response using a RAG approach."""
        if not self.llm:
            logger.error("LLM not available for response generation")
            return {
                "result": "Error: LLM not available for response generation",
                "documents": []
            }
        
        # Retrieve documents
        retrieval_result = self.retrieve(query, top_k=top_k)
        documents = retrieval_result.get("documents", [])
        
        if not documents:
            return {
                "result": "No relevant documents found",
                "documents": [],
                "retrieval_metadata": retrieval_result.get("retrieval_metadata", {}),
                "enhanced_query": retrieval_result.get("enhanced_query", query)
            }
        
        # Format documents for context
        formatted_docs = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Unknown')
            formatted_docs.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}\n")
        
        context = "\n".join(formatted_docs)
        
        # Create RAG prompt
        rag_prompt = ChatPromptTemplate.from_template("""
        You are an expert researcher analyzing declassified JFK assassination files. Your goal is to provide insights based on the actual document content.
        
        CONTEXT INFORMATION:
        {context}
        
        USER QUESTION: {question}
        
        INSTRUCTIONS:
        1. Answer based ONLY on the provided context - be specific and detailed
        2. If you can't find a direct answer in the context, analyze what might be implied or suggested
        3. Always cite which specific documents you're using (include document numbers)
        4. Be factual and objective - these are historical government documents
        5. Note any inconsistencies or uncertainties in the documents
        6. IMPORTANT: Never say "I don't know" - instead, explain what information is present in the documents and what would be needed to answer more fully
        7. Focus on helping the researcher understand these historical documents
        8. Look for connections between different documents that might provide deeper insights
        
        YOUR ANSWER:
        """)
        
        # Define the RAG chain
        try:
            # Generate response with a properly constructed chain
            logger.info("Generating RAG response")
            
            # Create a simpler chain using direct function calls
            inputs = {"context": context, "question": query}
            prompt_value = rag_prompt.invoke(inputs)
            llm_response = self.llm.invoke(prompt_value)
            result = StrOutputParser().invoke(llm_response)
            
            return {
                "result": result,
                "documents": documents,
                "retrieval_metadata": retrieval_result.get("retrieval_metadata", {}),
                "enhanced_query": retrieval_result.get("enhanced_query", query)
            }
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "result": f"Error generating response: {str(e)}",
                "documents": documents,
                "retrieval_metadata": retrieval_result.get("retrieval_metadata", {}),
                "enhanced_query": retrieval_result.get("enhanced_query", query)
            }

# Simple test function
def test_retrieval():
    """Test the hybrid retriever with a sample query."""
    retriever = HybridRetriever()
    
    test_query = "What documents link Oswald to Cuba?"
    
    # Test basic retrieval
    logger.info(f"Testing retrieval with query: {test_query}")
    result = retriever.retrieve(test_query)
    
    logger.info(f"Retrieved {len(result['documents'])} documents in {result['retrieval_time']:.2f} seconds")
    for i, doc in enumerate(result['documents'][:3]):  # Show first 3 docs
        logger.info(f"Document {i+1}:")
        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"Content: {doc.page_content[:200]}...")
    
    # Test RAG
    logger.info("Testing RAG response generation")
    rag_result = retriever.retrieve_and_generate(test_query)
    
    logger.info(f"RAG Response: {rag_result['result']}")
    
    return rag_result

if __name__ == "__main__":
    test_retrieval() 