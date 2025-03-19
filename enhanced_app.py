import streamlit as st
from dotenv import load_dotenv
import os
import time
import warnings
import threading
import sys
import json
from pathlib import Path

# Import our advanced retrieval system
from advanced_retrieval import HybridRetriever, DEFAULT_TOP_K

# Load environment variables and configure settings
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants and configuration
ENHANCED_DB_DIR = "enhanced_chroma_db"
DEFAULT_MODEL = "gpt-4o"
LLM_TIMEOUT = 120  # Increased timeout for more comprehensive responses
MAX_CONCURRENT_USERS = 50

# Set page configuration
st.set_page_config(
    page_title="Enhanced JFK Files Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables with default values"""
    if "concurrent_users" not in st.session_state:
        st.session_state.concurrent_users = 0
        
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = DEFAULT_MODEL
        
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = DEFAULT_TOP_K
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "use_advanced_retrieval" not in st.session_state:
        st.session_state.use_advanced_retrieval = True
        
    if "use_hierarchical_retrieval" not in st.session_state:
        st.session_state.use_hierarchical_retrieval = True
        
    if "show_metadata" not in st.session_state:
        st.session_state.show_metadata = False
        
    if "show_enhanced_query" not in st.session_state:
        st.session_state.show_enhanced_query = False
        
    if "advanced_options_expanded" not in st.session_state:
        st.session_state.advanced_options_expanded = False
        
    if "last_retrieval_stats" not in st.session_state:
        st.session_state.last_retrieval_stats = None

# Call initialization early
init_session_state()

# Cache the HybridRetriever instance
@st.cache_resource
def get_hybrid_retriever():
    """Get or create a hybrid retriever instance"""
    # Check if enhanced database exists, otherwise use original
    if os.path.exists(ENHANCED_DB_DIR):
        retriever = HybridRetriever(
            db_dir=ENHANCED_DB_DIR,
            use_openai=True if os.getenv("OPENAI_API_KEY") else False,
            gpt_model=st.session_state.model_choice
        )
    else:
        # Fallback to original database
        retriever = HybridRetriever(
            db_dir="merged_chroma_db",
            use_openai=False,  # Use HuggingFace with original DB
            gpt_model=st.session_state.model_choice
        )
    
    return retriever

# Get model options
def get_model_options():
    return {
        "gpt-4o": "GPT-4o (Recommended)",
        "gpt-3.5-turbo": "GPT-3.5 Turbo (Faster)",
        "gpt-4o-mini": "GPT-4o-mini (Balanced)",
    }

# Streamlit UI
st.title("Enhanced JFK Files Explorer 🔍")
st.markdown("Ask questions about the JFK assassination files with our advanced document analysis system")

# Sidebar for options
with st.sidebar:
    st.header("Options")
    
    # Model selection
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(get_model_options().keys()),
        format_func=lambda x: get_model_options()[x],
        index=list(get_model_options().keys()).index(st.session_state.model_choice),
    )
    st.session_state.model_choice = selected_model
    
    # Basic retrieval settings
    st.session_state.retrieval_k = st.slider(
        "Number of documents to retrieve", 
        min_value=5, 
        max_value=30, 
        value=st.session_state.retrieval_k
    )
    
    # Advanced options in an expandable section
    with st.expander("Advanced Settings", expanded=st.session_state.advanced_options_expanded):
        st.session_state.advanced_options_expanded = True
        
        st.session_state.use_advanced_retrieval = st.checkbox(
            "Use Advanced Retrieval", 
            value=st.session_state.use_advanced_retrieval,
            help="Use our advanced retrieval system with LLM-enhanced queries"
        )
        
        st.session_state.use_hierarchical_retrieval = st.checkbox(
            "Use Hierarchical Retrieval", 
            value=st.session_state.use_hierarchical_retrieval,
            help="Combine different document chunk sizes for better context"
        )
        
        st.session_state.show_metadata = st.checkbox(
            "Show Document Metadata", 
            value=st.session_state.show_metadata,
            help="Display additional document metadata in results"
        )
        
        st.session_state.show_enhanced_query = st.checkbox(
            "Show Enhanced Queries", 
            value=st.session_state.show_enhanced_query,
            help="Display the LLM-enhanced versions of your queries"
        )
    
    # Database info section
    st.markdown("---")
    
    # Check database status
    if os.path.exists(ENHANCED_DB_DIR):
        st.success("✅ Using Enhanced Document Database")
        
        # Show database stats if available
        metadata_path = os.path.join(ENHANCED_DB_DIR, "metadata", "document_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                st.write(f"📊 {len(metadata)} Enhanced Documents")
            except:
                st.write("📊 Enhanced Database Available")
    else:
        st.warning("⚠️ Using Original Database (Limited Features)")
        st.info("Run preprocessing script to enable enhanced features")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """This enhanced app provides powerful search and analysis of declassified JFK 
        assassination files using advanced document processing, hierarchical chunking, 
        and multi-strategy retrieval."""
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display metadata if present and enabled
        if st.session_state.show_metadata and message.get("metadata"):
            with st.expander("Query Details"):
                st.json(message["metadata"])

# User input and response
if prompt := st.chat_input("Ask about the JFK files"):
    # Check concurrent user limit
    if st.session_state.concurrent_users >= MAX_CONCURRENT_USERS:
        st.error("Server is experiencing high load. Please try again in a few minutes.")
    else:
        # Track this user session
        st.session_state.concurrent_users += 1
        
        try:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching through declassified files..."):
                    # Get retriever
                    retriever = get_hybrid_retriever()
                    
                    # Update the retriever's model if needed
                    if hasattr(retriever, "llm") and retriever.llm and st.session_state.model_choice != retriever.gpt_model:
                        try:
                            retriever.llm.model = st.session_state.model_choice
                            retriever.gpt_model = st.session_state.model_choice
                        except:
                            pass  # If model update fails, continue with existing model
                    
                    response_metadata = {}
                    
                    if st.session_state.use_advanced_retrieval:
                        # Use the advanced RAG system
                        try:
                            # Process with the hybrid retriever
                            result = retriever.retrieve_and_generate(
                                query=prompt,
                                top_k=st.session_state.retrieval_k
                            )
                            
                            answer_text = result.get("result", "")
                            source_documents = result.get("documents", [])
                            
                            # Store metadata for display
                            response_metadata = {
                                "retrieval_time": result.get("retrieval_time", 0),
                                "enhanced_query": result.get("enhanced_query", ""),
                                "source_count": len(source_documents),
                                "retrieval_metadata": result.get("retrieval_metadata", {})
                            }
                            
                            # Display the enhanced query if enabled
                            if st.session_state.show_enhanced_query and "enhanced_query" in result:
                                st.info(f"**Enhanced Query:** {result['enhanced_query']}")
                            
                            # Display the answer
                            st.markdown(answer_text)
                            
                            # Display sources
                            if source_documents:
                                st.markdown("#### Sources:")
                                sources = []
                                for doc in source_documents:
                                    source = doc.metadata.get("source", "Unknown")
                                    if source not in sources:
                                        sources.append(source)
                                
                                for i, source in enumerate(sources[:7]):  # Show up to 7 sources
                                    st.markdown(f"{i+1}. {source}")
                                
                                # Show document metadata if enabled
                                if st.session_state.show_metadata:
                                    with st.expander("Document Details"):
                                        for i, doc in enumerate(source_documents[:3]):  # Show first 3 docs
                                            st.markdown(f"**Document {i+1}**")
                                            st.markdown(f"Source: {doc.metadata.get('source', 'Unknown')}")
                                            
                                            # Show enhanced metadata if available
                                            if "document_type" in doc.metadata:
                                                st.markdown(f"Type: {doc.metadata.get('document_type', 'Unknown')}")
                                            
                                            if "importance_score" in doc.metadata:
                                                st.markdown(f"Importance: {doc.metadata.get('importance_score', 'N/A')}/10")
                                            
                                            if "summary" in doc.metadata:
                                                st.markdown(f"Summary: {doc.metadata.get('summary', 'N/A')}")
                                            
                                            if "people_mentioned" in doc.metadata and doc.metadata["people_mentioned"]:
                                                st.markdown(f"People: {', '.join(doc.metadata['people_mentioned'][:5])}")
                                            
                                            if "organizations_mentioned" in doc.metadata and doc.metadata["organizations_mentioned"]:
                                                st.markdown(f"Organizations: {', '.join(doc.metadata['organizations_mentioned'][:5])}")
                                            
                                            st.markdown("---")
                            else:
                                st.markdown("*No source documents were found*")
                        
                        except Exception as e:
                            st.error(f"Advanced retrieval error: {str(e)}")
                            answer_text = f"Error: Advanced retrieval failed. Please try again or switch to basic retrieval in the sidebar options."
                            response_metadata = {"error": str(e)}
                    
                    else:
                        # Fallback to basic retrieval
                        try:
                            # Use basic retrieve method
                            result = retriever.retrieve(
                                query=prompt,
                                top_k=st.session_state.retrieval_k,
                                use_all_retrievers=st.session_state.use_hierarchical_retrieval
                            )
                            
                            documents = result.get("documents", [])
                            
                            # Display basic results
                            st.markdown("#### Top Relevant Documents:")
                            
                            if documents:
                                for i, doc in enumerate(documents[:5]):
                                    st.markdown(f"**{i+1}. {doc.metadata.get('source', 'Unknown')}**")
                                    st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                    st.markdown("---")
                            else:
                                st.markdown("*No relevant documents found*")
                            
                            answer_text = "Document search results displayed."
                            response_metadata = {
                                "retrieval_time": result.get("retrieval_time", 0),
                                "enhanced_query": result.get("enhanced_query", ""),
                                "source_count": len(documents)
                            }
                            
                        except Exception as e:
                            st.error(f"Retrieval error: {str(e)}")
                            answer_text = f"Error: Document retrieval failed. Please try again."
                            response_metadata = {"error": str(e)}
                
                # Store the answer and metadata in chat history
                message_data = {
                    "role": "assistant", 
                    "content": answer_text,
                    "metadata": response_metadata
                }
                st.session_state.messages.append(message_data)
                
                # Store retrieval stats for analysis
                st.session_state.last_retrieval_stats = response_metadata
        
        finally:
            # Release the user count
            st.session_state.concurrent_users -= 1

# Add footer with enhanced features info
st.markdown("""
---
*This enhanced application provides access to declassified documents related to the JFK assassination, utilizing advanced document processing, hierarchical chunking, and LLM-enhanced retrieval.*
""") 