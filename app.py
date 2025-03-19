import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
import time
import threading
from dotenv import load_dotenv
import warnings
import torch  # Add torch import to check version
import sys  # Add sys import to check Python version
import asyncio

# Configure caching and performance parameters
CHROMA_DB_DIR = "merged_chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_CACHE_TIME = 3600  # 1 hour
RETRIEVAL_K = 8  # Number of documents to retrieve
LLM_TIMEOUT = 60  # Seconds before timing out LLM request
MAX_CONCURRENT_USERS = 100  # Adjust based on load testing
DEFAULT_MODEL = "gpt-4o"

# Print diagnostic information
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch path: {torch.__file__}")

# Fix for asyncio-related issues
try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception as e:
    print(f"Error setting asyncio policy: {e}")

# Suppress HuggingFace tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# More comprehensive workaround for PyTorch/Streamlit file watcher incompatibility
# This patch prevents Streamlit from watching torch modules that cause issues
try:
    import streamlit.watcher.local_sources_watcher as lsw
    original_get_module_paths = lsw.get_module_paths
    
    def patched_get_module_paths(module):
        # Skip problematic torch modules
        if hasattr(module, "__name__") and (
            module.__name__ == "torch._classes" or 
            module.__name__.startswith("torch._C") or
            module.__name__.startswith("torch.classes") or
            "._jit_internal" in module.__name__
        ):
            print(f"Skipping module path extraction for {module.__name__}")
            return []
        
        # Handle potential errors in other modules
        try:
            return original_get_module_paths(module)
        except Exception as e:
            print(f"Error extracting module paths for {module.__name__ if hasattr(module, '__name__') else 'unknown'}: {e}")
            return []
    
    lsw.get_module_paths = patched_get_module_paths
    print("Applied comprehensive patch to streamlit.watcher.local_sources_watcher.get_module_paths")
except Exception as e:
    print(f"Failed to apply torch patch: {str(e)}")

# Set page configuration
st.set_page_config(
    page_title="JFK Files Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run a health check at app startup
def run_health_check():
    """Check for common issues at app startup and display warnings if needed"""
    issues = []
    
    # Check if PyTorch is properly initialized
    try:
        # Test a basic PyTorch operation
        import torch
        tensor = torch.zeros(1)
        print(f"PyTorch test successful: {tensor}")
    except Exception as e:
        issues.append(f"PyTorch initialization issue: {str(e)}")
    
    # Check if ChromaDB directory exists
    if not os.path.exists(CHROMA_DB_DIR):
        issues.append(f"ChromaDB directory not found at {CHROMA_DB_DIR}")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OpenAI API key not found in environment variables")
    
    # Display issues if any
    if issues:
        st.sidebar.warning("⚠️ Application Health Check Issues:")
        for issue in issues:
            st.sidebar.warning(f"- {issue}")
        print("Health check detected issues:", issues)
    
    return len(issues) == 0

# Run health check
app_healthy = run_health_check()

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables with default values"""
    if "concurrent_users" not in st.session_state:
        st.session_state.concurrent_users = 0
        
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = DEFAULT_MODEL
        
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = RETRIEVAL_K
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Call initialization early in the script
init_session_state()

# Cache the ChromaDB connection
@st.cache_resource
def load_database():
    try:
        print("Attempting to load HuggingFace embeddings...")
        embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        print(f"Embedding function created successfully with model: {MODEL_NAME}")
        
        print(f"Attempting to connect to ChromaDB at: {CHROMA_DB_DIR}")
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
        print("ChromaDB connected successfully")
        return db
    except Exception as e:
        print(f"ERROR loading database: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        st.error(f"Failed to load database: {str(e)}")
        return None

# Cache query results
@st.cache_data(ttl=EMBEDDING_CACHE_TIME)
def query_database(query_text, k=RETRIEVAL_K):
    db = load_database()
    if db is None:
        return [], 0
    
    start = time.time()
    try:
        results = db.similarity_search_with_score(query_text, k=k)
        end = time.time()
        return results, end-start
    except Exception as e:
        print(f"ERROR in query_database: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return [], 0

# Format documents for prompt context
def format_docs(docs):
    return "\n\n".join(f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)

# Setup RAG chain with timeout
def get_rag_response(query, db, timeout=LLM_TIMEOUT):
    # Initialize OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return {
            "result": "Error: OpenAI API key not found. Please check environment variables.",
            "source_documents": []
        }
    
    if db is None:
        return {
            "result": "Error: Database connection failed. Please check logs for details.",
            "source_documents": []
        }
    
    # Get retrieval_k value from session state before thread starts
    retrieval_k = st.session_state.retrieval_k if "retrieval_k" in st.session_state else RETRIEVAL_K
    
    # Create the prompt template with advanced instructions
    prompt_template = """You are an expert researcher analyzing declassified JFK assassination files. Your goal is to provide insights based on the actual document content.
    
CONTEXT INFORMATION:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context - be specific and detailed
2. If you can't find a direct answer in the context, analyze what might be implied or suggested
3. Always cite which specific documents you're using (file names)
4. Be factual and objective but also think outside the box - these are historical government documents
5. Note any inconsistencies or uncertainties in the documents
6. IMPORTANT: Never say "I don't know" - instead, explain what information is present in the documents and what would be needed to answer more fully
7. Focus on helping the researcher understand these historical documents
8. Look for connections between the documents and the question

YOUR ANSWER:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Use the model selected in the UI - get before thread starts
    selected_model = st.session_state.model_choice if "model_choice" in st.session_state else "gpt-4o"
    
    # Use the new OpenAI chat model with the selected model
    llm = ChatOpenAI(
        model=selected_model,
        temperature=0.4, 
        api_key=openai_api_key,
        max_tokens=4000
    )
    
    try:
        # Create the retrieval chain using the newer LangChain pattern
        retriever = db.as_retriever(search_kwargs={"k": retrieval_k})
        
        # Define the RAG chain using LCEL
        chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    except Exception as e:
        print(f"ERROR setting up RAG chain: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            "result": f"Error setting up RAG chain: {str(e)}",
            "source_documents": []
        }
    
    result = {"result": "Request timed out", "source_documents": []}
    
    def process():
        nonlocal result, retrieval_k
        try:
            print(f"Thread started: Processing query in thread: {query[:50]}...")
            
            # Run the chain
            answer = chain.invoke(query)
            print("Chain invocation completed successfully")
            
            # Retrieve the documents separately for display
            print(f"Retrieving source documents with k={retrieval_k}...")
            docs = db.similarity_search(query, k=retrieval_k)
            print(f"Retrieved {len(docs)} source documents")
            
            result = {
                "result": answer,
                "source_documents": docs
            }
        except Exception as e:
            print(f"ERROR in RAG thread: {type(e).__name__}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            result = {"result": f"Error: {str(e)}", "source_documents": []}
    
    # Run with timeout
    print(f"Starting RAG processing thread with timeout {timeout}s")
    try:
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            print("Thread timed out")
    except Exception as e:
        print(f"ERROR starting/joining thread: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        result = {"result": f"Error with thread management: {str(e)}", "source_documents": []}
    
    print("Returning RAG result")
    return result

# Add model selector
def get_model_options():
    return {
        "gpt-4o": "GPT-4o (Recommended)",
        "gpt-3.5-turbo": "GPT-3.5 Turbo (Faster)",
    }

# Track concurrent users
if "concurrent_users" not in st.session_state:
    st.session_state.concurrent_users = 0

# Streamlit UI
st.title("JFK Files Explorer")
st.markdown("Ask questions about the JFK assassination files")

# Sidebar for options
with st.sidebar:
    st.header("Options")
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(get_model_options().keys()),
        format_func=lambda x: get_model_options()[x],
        index=list(get_model_options().keys()).index(st.session_state.model_choice),
    )
    st.session_state.model_choice = selected_model
    
    st.session_state.retrieval_k = st.slider(
        "Number of documents to retrieve", 
        min_value=3, 
        max_value=15, 
        value=st.session_state.retrieval_k
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """This app allows you to search through declassified JFK assassination files 
        using AI-powered semantic search and retrieval."""
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
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
                    # First get vector search results for direct display
                    db = load_database()
                    
                    # Check if database connection was successful
                    if db is None:
                        st.error("Database connection failed. Please check server logs for details.")
                        answer_text = "Error: Unable to connect to the JFK files database."
                    else:
                        # Fallback to direct vector search if RAG fails
                        try:
                            # Try RAG with LLM
                            response = get_rag_response(prompt, db)
                            st.markdown(response["result"])
                            
                            # Display sources
                            if response["source_documents"]:
                                st.markdown("#### Sources:")
                                sources = []
                                for doc in response["source_documents"]:
                                    source = doc.metadata.get("source", "Unknown")
                                    if source not in sources:
                                        sources.append(source)
                                
                                for i, source in enumerate(sources[:5]):
                                    st.markdown(f"{i+1}. {source}")
                            else:
                                st.markdown("*No source documents were found*")
                                
                            answer_text = response["result"]
                        except Exception as e:
                            # Fallback to vector search
                            st.error(f"RAG query failed: {str(e)}. Falling back to vector search.")
                            try:
                                results, search_time = query_database(prompt)
                                
                                if results:
                                    st.markdown("#### Top Relevant Documents:")
                                    for i, (doc, score) in enumerate(results[:5]):
                                        st.markdown(f"**{i+1}. {doc.metadata.get('source', 'Unknown')}** (Score: {score:.4f})")
                                        st.markdown(doc.page_content[:500] + "...")
                                else:
                                    st.markdown("*No relevant documents found*")
                                
                                answer_text = "Search results displayed directly due to RAG error."
                            except Exception as inner_e:
                                st.error(f"Vector search also failed: {str(inner_e)}")
                                answer_text = f"Error: Both RAG and vector search failed. Please try again later."
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
        
        finally:
            # Release the user count
            st.session_state.concurrent_users -= 1

# Add footer
st.markdown("""
---
*This application provides access to declassified documents related to the JFK assassination.*
""") 