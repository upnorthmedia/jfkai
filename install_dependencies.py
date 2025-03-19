#!/usr/bin/env python
# install_dependencies.py
# Script to install all dependencies for the JFK Files Explorer

import sys
import subprocess
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def install_package(package):
    """Install a package using pip."""
    try:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install {package}")
        return False

def install_spacy_model(model):
    """Install a spaCy model."""
    try:
        logger.info(f"Installing spaCy model {model}...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install spaCy model {model}")
        return False

def main():
    """Main function to install dependencies."""
    logger.info("Starting installation of dependencies for JFK Files Explorer")
    
    # Check if .env file exists and has OpenAI API key
    env_file_exists = os.path.exists(".env")
    if not env_file_exists:
        logger.warning(".env file not found. Creating a template...")
        with open(".env", "w") as f:
            f.write("# OpenAI API Key\n")
            f.write("OPENAI_API_KEY=\n")
        logger.info(".env file created. Please edit it to add your OpenAI API key.")
    else:
        logger.info(".env file found. Please make sure it contains your OpenAI API key.")
    
    # Required packages
    required_packages = [
        "langchain>=0.0.300",
        "langchain-community>=0.0.10",
        "langchain-openai>=0.0.2",
        "chromadb>=0.4.18",
        "openai>=1.10.0",
        "tenacity>=8.2.3",
        "python-dotenv>=1.0.0",
        "streamlit>=1.29.0",
        "tqdm>=4.66.1",
        "spacy>=3.7.2",
        "nltk>=3.8.1",
        "sentence-transformers>=2.2.2",
        "numpy>=1.26.0",
        "pydantic>=2.5.2",
        "tiktoken>=0.5.1"
    ]
    
    # Optional packages for advanced features
    optional_packages = [
        "bertopic>=0.15.0",
        "scikit-learn>=1.3.2"
    ]
    
    # Install required packages
    all_required_succeeded = True
    logger.info("Installing required packages...")
    for package in required_packages:
        if not install_package(package):
            all_required_succeeded = False
    
    # Install optional packages
    logger.info("Installing optional packages for advanced features...")
    for package in optional_packages:
        install_package(package)  # Don't fail if optional packages fail
    
    # Install spaCy models
    try:
        logger.info("Installing spaCy models...")
        import spacy
        # Install small model first (fast download)
        install_spacy_model("en_core_web_sm")
        # Optionally install larger model if enough disk space
        logger.info("Would you like to install the larger spaCy model for better entity recognition? (y/n)")
        response = input().lower()
        if response == 'y':
            install_spacy_model("en_core_web_lg")
    except ImportError:
        logger.error("Failed to import spaCy. Please make sure it's installed correctly.")
    
    # Install NLTK resources
    try:
        logger.info("Downloading NLTK resources...")
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
    except ImportError:
        logger.error("Failed to import NLTK. Please make sure it's installed correctly.")
    
    # Final message
    if all_required_succeeded:
        logger.info("\n============== Installation Complete ==============")
        logger.info("All required dependencies have been installed successfully.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Make sure your OpenAI API key is in the .env file")
        logger.info("2. Test preprocessing with a small sample:")
        logger.info("   python run_preprocessing.py --sample 5 --batch-size 2 --concurrent-tasks 1")
        logger.info("3. Run the enhanced Streamlit app:")
        logger.info("   streamlit run enhanced_app.py")
        logger.info("================================================\n")
        return 0
    else:
        logger.error("\n============== Installation Incomplete ==============")
        logger.error("Some required dependencies could not be installed.")
        logger.error("Please check the error messages above and try again.")
        logger.error("==================================================\n")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 