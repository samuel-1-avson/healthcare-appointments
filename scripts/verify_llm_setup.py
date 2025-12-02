import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def verify_imports():
    logger.info("Verifying LLM imports...")
    
    # Check LLM Routes
    try:
        from api.routes import llm_routes
        logger.info("✅ llm_routes imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import llm_routes: {e}")
        
    # Check RAG Routes
    try:
        from api.routes import rag_routes
        logger.info("✅ rag_routes imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import rag_routes: {e}")

    # Check Dependencies
    try:
        import langchain
        logger.info(f"✅ langchain version: {langchain.__version__}")
    except ImportError:
        logger.error("❌ langchain not installed")

    try:
        import openai
        logger.info(f"✅ openai version: {openai.__version__}")
    except ImportError:
        logger.error("❌ openai not installed")

if __name__ == "__main__":
    verify_imports()
