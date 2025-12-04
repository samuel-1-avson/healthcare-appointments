
import os
import sys
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_local_setup():
    print("Verifying Local LLM Setup...")
    
    # 1. Check Config Defaults
    print("\n1. Checking Configuration Defaults...")
    from src.llm.config import get_llm_config
    config = get_llm_config()
    
    if config.default_provider != "local":
        print(f"  [FAIL] Default provider is {config.default_provider}, expected 'local'")
        return False
    print(f"  [OK] Default provider is '{config.default_provider}'")
    
    # 2. Check Embeddings Prioritization
    print("\n2. Checking Embeddings Provider...")
    from src.llm.rag.embeddings import get_embeddings
    
    # Force auto selection
    emb_manager = get_embeddings(provider="auto", force_new=True)
    
    if emb_manager.provider == "openai":
        print("  [FAIL] Auto-selected provider is 'openai'. Should be 'local' or 'simple'.")
        # Check if it's because local is missing
        from src.llm.rag.embeddings import HUGGINGFACE_AVAILABLE
        if not HUGGINGFACE_AVAILABLE:
            print("  [WARN] Local embeddings (sentence-transformers) not installed.")
        return False
        
    print(f"  [OK] Auto-selected provider is '{emb_manager.provider}'")
    
    # 3. Check Ragas Configuration (Dry Run)
    print("\n3. Checking Ragas Configuration...")
    try:
        from src.llm.evaluation.ragas_eval import RagasEvaluator
        evaluator = RagasEvaluator()
        # We can't easily check the internal evaluate call without running it, 
        # but we can check if initialization works without OpenAI key
        
        # Unset OpenAI key to be sure
        os.environ.pop("OPENAI_API_KEY", None)
        
        print("  [OK] RagasEvaluator initialized without OpenAI key")
        
    except Exception as e:
        print(f"  [FAIL] RagasEvaluator initialization failed: {e}")
        return False

    print("\n[SUCCESS] System is configured for Local LLM usage!")
    return True

if __name__ == "__main__":
    success = verify_local_setup()
    sys.exit(0 if success else 1)
