import sys
import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path.cwd()
sys.path.insert(0, str(project_root))
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

print("1. Importing src.llm.rag...")
try:
    from src.llm.rag import VectorStoreManager
    print("   [OK] src.llm.rag")
except Exception as e:
    print(f"   [FAIL] src.llm.rag: {e}")

print("2. Importing src.llm.rag.chains...")
try:
    from src.llm.rag.chains import RAGChain
    print("   [OK] src.llm.rag.chains")
except Exception as e:
    print(f"   [FAIL] src.llm.rag.chains: {e}")

print("3. Importing src.llm.evaluation...")
try:
    import src.llm.evaluation
    print("   [OK] src.llm.evaluation module")
except Exception as e:
    print(f"   [FAIL] src.llm.evaluation module: {e}")

print("4. Importing RagasEvaluator...")
try:
    from src.llm.evaluation import RagasEvaluator
    print("   [OK] RagasEvaluator")
except Exception as e:
    print(f"   [FAIL] RagasEvaluator: {e}")

print("5. Importing HallucinationDetector...")
try:
    from src.llm.evaluation import HallucinationDetector
    print("   [OK] HallucinationDetector")
except Exception as e:
    print(f"   [FAIL] HallucinationDetector: {e}")

print("6. Importing SafetyEvaluator...")
try:
    from src.llm.evaluation import SafetyEvaluator
    print("   [OK] SafetyEvaluator")
except Exception as e:
    print(f"   [FAIL] SafetyEvaluator: {e}")

print("7. Importing monitoring...")
try:
    from src.llm.production.monitoring import LLMMonitor
    print("   [OK] LLMMonitor")
except Exception as e:
    print(f"   [FAIL] LLMMonitor: {e}")
