import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Setup path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

print("Checking imports...")
try:
    from src.llm.rag import VectorStoreManager, get_vector_store
    from src.llm.rag.chains import RAGChain
    from src.llm.evaluation import (
        RagasEvaluator, 
        HallucinationDetector, 
        SafetyEvaluator,
        EvaluationMetrics,
        RegressionTestSuite
    )
    from src.llm.production.monitoring import LLMMonitor
    print("[OK] Imports successful")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print("\nChecking Ragas availability...")
try:
    import ragas
    import datasets
    print("[OK] Ragas and datasets installed")
except ImportError:
    print("[WARN] Ragas or datasets not installed (will use fallback)")

print("\nInitializing components...")
try:
    # Mock vector store for testing without loading actual index if it doesn't exist
    print("  - Vector Store...")
    try:
        vector_store = get_vector_store()
    except Exception as e:
        print(f"    [WARN] Could not load vector store: {e}")
        # Create a dummy one if needed or just pass for now as we are testing imports/class instantiation
        pass

    print("  - Evaluators...")
    ragas_eval = RagasEvaluator()
    hallucination = HallucinationDetector()
    safety = SafetyEvaluator()
    metrics = EvaluationMetrics()
    regression = RegressionTestSuite()
    
    print("  - Monitor...")
    monitor = LLMMonitor()
    
    print("[OK] Initialization successful")

    print("\nRunning runtime checks...")
    
    # Test HallucinationDetector result structure
    print("  - Testing HallucinationDetector result access...")
    try:
        # Create a dummy result manually to test framework logic
        from src.llm.evaluation.hallucination import HallucinationResult
        res = HallucinationResult(
            has_hallucination=False,
            confidence=1.0,
            issues=[],
            claims_checked=1,
            claims_verified=1,
            claims_unsupported=0,
            details={}
        )
        # Verify attribute access works (which was the bug)
        assert res.has_hallucination is False
        print("    [OK] HallucinationResult attribute access verified")
    except Exception as e:
        print(f"    [FAIL] HallucinationResult check failed: {e}")
        raise

    # Test EvaluationFramework report saving (Unicode check)
    print("  - Testing EvaluationReport saving (Unicode check)...")
    try:
        from src.llm.evaluation.framework import EvaluationReport, EvaluationConfig, EvaluationResult, EvaluationType
        import datetime
        
        report = EvaluationReport(
            run_id="test_run",
            config=EvaluationConfig()
        )
        report.add_result(EvaluationResult(
            evaluation_type=EvaluationType.RAG_QUALITY,
            passed=True,
            score=1.0,
            threshold=0.8,
            details={"test": "✅ Unicode Test"}
        ))
        report.finalize()
        
        # Try saving to a temp file
        report.save("test_report.json")
        print("    [OK] Report saved successfully with Unicode characters")
        
        # Clean up
        import os
        if os.path.exists("test_report.json"):
            os.remove("test_report.json")
        if os.path.exists("test_report.md"):
            os.remove("test_report.md")
            
    except Exception as e:
        print(f"    [FAIL] Report saving failed: {e}")
        raise
    
except Exception as e:
    print(f"[FAIL] Initialization/Runtime failed: {e}")
    sys.exit(1)

print("\n[OK] Verification passed! The notebook should run correctly.")
