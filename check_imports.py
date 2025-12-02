import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

print("Checking imports...")

try:
    print("Importing llm_routes...")
    from api.routes import llm_routes
    print("✅ llm_routes imported successfully")
except Exception as e:
    print(f"❌ Failed to import llm_routes: {e}")

try:
    print("Importing rag_routes...")
    from api.routes import rag_routes
    print("✅ rag_routes imported successfully")
except Exception as e:
    print(f"❌ Failed to import rag_routes: {e}")
