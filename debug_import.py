import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    import src.llm.memory
    print(f"Module file: {src.llm.memory.__file__}")
    
    with open(src.llm.memory.__file__, 'r') as f:
        print("--- Content of module ---")
        print(f.read())
        print("-------------------------")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
