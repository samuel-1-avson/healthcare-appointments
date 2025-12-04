import sys
import os
import importlib.util

sys.path.insert(0, os.getcwd())

print(f"Searching for src.llm.memory in: {sys.path}")

try:
    spec = importlib.util.find_spec("src.llm.memory")
    if spec:
        print(f"Found spec: {spec}")
        print(f"Origin: {spec.origin}")
        if spec.origin:
            with open(spec.origin, 'r') as f:
                print("--- Content of origin ---")
                print(f.read())
                print("-------------------------")
    else:
        print("Module src.llm.memory not found")
except Exception as e:
    print(f"Error finding spec: {e}")
