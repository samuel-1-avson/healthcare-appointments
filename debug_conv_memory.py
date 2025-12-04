import sys
import os
import importlib.util

sys.path.insert(0, os.getcwd())

print(f"Searching for src.llm.memory.conversation_memory")

try:
    spec = importlib.util.find_spec("src.llm.memory.conversation_memory")
    if spec:
        print(f"Found spec: {spec}")
        print(f"Origin: {spec.origin}")
        if spec.origin:
            with open(spec.origin, 'r') as f:
                print("--- Content of origin (first 30 lines) ---")
                lines = f.readlines()
                for i, line in enumerate(lines[:30]):
                    print(f"{i+1}: {line}", end='')
                print("\n-------------------------")
    else:
        print("Module src.llm.memory.conversation_memory not found")
except Exception as e:
    print(f"Error finding spec: {e}")
