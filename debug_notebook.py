import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Simulate notebook environment
# Notebook is in notebooks/week9_prompt_engineering.ipynb
# It does: project_root = Path.cwd().parent
# If we run this script from project root, we should adjust.

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.llm import LLMClient, get_llm_config

def debug():
    config = get_llm_config()
    print(f"Default Provider: {config.default_provider}")
    print(f"Default Model: {config.default_model}")
    
    client = LLMClient(config)
    print(f"Client Available: {client.is_available}")
    print(f"Providers: {list(client._providers.keys())}")
    
    if "local" not in client._providers:
        print("ERROR: Local provider missing!")
    else:
        print("SUCCESS: Local provider present.")

if __name__ == "__main__":
    debug()
