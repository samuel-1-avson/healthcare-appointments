import json
from pathlib import Path

notebook_path = Path(r"C:\Users\samue\Desktop\NSP\healthcare-appointments\notebooks\week10_langchain.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New source code for Cell 1
new_source = [
    "# Cell 1: Setup and Imports\n",
    "import os\n",
    "import sys\n",
    "from pathlib import Path\n",
    "from datetime import datetime\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "# Add project root\n",
    "project_root = Path.cwd().parent\n",
    "sys.path.insert(0, str(project_root))\n",
    "\n",
    "# Environment Check\n",
    "print(f\"Python Executable: {sys.executable}\")\n",
    "print(f\"Python Version: {sys.version}\")\n",
    "\n",
    "try:\n",
    "    import langchain\n",
    "    print(f\"LangChain Version: {langchain.__version__}\")\n",
    "except ImportError:\n",
    "    print(\"⚠️ LangChain not found. Installing dependencies...\")\n",
    "    import subprocess\n",
    "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-r\", str(project_root / \"requirements.txt\")])\n",
    "    print(\"Dependencies installed. Please restart the kernel.\")\n",
    "\n",
    "load_dotenv()\n",
    "\n",
    "# Verify setup\n",
    "print(\"OpenAI API Key:\", \"✅\" if os.getenv(\"OPENAI_API_KEY\") else \"❌\")\n",
    "print(\"Anthropic API Key:\", \"✅\" if os.getenv(\"ANTHROPIC_API_KEY\") else \"❌\")\n",
    "\n",
    "# LangChain imports\n",
    "try:\n",
    "    from langchain_core.prompts import ChatPromptTemplate\n",
    "    from langchain_core.output_parsers import StrOutputParser\n",
    "    from langchain_core.runnables import RunnablePassthrough\n",
    "    print(\"LangChain imports: ✅\")\n",
    "except ImportError as e:\n",
    "    print(f\"LangChain imports failed: {e}\")\n",
    "    print(\"Please ensure you are running in the correct virtual environment.\")\n"
]

# Find the first code cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Verify it's the setup cell by checking content
        if "Setup and Imports" in "".join(cell['source']):
            cell['source'] = new_source
            cell['outputs'] = [] # Clear outputs to avoid confusion
            print("Updated setup cell.")
            break
else:
    print("Setup cell not found!")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook saved.")
