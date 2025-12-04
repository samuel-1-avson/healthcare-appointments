import json

def add_reload():
    notebook_path = r"C:\Users\samue\Desktop\NSP\healthcare-appointments\notebooks\week9_prompt_engineering.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Create a new cell for force reload
    reload_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "force_reload",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Force reload of src.llm modules to pick up latest changes\n",
            "import importlib\n",
            "import sys\n",
            "\n",
            "# Remove cached modules\n",
            "modules_to_reload = [key for key in sys.modules.keys() if key.startswith('src.llm')]\n",
            "for mod in modules_to_reload:\n",
            "    del sys.modules[mod]\n",
            "\n",
            "print(f\"Cleared {len(modules_to_reload)} cached src.llm modules\")\n",
            "\n"
        ]
    }
    
    # Find where to insert (after Cell 1, before Cell 2)
    insert_index = None
    for i, cell in enumerate(nb['cells']):
        if "Cell 1: Imports and Setup" in "".join(cell.get('source', [])):
            insert_index = i + 1
            break
    
    if insert_index:
        nb['cells'].insert(insert_index, reload_cell)
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        
        print(f"Added reload cell at position {insert_index}")
    else:
        print("Could not find Cell 1 to insert after")

if __name__ == "__main__":
    add_reload()
