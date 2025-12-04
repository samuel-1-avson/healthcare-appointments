import json

def add_autoreload():
    notebook_path = r"C:\Users\samue\Desktop\NSP\healthcare-appointments\notebooks\week9_prompt_engineering.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Find Cell 1 (Imports)
    for cell in nb['cells']:
        if "Cell 1: Imports and Setup" in "".join(cell['source']):
            # Add autoreload magic
            new_source = ["%load_ext autoreload\n", "%autoreload 2\n", "\n"] + cell['source']
            cell['source'] = new_source
            break
            
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Autoreload added to notebook.")

if __name__ == "__main__":
    add_autoreload()
