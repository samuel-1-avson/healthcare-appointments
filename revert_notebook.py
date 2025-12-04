import json

def revert():
    notebook_path = r"C:\Users\samue\Desktop\NSP\healthcare-appointments\notebooks\week9_prompt_engineering.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Find Cell 2
    for cell in nb['cells']:
        if "Cell 2: Initialize LLM Client" in "".join(cell['source']):
            # Remove debug import
            new_source = []
            for line in cell['source']:
                if "import src.llm.client" not in line and "Client File:" not in line:
                    new_source.append(line)
            cell['source'] = new_source
            break
            
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Debug removed from notebook.")

if __name__ == "__main__":
    revert()
