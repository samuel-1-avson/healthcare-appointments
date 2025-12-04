import json

def add_debug():
    notebook_path = r"C:\Users\samue\Desktop\NSP\healthcare-appointments\notebooks\week9_prompt_engineering.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Find Cell 2
    for cell in nb['cells']:
        if "Cell 2: Initialize LLM Client" in "".join(cell['source']):
            # Inject debug import
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if "from src.llm import LLMClient" in line:
                    new_source.append("import src.llm.client\n")
                    new_source.append("print(f'Client File: {src.llm.client.__file__}')\n")
            cell['source'] = new_source
            break
            
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Debug added to notebook.")

if __name__ == "__main__":
    add_debug()
