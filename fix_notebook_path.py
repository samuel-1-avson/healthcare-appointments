import json

def fix_path():
    notebook_path = r"C:\Users\samue\Desktop\NSP\healthcare-appointments\notebooks\week9_prompt_engineering.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Find Cell 1 (Imports)
    for cell in nb['cells']:
        if "Cell 1: Imports and Setup" in "".join(cell['source']):
            # Replace path setup logic
            new_source = []
            for line in cell['source']:
                if "project_root = Path.cwd().parent" in line:
                    new_source.append("# Add project root to path\n")
                    new_source.append("current_dir = Path.cwd()\n")
                    new_source.append("if current_dir.name == 'notebooks':\n")
                    new_source.append("    project_root = current_dir.parent\n")
                    new_source.append("else:\n")
                    new_source.append("    project_root = current_dir\n")
                elif "sys.path.insert(0, str(project_root))" in line:
                    new_source.append("sys.path.insert(0, str(project_root))\n")
                    new_source.append("print(f'Project Root: {project_root}')\n") # Debug print
                elif "# Add project root to path" in line:
                    pass # Skip, we added it above
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break
            
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Path logic fixed in notebook.")

if __name__ == "__main__":
    fix_path()
