import json
import shutil
from pathlib import Path

def fix_notebook():
    notebook_path = Path('notebooks/03_python_pipeline.ipynb')
    
    if not notebook_path.exists():
        print(f"Error: {notebook_path} does not exist.")
        return

    # Create backup
    backup_path = notebook_path.with_suffix('.ipynb.bak')
    shutil.copy(notebook_path, backup_path)
    print(f"Created backup at {backup_path}")

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it has the expected corrupted structure
        if 'cells' in data and len(data['cells']) > 0:
            first_cell = data['cells'][0]
            if first_cell.get('cell_type') == 'code':
                source_list = first_cell.get('source', [])
                source_text = ''.join(source_list)
                
                # Try to parse the source as JSON
                try:
                    inner_notebook = json.loads(source_text)
                    
                    # Verify it looks like a notebook
                    if 'cells' in inner_notebook and 'metadata' in inner_notebook:
                        print("Found valid inner notebook structure.")
                        
                        # Write back to file
                        with open(notebook_path, 'w', encoding='utf-8') as f:
                            json.dump(inner_notebook, f, indent=1)
                        print(f"Successfully restored {notebook_path}")
                    else:
                        print("Inner JSON does not look like a valid notebook.")
                except json.JSONDecodeError as e:
                    print(f"Error parsing inner JSON: {e}")
                    # Debug: print first few chars
                    print(f"Source start: {source_text[:100]}")
            else:
                print("First cell is not a code cell.")
        else:
            print("Notebook has no cells.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fix_notebook()
