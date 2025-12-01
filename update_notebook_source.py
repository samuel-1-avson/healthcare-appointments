import nbformat

notebook_path = 'healthcare_appointments_eda.ipynb'
target_string = 'url = "https://raw.githubusercontent.com/datasets/no-show-appointments/main/data/KaggleV2-May-2016.csv"'
replacement_code = """# Use local raw file
    raw_file = "data/raw/KaggleV2-May-2016.csv"
    print(f"📂 Loading raw data from: {raw_file}")
    df = pd.read_csv(raw_file, encoding='latin-1')"""

def update_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    found = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if target_string in cell.source:
                print("Found target cell.")
                # Replace the specific lines
                new_source = cell.source.replace(
                    'url = "https://raw.githubusercontent.com/datasets/no-show-appointments/main/data/KaggleV2-May-2016.csv"\n    df = pd.read_csv(url, encoding=\'latin-1\')',
                    replacement_code
                )
                cell.source = new_source
                found = True
                break
    
    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Successfully updated {notebook_path}")
    else:
        print("Target string not found in notebook.")

if __name__ == "__main__":
    update_notebook()
