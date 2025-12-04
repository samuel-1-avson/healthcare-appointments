import json

def add_prompts():
    notebook_path = r"C:\Users\samue\Desktop\NSP\healthcare-appointments\notebooks\week9_prompt_engineering.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Create a new cell with prompt definitions
    prompts_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "prompt_constants",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define healthcare system prompts and safety guidelines\n",
            "HEALTHCARE_SYSTEM_PROMPT = \"\"\"You are a healthcare appointment assistant.\n",
            "Your role is to help explain appointment predictions and provide guidance.\n",
            "\n",
            "Important guidelines:\n",
            "- Do NOT provide medical diagnosis or treatment advice\n",
            "- Focus on appointment scheduling and attendance patterns\n",
            "- Be professional, empathetic, and concise\n",
            "- Respect patient privacy and confidentiality\n",
            "\"\"\"\n",
            "\n",
            "SAFETY_GUIDELINES = \"\"\"\\n\\nSafety Rules:\n",
            "1. Never make medical diagnoses or recommend treatments\n",
            "2. Always defer medical questions to healthcare providers\n",
            "3. Do not speculate about patient health conditions\n",
            "4. Keep responses focused on appointment logistics\n",
            "\"\"\"\n",
            "\n",
            "print(\"Healthcare prompts loaded\")\n",
            "\n"
        ]
    }
    
    # Find Cell 12 (Safety and Guardrails)
    insert_index = None
    for i, cell in enumerate(nb['cells']):
        if "Cell 12: Safety and Guardrails" in "".join(cell.get('source', [])):
            insert_index = i
            break
    
    if insert_index:
        nb['cells'].insert(insert_index, prompts_cell)
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        
        print(f"Added prompt definitions before Cell 12 at position {insert_index}")
    else:
        print("Could not find Cell 12")

if __name__ == "__main__":
    add_prompts()
