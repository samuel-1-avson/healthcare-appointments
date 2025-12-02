# scripts/init-rag.ps1
# Initialize or rebuild RAG vector store

param(
    [string]$DocumentsPath = "data/documents",
    [string]$IndexName = "default",
    [switch]$Force
)

$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

# Activate venv
$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) { & $venvActivate }

Write-Host "Initializing RAG Vector Store..." -ForegroundColor Cyan

$forceFlag = if ($Force) { "True" } else { "False" }

python -c @"
import sys
sys.path.insert(0, '.')

from src.llm.rag import load_policy_documents, VectorStoreManager
from pathlib import Path

docs_path = '$DocumentsPath'
index_name = '$IndexName'
force = $forceFlag

# Check existing
store_path = Path('data/vector_store') / index_name
if store_path.exists() and not force:
    print(f'Index already exists at {store_path}. Use -Force to rebuild.')
else:
    print(f'Loading documents from {docs_path}...')
    docs = load_policy_documents(docs_path)
    print(f'Loaded {len(docs)} documents')
    
    print('Creating vector store...')
    manager = VectorStoreManager()
    manager.create_from_documents(docs, chunk_size=1000, chunk_overlap=200)
    manager.save(index_name)
    print(f'Vector store saved as "{index_name}"!')
"@

Write-Host "RAG initialization complete!" -ForegroundColor Green