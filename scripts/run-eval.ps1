# scripts/run-eval.ps1
# Run evaluation suite

param(
    [switch]$RAG,
    [switch]$Safety,
    [switch]$Full
)

$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

# Activate venv
$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) { & $venvActivate }

Write-Host "Running Evaluation Suite..." -ForegroundColor Cyan

if ($Full -or (-not $RAG -and -not $Safety)) {
    python -c @"
import sys
sys.path.insert(0, '.')

from src.llm.evaluation import EvaluationFramework, EvaluationConfig
from src.llm.rag.chains import RAGChain
from src.llm.rag import get_vector_store

# Load vector store
vs = get_vector_store()
vs.load('default')

# Create RAG chain
rag = RAGChain(vs)

# Create evaluation framework
config = EvaluationConfig()
framework = EvaluationFramework(config)
framework.register_rag_chain(rag)
framework.load_golden_set('evals/golden_set.json')

# Run evaluation
report = framework.run_full_evaluation()

# Print summary
print(report.to_markdown())
"@
}

Write-Host "Evaluation complete! Check evals/results/ for detailed reports." -ForegroundColor Green