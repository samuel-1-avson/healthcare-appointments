"""
Generate SQL Analytics Report
==============================
Reads all CSV results and generates comprehensive markdown report
"""

import pandas as pd
from pathlib import Path

RESULTS_DIR = "sql_analytics/results"

# Read all results
results = {}
for csv_file in Path(RESULTS_DIR).glob("*.csv"):
    if csv_file.stem != "_execution_summary":
        results[csv_file.stem] = pd.read_csv(csv_file)

# Print key findings for report generation
print("="*60)
print("QUERY RESULTS SUMMARY")
print("="*60)

for name, df in results.items():
    print(f"\n{'='*60}")
    print(f"Query: {name}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print()

