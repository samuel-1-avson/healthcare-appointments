"""
SQL Analytics Query Executor
=============================
Week 2: SQL for Analytics - Healthcare Appointments

This script executes 10 stakeholder-focused KPI queries against
the healthcare.db SQLite database and exports results to CSV.

Usage:
    python sql_analytics/run_queries.py

Output:
    - CSV files in sql_analytics/results/
    - Execution log with timing and row counts
"""

import sqlite3
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path

# Configuration
DB_PATH = "healthcare.db"
QUERIES_FILE = "sql_analytics/queries.sql"
RESULTS_DIR = "sql_analytics/results"

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def parse_sql_file(filepath):
    """
    Parse SQL file and extract individual queries with their names.
    Queries are separated by comments containing '-- name: query_name'
    
    Returns:
        list of tuples: (query_name, sql_query)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    queries = []
    current_name = None
    current_query = []
    
    for line in content.split('\n'):
        # Check if this is a name declaration
        if line.strip().startswith('-- name:'):
            # Save previous query if exists
            if current_name and current_query:
                sql = '\n'.join(current_query).strip()
                if sql and not sql.startswith('--'):
                    queries.append((current_name, sql))
            
            # Start new query
            current_name = line.split('-- name:')[1].strip()
            current_query = []
        elif not line.strip().startswith('--') or line.strip() == '--':
            # Add to current query (skip standalone comment lines)
            if line.strip() and not line.strip().startswith('-- ='):
                current_query.append(line)
    
    # Save last query
    if current_name and current_query:
        sql = '\n'.join(current_query).strip()
        if sql:
            queries.append((current_name, sql))
    
    return queries

def execute_query(conn, query_name, sql):
    """
    Execute a single SQL query and return results as DataFrame
    
    Args:
        conn: SQLite connection
        query_name: Name of the query
        sql: SQL query string
    
    Returns:
        tuple: (DataFrame, execution_time_ms, row_count)
    """
    start_time = time.time()
    
    try:
        df = pd.read_sql_query(sql, conn)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        return df, execution_time, len(df)
    except Exception as e:
        print_error(f"Query '{query_name}' failed: {str(e)}")
        return None, 0, 0

def save_to_csv(df, filepath):
    """Save DataFrame to CSV with proper formatting"""
    df.to_csv(filepath, index=False, encoding='utf-8')

def main():
    """Main execution function"""
    
    print_header("SQL Analytics Query Executor")
    print(f"Database: {DB_PATH}")
    print(f"Queries: {QUERIES_FILE}")
    print(f"Output: {RESULTS_DIR}/")
    
    # Create results directory
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    print_success(f"Results directory ready: {RESULTS_DIR}/")
    
    # Connect to database
    try:
        conn = sqlite3.connect(DB_PATH)
        print_success(f"Connected to database: {DB_PATH}")
    except Exception as e:
        print_error(f"Failed to connect to database: {e}")
        return
    
    # Parse SQL queries
    try:
        queries = parse_sql_file(QUERIES_FILE)
        print_success(f"Loaded {len(queries)} queries from {QUERIES_FILE}")
    except Exception as e:
        print_error(f"Failed to parse SQL file: {e}")
        conn.close()
        return
    
    # Execute each query
    print_header(f"Executing {len(queries)} KPI Queries")
    
    results_summary = []
    total_start = time.time()
    
    for i, (query_name, sql) in enumerate(queries, 1):
        print(f"\n{Colors.BOLD}[{i}/{len(queries)}] {query_name}{Colors.END}")
        print_info(f"Executing query...")
        
        df, exec_time, row_count = execute_query(conn, query_name, sql)
        
        if df is not None:
            # Save to CSV
            csv_path = os.path.join(RESULTS_DIR, f"{query_name}.csv")
            save_to_csv(df, csv_path)
            
            print_success(f"Returned {row_count} rows in {exec_time:.2f}ms")
            print_success(f"Saved to: {csv_path}")
            
            # Add to summary
            results_summary.append({
                'query_number': i,
                'query_name': query_name,
                'rows_returned': row_count,
                'execution_time_ms': round(exec_time, 2),
                'output_file': f"{query_name}.csv"
            })
            
            # Show preview of first few rows
            if row_count > 0:
                print(f"\n{Colors.CYAN}Preview (first 3 rows):{Colors.END}")
                print(df.head(3).to_string(index=False))
        else:
            results_summary.append({
                'query_number': i,
                'query_name': query_name,
                'rows_returned': 0,
                'execution_time_ms': 0,
                'output_file': 'FAILED'
            })
    
    total_time = (time.time() - total_start) * 1000
    
    # Close database connection
    conn.close()
    print_success("\nDatabase connection closed")
    
    # Create summary DataFrame and save
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(RESULTS_DIR, "_execution_summary.csv")
    save_to_csv(summary_df, summary_path)
    
    # Print final summary
    print_header("Execution Summary")
    print(f"Total queries executed: {len(queries)}")
    print(f"Total execution time: {total_time:.2f}ms")
    print(f"Average time per query: {total_time/len(queries):.2f}ms")
    print(f"Total rows returned: {summary_df['rows_returned'].sum():,}")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Summary saved to: {summary_path}")
    
    # Show summary table
    print(f"\n{Colors.BOLD}Query Performance Summary:{Colors.END}")
    print(summary_df.to_string(index=False))
    
    print_header("✓ All Queries Completed Successfully!")

if __name__ == "__main__":
    main()
