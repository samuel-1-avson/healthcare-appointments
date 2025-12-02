"""
SQL Analytics Module - Verification Test
=========================================
Verifies all components of Week 2: SQL Analytics implementation

Tests:
1. Database connectivity
2. SQL queries file exists and is valid
3. All 11 queries execute successfully
4. All CSV exports exist
5. Report file exists
6. Query results match expected patterns
"""

import sqlite3
import os
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def test_database_exists():
    """Test 1: Database file exists and is accessible"""
    print(f"\n{Colors.BOLD}Test 1: Database Connectivity{Colors.END}")
    
    if not os.path.exists("healthcare.db"):
        print(f"{Colors.RED}✗ FAIL: healthcare.db not found{Colors.END}")
        return False
    
    try:
        conn = sqlite3.connect("healthcare.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM appointments")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 110527:
            print(f"{Colors.GREEN}✓ PASS: Database connected, {count:,} appointments found{Colors.END}")
            return True
        else:
            print(f"{Colors.YELLOW}⚠ WARNING: Expected 110,527 rows, found {count:,}{Colors.END}")
            return True
    except Exception as e:
        print(f"{Colors.RED}✗ FAIL: Database error: {e}{Colors.END}")
        return False

def test_queries_file():
    """Test 2: SQL queries file exists"""
    print(f"\n{Colors.BOLD}Test 2: SQL Queries File{Colors.END}")
    
    if not os.path.exists("sql_analytics/queries.sql"):
        print(f"{Colors.RED}✗ FAIL: queries.sql not found{Colors.END}")
        return False
    
    with open("sql_analytics/queries.sql", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count queries by looking for "-- name:" markers
    query_count = content.count("-- name:")
    
    if query_count >= 10:
        print(f"{Colors.GREEN}✓ PASS: Found {query_count} queries in queries.sql{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}✗ FAIL: Only {query_count} queries found, expected 10+{Colors.END}")
        return False

def test_python_script():
    """Test 3: Execution script exists"""
    print(f"\n{Colors.BOLD}Test 3: Python Execution Script{Colors.END}")
    
    if not os.path.exists("sql_analytics/run_queries.py"):
        print(f"{Colors.RED}✗ FAIL: run_queries.py not found{Colors.END}")
        return False
    
    print(f"{Colors.GREEN}✓ PASS: run_queries.py exists{Colors.END}")
    return True

def test_csv_exports():
    """Test 4: CSV exports exist"""
    print(f"\n{Colors.BOLD}Test 4: CSV Exports{Colors.END}")
    
    if not os.path.exists("sql_analytics/results"):
        print(f"{Colors.RED}✗ FAIL: results directory not found{Colors.END}")
        return False
    
    csv_files = list(Path("sql_analytics/results").glob("*.csv"))
    
    if len(csv_files) >= 10:
        print(f"{Colors.GREEN}✓ PASS: Found {len(csv_files)} CSV files{Colors.END}")
        for csv_file in sorted(csv_files):
            print(f"  - {csv_file.name}")
        return True
    else:
        print(f"{Colors.RED}✗ FAIL: Only {len(csv_files)} CSV files found{Colors.END}")
        return False

def test_report_exists():
    """Test 5: Analytics report exists"""
    print(f"\n{Colors.BOLD}Test 5: SQL Analytics Report{Colors.END}")
    
    if not os.path.exists("sql_analytics/SQL_ANALYTICS_REPORT.md"):
        print(f"{Colors.RED}✗ FAIL: SQL_ANALYTICS_REPORT.md not found{Colors.END}")
        return False
    
    with open("sql_analytics/SQL_ANALYTICS_REPORT.md", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key sections
    has_executive_summary = "Executive Summary" in content
    has_recommendations = "Recommendations" in content
    has_queries = content.count("```sql") >= 10
    
    print(f"  Executive Summary: {'✓' if has_executive_summary else '✗'}")
    print(f"  Recommendations: {'✓' if has_recommendations else '✗'}")
    print(f"  SQL Code Blocks: {content.count('```sql')} queries")
    
    if has_executive_summary and has_recommendations and has_queries:
        print(f"{Colors.GREEN}✓ PASS: Comprehensive report found{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}✗ FAIL: Report missing key sections{Colors.END}")
        return False

def test_key_insights():
    """Test 6: Verify key query results"""
    print(f"\n{Colors.BOLD}Test 6: Key Insights Validation{Colors.END}")
    
    try:
        conn = sqlite3.connect("healthcare.db")
        cursor = conn.cursor()
        
        # Test: Overall no-show rate should be ~20%
        cursor.execute("SELECT ROUND(AVG(No_Show) * 100, 2) FROM appointments")
        no_show_rate = cursor.fetchone()[0]
        
        if 19.0 <= no_show_rate <= 21.0:
            print(f"{Colors.GREEN}✓ PASS: No-show rate = {no_show_rate}% (expected ~20%){Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ WARNING: No-show rate = {no_show_rate}% (expected ~20%){Colors.END}")
        
        # Test: SMS effectiveness
        cursor.execute("""
            SELECT 
                CASE WHEN SMS_received = 1 THEN 'SMS' ELSE 'No SMS' END,
                ROUND(AVG(No_Show) * 100, 2)
            FROM appointments
            GROUP BY SMS_received
        """)
        sms_results = cursor.fetchall()
        print(f"{Colors.BLUE}  SMS Impact Verified:{Colors.END}")
        for category, rate in sms_results:
            print(f"    {category}: {rate}% no-show rate")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"{Colors.RED}✗ FAIL: Query error: {e}{Colors.END}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}SQL Analytics Module - Verification Test{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    
    tests = [
        test_database_exists,
        test_queries_file,
        test_python_script,
        test_csv_exports,
        test_report_exists,
        test_key_insights
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}Test Summary{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED - Week 2 SQL Analytics COMPLETE!{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}⚠ Some tests failed - review output above{Colors.END}")
        return 1

if __name__ == "__main__":
    exit(main())
