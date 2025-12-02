"""Check the structure of healthcare.db"""
import sqlite3

conn = sqlite3.connect('healthcare.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("Tables in healthcare.db:")
print("=" * 50)
for table in tables:
    print(f"\n📋 {table[0]}")
    cursor.execute(f"PRAGMA table_info({table[0]})")
    columns = cursor.fetchall()
    print("  Columns:")
    for col in columns:
        print(f"    - {col[1]} ({col[2]})")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()[0]
    print(f"  Row count: {count:,}")

conn.close()
