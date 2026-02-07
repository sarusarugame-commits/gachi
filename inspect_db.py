import sqlite3
import pandas as pd
from datetime import datetime

def inspect_db():
    conn = sqlite3.connect('race_data.db')
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", tables)
    
    for table_name in tables:
        table_name = table_name[0]
        print(f"\nSchema for {table_name}:")
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        for col in columns:
            print(col)
            
        # Try to get data for today if there's a date column
        # Assuming there might be a 'date' or 'timestamp' column
        print(f"\nSample data from {table_name} (first 5 rows):")
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
            print(df)
        except Exception as e:
            print(f"Error reading data: {e}")

        # Check for today's data (2026-02-07)
        # This is a guess on column names based on common practices
        try:
            # common date columns: date, race_date, timestamp
            date_cols = [col[1] for col in columns if 'date' in col[1].lower() or 'time' in col[1].lower()]
            if date_cols:
                col = date_cols[0]
                print(f"\nChecking for today's data in {table_name} column {col}...")
                query = f"SELECT * FROM {table_name} WHERE {col} LIKE '2026-02-07%'"
                df_today = pd.read_sql_query(query, conn)
                print(f"Found {len(df_today)} records for today.")
                if not df_today.empty:
                    print(df_today.head())
        except Exception as e:
            print(f"Error checking today's data: {e}")

    conn.close()

if __name__ == "__main__":
    inspect_db()
