"""Test database connection and configuration."""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.db_config import DatabaseConfig
from src.utils.data_fetcher import DataFetcher


def test_connection():
    """Test database connection."""
    print("\n" + "="*60)
    print("Testing Database Connection")
    print("="*60)
    
    try:
        conn = DatabaseConfig.connect()
        print("✓ Connection successful!")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def test_table_info(table_name='Transactions'):
    """Get table information."""
    print(f"\n" + "="*60)
    print(f"Fetching table info for: {table_name}")
    print("="*60)
    
    try:
        # Get columns
        columns = DataFetcher.get_table_columns(table_name)
        print(f"\nColumns found:")
        for col, dtype in columns.items():
            print(f"  - {col}: {dtype}")
        
        # Get row count
        count = DataFetcher.get_row_count(table_name)
        print(f"\nTotal rows: {count:,}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_fetch_sample(table_name='Transactions', limit=5):
    """Fetch and display sample data."""
    print(f"\n" + "="*60)
    print(f"Fetching sample data from {table_name}")
    print("="*60)
    
    try:
        df = DataFetcher.fetch_transactions(table_name)
        
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nFirst {limit} rows:")
        print(df.head(limit).to_string())
        
        print(f"\nData types:")
        print(df.dtypes)
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATABASE TESTING UTILITY")
    print("="*60)
    
    # Test 1: Connection
    if not test_connection():
        print("\n⚠ Cannot proceed without database connection")
        print("Please configure .env file with correct credentials")
        sys.exit(1)
    
    # Test 2: Table info
    if not test_table_info():
        print("\n⚠ Could not retrieve table information")
        print("Verify the table name and database permissions")
        sys.exit(1)
    
    # Test 3: Sample data
    if not test_fetch_sample():
        print("\n⚠ Could not fetch sample data")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
