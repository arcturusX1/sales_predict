# Database Read-Only Security

## Overview
The database modules (`src/config/db_config.py` and `src/utils/data_fetcher.py`) are configured for **READ-ONLY mode only**. 

**NO WRITE OPERATIONS ARE PERMITTED.** Any attempt to execute INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, or other data-modifying commands will be blocked with a `PermissionError`.

---

## Protected Operations

The following SQL operations are **strictly blocked**:

### Data Modification
- ✗ `INSERT` - Adding new records
- ✗ `UPDATE` - Modifying existing records
- ✗ `DELETE` - Removing records
- ✗ `TRUNCATE` - Clearing table data

### Schema Changes (DDL)
- ✗ `CREATE` - Creating new tables/indexes
- ✗ `ALTER` - Modifying table structure
- ✗ `DROP` - Deleting tables/indexes

### Procedural Operations
- ✗ `EXEC` / `EXECUTE` - Running stored procedures
- ✗ `BEGIN` / `COMMIT` / `ROLLBACK` - Transaction control
- ✗ `GRANT` / `REVOKE` - Permission changes

---

## Allowed Operations

Only **SELECT queries** are permitted:

- ✓ `SELECT * FROM table_name`
- ✓ `SELECT col1, col2 FROM table_name WHERE condition`
- ✓ `SELECT COUNT(*) FROM table_name`
- ✓ Queries from system views like `INFORMATION_SCHEMA.COLUMNS`

---

## Implementation Details

### DatabaseConfig.is_read_only_query(query)

This static method validates all queries before execution:

```python
# Validates query is SELECT-only
DatabaseConfig.is_read_only_query(query)  # Returns True if valid

# Raises PermissionError if query contains write operations
try:
    DatabaseConfig.is_read_only_query("INSERT INTO table VALUES (1)")
except PermissionError as e:
    print(e)  # "WRITE OPERATION BLOCKED: Query contains 'INSERT'..."
```

### Validation Points

All DataFetcher methods validate queries before execution:

1. **fetch_transactions()** - Validates SELECT query before fetch
2. **get_table_columns()** - Validates schema query before execution
3. **get_row_count()** - Validates COUNT query before execution

---

## Query Validation Examples

### ✓ Valid (Allowed)
```python
DataFetcher.fetch_transactions('Transactions')  # SELECT * FROM Transactions

DataFetcher.fetch_transactions('Sales', start_date='2025-01-01', end_date='2025-01-31')
# SELECT * FROM Sales WHERE Date >= ? AND Date <= ?

DataFetcher.get_table_columns('Transactions')
# SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?

DataFetcher.get_row_count('Transactions')
# SELECT COUNT(*) as row_count FROM Transactions
```

### ✗ Invalid (Blocked)
```python
# Any attempt to use write operations would fail at query validation:
# - Cannot be modified to support INSERT, UPDATE, DELETE, etc.
# - Cannot add schema modification capabilities
# - Cannot execute stored procedures
```

---

## Security Guarantees

1. **No Database Modifications**: The application **cannot** modify any data in the database
2. **Read-Only Only**: All operations are restricted to SELECT statements
3. **Query Validation**: Every query is validated before execution
4. **Explicit Error Messages**: Write attempts fail with clear error messages
5. **No Workarounds**: Keyword detection blocks INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, TRUNCATE, EXEC, and transaction control

---

## Testing

Run the comprehensive read-only test suite:

```bash
python3 test_db_connection.py
```

This validates:
- Database connectivity
- Table schema inspection  
- Sample data fetching
- Read-only enforcement

---

## Usage

```python
from src.utils.data_fetcher import DataFetcher

# Safe operations - no write risk
df = DataFetcher.fetch_transactions('Transactions')
columns = DataFetcher.get_table_columns('Transactions')
count = DataFetcher.get_row_count('Transactions')

# These would raise PermissionError
# df = DataFrame.execute("INSERT INTO Transactions VALUES (...)")  # ✗ Blocked
```

---

## Client Database Safety

✓ **Safe for production use with client databases**  
✓ **No risk of accidental data modification**  
✓ **All write operations explicitly blocked**  
✓ **Clear error messages for any write attempt**
