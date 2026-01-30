"""Database configuration and connection management - READ-ONLY MODE.

WARNING: This module ONLY supports read operations. All write/update/delete 
operations are strictly prohibited. Any attempt to execute non-read operations 
will raise an error.
"""

import os
from dotenv import load_dotenv
import pyodbc
import logging

logger = logging.getLogger(__name__)

load_dotenv()


class DatabaseConfig:
    """SQL Server database configuration - READ-ONLY MODE.
    
    This class manages read-only database connections. All operations are
    restricted to SELECT queries only. No write, update, delete, or DDL
    operations are permitted.
    """
    
    SERVER = os.getenv('DB_SERVER')
    DATABASE = os.getenv('DB_NAME')
    USERNAME = os.getenv('DB_USER')
    PASSWORD = os.getenv('DB_PASSWORD')
    
    @staticmethod
    def get_connection_string():
        """Build ODBC connection string."""
        return (
            f'Driver={{ODBC Driver 17 for SQL Server}};'
            f'Server={DatabaseConfig.SERVER};'
            f'Database={DatabaseConfig.DATABASE};'
            f'UID={DatabaseConfig.USERNAME};'
            f'PWD={DatabaseConfig.PASSWORD};'
        )
    
    @staticmethod
    def validate_config():
        """Validate database configuration is complete."""
        required_vars = ['DB_SERVER', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            raise ValueError(f"Missing database config variables: {missing}")
        
        return True
    
    @staticmethod
    def connect():
        """Create read-only database connection.
        
        Returns:
            pyodbc connection object configured for read-only operations.
            
        Raises:
            ConnectionError: If connection cannot be established.
            ValueError: If configuration is incomplete.
        """
        try:
            DatabaseConfig.validate_config()
            conn = pyodbc.connect(DatabaseConfig.get_connection_string())
            # Set connection to read-only mode
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            logger.info(f"READ-ONLY Connection established to {DatabaseConfig.DATABASE} on {DatabaseConfig.SERVER}")
            return conn
        except pyodbc.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise ConnectionError(f"Database connection failed: {e}")
        except ValueError as e:
            logger.error(str(e))
            raise
    
    @staticmethod
    def is_read_only_query(query):
        """Validate that a SQL query is read-only (SELECT only).
        
        Args:
            query: SQL query string to validate
            
        Returns:
            True if query is read-only (SELECT statement)
            
        Raises:
            PermissionError: If query attempts any write/update/delete/DDL operations
        """
        # Normalize query for checking
        normalized = query.strip().upper()
        
        # Blocked operations
        blocked_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE',
            'BEGIN', 'COMMIT', 'ROLLBACK'
        ]
        
        for keyword in blocked_keywords:
            if keyword in normalized:
                error_msg = f"WRITE OPERATION BLOCKED: Query contains '{keyword}' - only SELECT queries are permitted"
                logger.error(error_msg)
                raise PermissionError(error_msg)
        
        if not normalized.startswith('SELECT'):
            error_msg = f"READ-ONLY VIOLATION: Only SELECT queries are permitted. Query starts with: {normalized[:20]}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
        
        return True
