"""Fetch transaction data from SQL Server database - READ-ONLY MODE.

WARNING: This module ONLY supports read operations. All operations are 
restricted to SELECT queries. No write, update, delete, or DDL operations 
are permitted.
"""

import pandas as pd
import logging
from datetime import datetime

from ..config.db_config import DatabaseConfig

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch and validate transaction data from SQL Server - READ-ONLY MODE.
    
    This class provides methods to retrieve data from SQL Server tables.
    All operations are restricted to SELECT queries only. No modifications
    to the database are permitted.
    """
    
    @staticmethod
    def fetch_transactions(table_name, date_column='Date', 
                          start_date=None, end_date=None):
        """
        Fetch transaction data from SQL Server.
        
        Args:
            table_name: Name of the transactions table
            date_column: Name of the date column
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
        
        Returns:
            DataFrame with transaction data
        
        Raises:
            ConnectionError: If database connection fails
            ValueError: If table or columns don't exist
        """
        try:
            conn = DatabaseConfig.connect()
            
            # Build query
            query = f"SELECT * FROM {table_name}"
            params = []
            
            # Validate read-only query before execution
            DatabaseConfig.is_read_only_query(query)
            
            if start_date or end_date:
                where_clauses = []
                if start_date:
                    where_clauses.append(f"{date_column} >= ?")
                    params.append(start_date)
                if end_date:
                    where_clauses.append(f"{date_column} <= ?")
                    params.append(end_date)
                
                query += " WHERE " + " AND ".join(where_clauses)
            
            logger.info(f"Executing query: {query}")
            
            # Fetch data with parameterized query (read-only)
            if params:
                df = pd.read_sql(query, conn, params=params)
            else:
                df = pd.read_sql(query, conn)
            
            logger.info(f"READ-ONLY: Successfully fetched data from {table_name}")
            
            conn.close()
            
            logger.info(f"Fetched {len(df)} records from {table_name}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {table_name}: {e}")
            raise
    
    @staticmethod
    def validate_columns(df, required_columns):
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
        
        Returns:
            True if valid, raises exception otherwise
        
        Raises:
            ValueError: If required columns are missing
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            available = list(df.columns)
            error_msg = (
                f"Missing required columns: {missing}\n"
                f"Available columns: {available}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"All required columns present: {required_columns}")
        return True
    
    @staticmethod
    def get_table_columns(table_name):
        """
        Get column names and types from a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            Dictionary of {column_name: data_type}
        """
        try:
            conn = DatabaseConfig.connect()
            
            # Get column information using SQL Server system views (read-only)
            query = """
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """
            
            # Validate read-only query before execution
            DatabaseConfig.is_read_only_query(query)
            
            df = pd.read_sql(query, conn, params=[table_name])
            logger.info(f"READ-ONLY: Successfully retrieved schema for {table_name}")
            conn.close()
            
            if df.empty:
                raise ValueError(f"Table '{table_name}' not found")
            
            columns_dict = dict(zip(df['COLUMN_NAME'], df['DATA_TYPE']))
            logger.info(f"Columns in {table_name}: {columns_dict}")
            
            return columns_dict
            
        except Exception as e:
            logger.error(f"Error getting table columns: {e}")
            raise
    
    @staticmethod
    def get_row_count(table_name):
        """
        Get row count from a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            Number of rows in the table
        """
        try:
            conn = DatabaseConfig.connect()
            query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            
            # Validate read-only query before execution
            DatabaseConfig.is_read_only_query(query)
            
            df = pd.read_sql(query, conn)
            logger.info(f"READ-ONLY: Successfully retrieved row count for {table_name}")
            conn.close()
            
            count = df['row_count'].iloc[0]
            logger.info(f"{table_name} has {count} rows")
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting row count: {e}")
            raise
