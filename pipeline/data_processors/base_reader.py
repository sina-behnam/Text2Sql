"""
# Base dataset reader for Text2SQL datasets
"""
import os
import sqlite3
import pandas as pd
import csv
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sqlparse

from instance import StandardizedInstance

# from src.utils import get_logger

class BaseDatasetReader(ABC):
    """Abstract base class for dataset readers"""
    
    def __init__(self, dataset_path: str, split: str = 'dev'):
        """
        Initialize the dataset reader
        
        Args:
            dataset_path: Path to the dataset directory
            split: Dataset split (train/dev/test)
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.instances = []
        
    @abstractmethod
    def load_instances(self, limit: Optional[int] = None) -> List[StandardizedInstance]:
        """Load and standardize dataset instances"""
        pass
    
    @abstractmethod
    def get_database_info(self, instance: Dict) -> Dict[str, Any]:
        """Extract database information from instance"""
        pass
    
    @abstractmethod
    def get_schema_info(self, instance: Dict) -> List[Dict[str, Any]]:
        """Extract schema information from instance"""
        pass
    
    def generate_ddl_from_sqlite(self, sqlite_file: str) -> Dict[str, str]:
        """
        Extract CREATE TABLE DDL statements from a SQLite file
        
        Args:
            sqlite_file: Path to the SQLite database file
            
        Returns:
            Dictionary mapping table names to their DDL statements
        """
        if not os.path.exists(sqlite_file):
            raise FileNotFoundError(f"SQLite file not found: {sqlite_file}")
        
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Query sqlite_master table to get all table definitions
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = cursor.fetchall()
        
        ddl_map = {}
        for table_name, ddl in tables:
            if ddl:
                ddl_map[table_name] = ddl
        
        cursor.close()
        conn.close()
        
        return ddl_map
    
    def save_schemas_to_csv(self, ddl_statements: Dict[str, str], output_path: str) -> pd.DataFrame:
        """Save DDL statements to a CSV file"""
        data = {
            'table_name': [],
            'description': [],
            'DDL': []
        }
        
        for table_name, ddl in ddl_statements.items():
            data['table_name'].append(table_name)
            data['description'].append('')  # Can be filled later
            data['DDL'].append(ddl)
        
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        return df
    
    def calculate_difficulty(self, sql_query: str) -> str:
        """
        Calculate difficulty based on SQL complexity
        
        Args:
            sql_query: SQL query string
            
        Returns:
            Difficulty level: 'simple', 'moderate', or 'challenging'
        """
        # Parse the SQL and get all non-whitespace tokens
        sql_tokens = []
        for statement in sqlparse.parse(sql_query):
            sql_tokens.extend([token for token in statement.flatten() if not token.is_whitespace])
        
        if len(sql_tokens) > 160:
            return 'challenging'
        elif len(sql_tokens) > 80:
            return 'moderate'
        else:
            return 'simple'