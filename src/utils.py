import os
import json
import sqlite3
import sqlparse
import pandas as pd
import csv
import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

def check_execution_accuracy_2(
    predicted_sql: str, 
    ground_truth_sql: str,
    db_connection: sqlite3.Connection
) -> Tuple[bool, str]:
    """
    Check if predicted SQL executes correctly and produces the same output as ground truth.
    
    Returns:
        - (True, ""): Results match
        - (False, "error message"): Execution failed with error
        - (False, ""): Execution succeeded but results don't match (needs semantic check)
    
    Args:
        predicted_sql: Predicted SQL query
        ground_truth_sql: Ground truth SQL query
        db_connection: SQLite database connection
        
    Returns:
        Tuple of (is_correct, error_message)
    """
    try:
        # Execute ground truth SQL
        cursor = db_connection.cursor()
        cursor.execute(ground_truth_sql)
        ground_truth_result = cursor.fetchall()
        
        try:
            # Execute predicted SQL
            cursor.execute(predicted_sql)
            predicted_result = cursor.fetchall()
            
            # Quick check: if results are identical
            if predicted_result == ground_truth_result:
                return True, ""
            
            # Check if same data but different order (handles column reordering)
            if len(predicted_result) == len(ground_truth_result):
                if len(predicted_result) == 0:
                    return True, ""  # Both empty
                
                # Check if same number of columns
                if predicted_result and len(predicted_result[0]) == len(ground_truth_result[0]):
                    # Sort values within each row to handle column reordering
                    pred_sorted_rows = set(tuple(sorted(row)) for row in predicted_result)
                    gt_sorted_rows = set(tuple(sorted(row)) for row in ground_truth_result)
                    
                    if pred_sorted_rows == gt_sorted_rows:
                        return True, ""
            
            # Execution succeeded but results don't match
            # Return False with EMPTY error message to trigger semantic analysis
            return False, ""
                
        except Exception as e:
            # Execution failed - return with actual error message
            return False, f"Execution error: {str(e)}"
            
    except Exception as e:
        # Ground truth execution failed - return with error message  
        return False, f"Ground truth execution error: {str(e)}"

def check_execution_accuracy(predicted_sql: str, ground_truth_sql: str, 
                                 db_connection: sqlite3.Connection) -> Tuple[bool, str]:
        """
        Check if predicted SQL executes correctly and produces the same output as ground truth.
        
        Args:
            predicted_sql: Predicted SQL query
            ground_truth_sql: Ground truth SQL query
            db_connection: SQLite database connection
            
        Returns:
            Tuple of (is_correct, error_message)
        """
        try:
            # Execute ground truth SQL
            cursor = db_connection.cursor()
            cursor.execute(ground_truth_sql)
            ground_truth_result = cursor.fetchall()
            
            # Convert to pandas DataFrame for easier comparison
            ground_truth_df = pd.DataFrame(ground_truth_result)
            
            try:
                # Execute predicted SQL
                cursor.execute(predicted_sql)
                predicted_result = cursor.fetchall()

                # Simple check 
                if set(predicted_result) == set(ground_truth_result):
                    return True, ""
                
                # Convert to pandas DataFrame
                predicted_df = pd.DataFrame(predicted_result)
                
                # Check if the results match
                if ground_truth_df.shape == predicted_df.shape:
                    # Sort both dataframes if they have values (not empty)
                    if not ground_truth_df.empty and not predicted_df.empty:
                        # First handle column ordering - reindex both DataFrames with sorted column names
                        # This ensures column order doesn't affect comparison
                        if len(ground_truth_df.columns) > 0:
                            ground_truth_columns = sorted(ground_truth_df.columns)
                            predicted_columns = sorted(predicted_df.columns)
                            
                            # If column sets are different, DataFrames are not equal
                            if set(ground_truth_columns) != set(predicted_columns):
                                return False, "Results have different column sets"
                            
                            # Reindex with sorted columns
                            ground_truth_df = ground_truth_df[ground_truth_columns]
                            predicted_df = predicted_df[predicted_columns]
                        
                        # Now sort by values in each row
                        ground_truth_sorted = ground_truth_df.sort_values(by=list(ground_truth_df.columns)).reset_index(drop=True)
                        predicted_sorted = predicted_df.sort_values(by=list(predicted_df.columns)).reset_index(drop=True)
                        
                        # Check equality
                        return ground_truth_sorted.equals(predicted_sorted), ""
                    else:
                        # If both empty, that's a match
                        return ground_truth_df.empty == predicted_df.empty, ""
                else:
                    return False, f"Results have different shapes: ground truth {ground_truth_df.shape} vs predicted {predicted_df.shape}"
                
            except Exception as e:
                return False, f"Execution error: {str(e)}"
                
        except Exception as e:
            return False, f"Ground truth execution error: {str(e)}"

def create_sql_prompt(question: str, schema, evidence = None) -> Tuple[str, str]:
    """
    Create a prompt for SQL generation.
    
    Args:
        question: Natural language question to generate SQL for
        schema: Database schema information
        evidence: Additional evidence to consider (optional)
        
    Returns:
        Tuple of (system_message, user_message)
    """
    if evidence:
        question = f"{question} \n (PS : {evidence})"
    
    # Format the system message
    system_message = (
        "You are a database expert. "
        "You are supposed to provide a SQL query based on the user's question and the provided database schema. "
        "Your response must be in JSON format with a field named 'sql' containing the generated SQL query. "
        "Example response format: {\"sql\": \"SELECT * FROM table WHERE condition\"}"
    )
    
    # Format the user message
    user_message = f"{question}\n\nHere is the database schema:\n```\n{schema}\n```"
    
    return system_message, user_message

def normalize_sql(sql: str) -> str:
    """
    Normalize SQL query by removing extra spaces and formatting.
    
    Args:
        sql: SQL query string
        
    Returns:
        Normalized SQL query string
    """
    # Use sqlparse to format the SQL query
    parsed = sqlparse.parse(sql)
    
    # Convert parsed SQL back to string
    normalized_sql = sqlparse.format(str(parsed[0]), reindent=True, keyword_case='upper')
    
    # Remove extra spaces
    normalized_sql = re.sub(r'\s+', ' ', normalized_sql).strip()
    
    return normalized_sql
    
def check_exact_match(predicted_sql: str, ground_truth_sql: str) -> bool:
    """
    Check if predicted SQL exactly matches ground truth after normalization.
    
    Args:
        predicted_sql: Predicted SQL query
        ground_truth_sql: Ground truth SQL query
        
    Returns:
        True if exact match, False otherwise
    """
    # Normalize both queries
    normalized_pred = normalize_sql(predicted_sql)
    normalized_gt = normalize_sql(ground_truth_sql)
    
    # Compare normalized queries
    return normalized_pred == normalized_gt

def extract_sql_query_from_text(text: str) -> str:
        """
        Extract SQL query from text, handling various formats (JSON, code blocks, etc.)
        
        Args:
            text: Raw text output that may contain SQL query
            
        Returns:
            Extracted SQL query as a string, or empty string if extraction fails
        """
        # Removing the <think> and </think> tags if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>', '', text)
        # Try to find SQL in JSON format first
        json_match = re.search(r'(\{.*"sql".*\})', text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            try:
                # Try to parse the matched string as JSON
                json_obj = json.loads(json_str)
                if "sql" in json_obj:
                    return json_obj["sql"]
            except json.JSONDecodeError:
                pass
        
        # Try to find SQL in code blocks with ```sql or ```SQL format
        sql_code_block = re.search(r'```(?:sql|SQL)\s*([\s\S]*?)```', text, re.DOTALL)
        if sql_code_block:
            return sql_code_block.group(1).strip()
        
        # Try to find SQL in any code blocks
        any_code_block = re.search(r'```\s*([\s\S]*?)```', text, re.DOTALL)
        if any_code_block:
            code_content = any_code_block.group(1).strip()
            # Check if it looks like SQL (contains SELECT, FROM, etc.)
            if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', code_content, re.IGNORECASE):
                return code_content
        
        # Try to find patterns that look like SQL queries directly in the text
        sql_patterns = [
            # Look for SELECT statement
            r'(?:query:?\s*)?(SELECT\s+[\s\S]*?(?:FROM\s+[\s\S]*?)(?:;|$))',
            # Look for other common SQL statements
            r'(?:query:?\s*)?(INSERT\s+INTO\s+[\s\S]*?(?:;|$))',
            r'(?:query:?\s*)?(UPDATE\s+[\s\S]*?(?:;|$))',
            r'(?:query:?\s*)?(DELETE\s+FROM\s+[\s\S]*?(?:;|$))',
            r'(?:query:?\s*)?(CREATE\s+TABLE\s+[\s\S]*?(?:;|$))'
        ]
        
        for pattern in sql_patterns:
            sql_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
        
        # If we still haven't found a SQL query, look for "The SQL query is:" patterns
        sql_intro_match = re.search(r'(?:The SQL query is:?|Here\'s the SQL:?|Generated SQL:?)\s*([\s\S]*?)(?:\n\n|$)', text, re.DOTALL)
        if sql_intro_match:
            # Get the content after the introduction
            potential_sql = sql_intro_match.group(1).strip()
            # Check if it looks like SQL
            if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', potential_sql, re.IGNORECASE):
                return potential_sql
        
        # No SQL query found
        return ""

def read_json_file(file_path: str):
    """Read JSON file - handles both regular JSON and JSONL."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to load as regular JSON first
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # If that fails, try JSONL format
            f.seek(0)
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
            return data
