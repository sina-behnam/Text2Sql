import os
import json
import sqlite3
import sqlparse
import pandas as pd
import csv
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict

def get_db_connection(instance, instance_path: str = None, snowflake_creds: Dict[str, str] = None):
    """Get database connection based on type"""
    database_info = instance.database
    db_type = database_info.get('type', 'sqlite').lower()
    if db_type == 'sqlite' and instance.dataset != 'spider2-lite':
        db_name = database_info['name']
        db_file = database_info['path'][0].split('/')[-1]  # Get the last part of the path
        database_path = os.path.join(os.path.dirname(instance_path),'databases', db_name,  db_file)
        return sqlite3.connect(database_path), 'sqlite'
    elif db_type == 'sqlite' and instance.dataset == 'spider2-lite':
        db_file = database_info['path'][0]
        database_path = os.path.join(os.path.dirname(instance_path), db_file)
        return sqlite3.connect(database_path), 'sqlite'
    elif db_type == 'snowflake':
        # Load credentials
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError("snowflake-connector-python is not installed. Please install it to use Snowflake databases.")
        conn = snowflake.connector.connect(
            database=database_info['name'],
            **snowflake_creds
        )
        return conn, 'snowflake'
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
def check_sql_semantic_equivalence(model_provider,predicted_sql: str, ground_truth_sql: str, 
                                  question: str,api_key:str) -> Tuple[bool, str]:
        """
        Use the configured model to determine if two SQL queries are semantically equivalent,
        even if they have syntactic differences.
        
        Args:
            model_provider: The model provider instance to use for generation
            predicted_sql: Predicted SQL query
            ground_truth_sql: Ground truth SQL query
            question: The original natural language question
            
        Returns:
            Tuple of (is_equivalent, explanation)
        """
        
        # Create user message
        user_message = (
            f"Question: {question}\n\n"
            f"Gold SQL Query: {ground_truth_sql}\n\n"
            f"Generated SQL Query: {predicted_sql}\n\n"
            "Are these two SQL queries semantically equivalent? Provide your judgment."
        )
        
        try:
            # Generate response using the configured model provider
            raw_response = model_provider.judge(user_message,api_key=api_key)
            
            # Try to extract JSON
            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
            if json_match:
                try:
                    json_obj = json.loads(json_match.group(1))
                    if "equivalent" in json_obj:
                        return json_obj["equivalent"], json_obj.get("explanation", "")
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction fails, look for yes/no in the response
            if re.search(r'\b(yes|equivalent|same|equal)\b', raw_response, re.IGNORECASE):
                return True, "Model indicated equivalence but didn't provide structured output"
            elif re.search(r'\b(no|not equivalent|different|not the same)\b', raw_response, re.IGNORECASE):
                return False, "Model indicated non-equivalence but didn't provide structured output"
            
            # Default to relying on execution results
            return False, "Could not determine semantic equivalence from model response"
        
        except Exception as e:
            # If the model call fails, default to relying on execution results
            return False, f"Error in semantic check: {str(e)}"

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
        
def get_spacy_models():
    '''
    It list out all installed spaCy models with details.
    '''
    import spacy
    import spacy.util

    # Get installed models with more details
    models = spacy.util.get_installed_models()
    for model_name in models:
        try:
            nlp = spacy.load(model_name)
            print(f"Model: {model_name}")
            print(f"Language: {nlp.lang}")
            print(f"Pipeline: {nlp.pipe_names}")
            print("---")
        except Exception as e:
            print(f"Could not load {model_name}: {e}")

def num_tokens(text: str, model : str) -> int:
    """
    Estimate number of tokens by using spacy tokenizer.
    
    Args:
        text: Input text string
    Returns:
        Estimated number of tokens
    """
    import spacy

    try:
        nlp = spacy.load(model)
    except Exception as e:
        raise ValueError(f"Could not load spaCy model '{model}': {e}")
    
    doc = nlp(text)
    return len(doc)