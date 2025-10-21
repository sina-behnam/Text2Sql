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

def create_sql_prompt(question: str, schema, evidence = None, model_type: str = "default",
                      model_name: str = None, few_shot_examples: List[Dict] = None) -> Tuple[str, str]:
    """
    Create a model-specific prompt for SQL generation with customizable templates.

    Args:
        question: Natural language question to generate SQL for
        schema: Database schema information
        evidence: Additional evidence to consider (optional)
        model_type: Type of model ("openai", "anthropic", "together_ai", "local", or "default")
        model_name: Specific model name for fine-tuned prompt selection
        few_shot_examples: Optional list of example dicts with 'question', 'schema', 'sql' keys

    Returns:
        Tuple of (system_message, user_message)
    """
    # Combine question with evidence if provided
    full_question = question
    if evidence:
        full_question = f"{question}\n\nAdditional Context: {evidence}"

    # Get model-specific prompt template
    system_message, user_template = _get_prompt_template(model_type, model_name)

    # Add few-shot examples if provided
    examples_text = ""
    if few_shot_examples:
        examples_text = "\n\nHere are some examples:\n\n"
        for i, example in enumerate(few_shot_examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Question: {example['question']}\n"
            examples_text += f"Schema: {example['schema']}\n"
            examples_text += f"SQL: {example['sql']}\n\n"

    # Format the user message
    user_message = user_template.format(
        question=full_question,
        schema=schema,
        examples=examples_text
    )

    return system_message, user_message


def _get_prompt_template(model_type: str, model_name: str = None) -> Tuple[str, str]:
    """
    Get model-specific prompt templates for SQL generation.

    Args:
        model_type: Type of model ("openai", "anthropic", "together_ai", "local", or "default")
        model_name: Specific model name for fine-tuned selection

    Returns:
        Tuple of (system_message_template, user_message_template)
    """
    # Anthropic models (Claude) - prefer structured output with thinking
    if model_type == "anthropic":
        system_message = (
            "You are an expert SQL developer with deep knowledge of database design and query optimization. "
            "Your task is to generate accurate SQL queries based on natural language questions and database schemas.\n\n"
            "Instructions:\n"
            "1. Carefully analyze the provided database schema to understand table relationships\n"
            "2. Consider the question's intent and required data\n"
            "3. Generate a syntactically correct SQL query\n"
            "4. Return ONLY the SQL query in a JSON format: {\"sql\": \"YOUR_QUERY_HERE\"}\n"
            "5. Do not include explanations or additional text outside the JSON\n\n"
            "Important:\n"
            "- Use proper JOIN syntax when combining tables\n"
            "- Pay attention to column names and data types\n"
            "- Use appropriate aggregate functions when needed\n"
            "- Ensure proper WHERE clause conditions"
        )
        user_template = (
            "{examples}"
            "Question: {question}\n\n"
            "Database Schema:\n"
            "```sql\n{schema}\n```\n\n"
            "Generate the SQL query in JSON format: {{\"sql\": \"YOUR_QUERY\"}}"
        )

    # OpenAI models - prefer structured output with clear formatting
    elif model_type == "openai":
        system_message = (
            "You are an expert database developer specialized in writing SQL queries. "
            "Given a natural language question and database schema, generate the appropriate SQL query.\n\n"
            "Response format: Return a JSON object with a single 'sql' field containing the query.\n"
            "Example: {\"sql\": \"SELECT column FROM table WHERE condition\"}\n\n"
            "Guidelines:\n"
            "- Write clean, efficient SQL queries\n"
            "- Use proper JOIN syntax for multi-table queries\n"
            "- Apply appropriate WHERE, GROUP BY, and ORDER BY clauses\n"
            "- Use standard SQL syntax compatible with SQLite\n"
            "- Return ONLY the JSON object, no additional text"
        )
        user_template = (
            "{examples}"
            "Natural Language Query:\n{question}\n\n"
            "Database Schema:\n```\n{schema}\n```\n\n"
            "Provide the SQL query as JSON:"
        )

    # Together.ai and other API models
    elif model_type == "together_ai":
        system_message = (
            "You are a SQL query generation expert. Convert natural language questions into SQL queries.\n\n"
            "Rules:\n"
            "1. Analyze the database schema carefully\n"
            "2. Generate syntactically correct SQL\n"
            "3. Return result as JSON: {\"sql\": \"query here\"}\n"
            "4. No explanations, only the JSON output\n\n"
            "SQL Best Practices:\n"
            "- Use explicit JOINs instead of implicit joins\n"
            "- Add appropriate filters in WHERE clause\n"
            "- Use aggregate functions correctly with GROUP BY"
        )
        user_template = (
            "{examples}"
            "Task: Convert this question to SQL\n\n"
            "Question: {question}\n\n"
            "Schema:\n```\n{schema}\n```\n\n"
            "Output JSON:"
        )

    # Local models - simpler, more direct prompts
    elif model_type == "local":
        system_message = (
            "You are a database expert. Generate SQL queries from natural language questions.\n"
            "Output format: {\"sql\": \"YOUR_SQL_QUERY\"}\n"
            "Only output the JSON, nothing else."
        )
        user_template = (
            "{examples}"
            "Question: {question}\n\n"
            "Schema:\n{schema}\n\n"
            "SQL (as JSON):"
        )

    # Default fallback template
    else:
        system_message = (
            "You are a database expert. "
            "Generate a SQL query based on the user's question and the provided database schema. "
            "Your response must be in JSON format with a field named 'sql' containing the generated SQL query. "
            "Example response format: {\"sql\": \"SELECT * FROM table WHERE condition\"}"
        )
        user_template = (
            "{examples}"
            "{question}\n\n"
            "Database schema:\n```\n{schema}\n```"
        )

    return system_message, user_template

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

def extract_sql_query_from_text(text: str, clean: bool = True) -> str:
    """
    Extract SQL query from text, handling various formats (JSON, code blocks, XML tags, etc.)
    with improved robustness and support for multiple model output formats.

    Args:
        text: Raw text output that may contain SQL query
        clean: Whether to clean the extracted SQL (remove comments, extra whitespace)

    Returns:
        Extracted SQL query as a string, or empty string if extraction fails
    """
    if not text or not isinstance(text, str):
        return ""

    original_text = text

    # Step 1: Remove thinking tags (multiple formats)
    # Remove <think>...</think> tags (Anthropic extended thinking)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove orphaned <think> or </think> tags
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    # Remove <thinking>...</thinking> tags (some models use this)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Step 2: Try JSON extraction with multiple approaches
    # Approach 2a: Look for JSON with "sql" field (most common)
    json_patterns = [
        r'\{[^{}]*"sql"\s*:\s*"([^"]*(?:\\.[^"]*)*)"[^{}]*\}',  # Compact JSON
        r'\{[^{}]*\'sql\'\s*:\s*\'([^\']*(?:\\.[^\']*)*)\'[^{}]*\}',  # Single quotes
        r'(\{[^{}]*"sql"[^{}]*\})',  # Full JSON object
    ]

    for pattern in json_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                if len(match.groups()) == 1 and not match.group(1).startswith('{'):
                    # Direct SQL extraction from "sql": "..." pattern
                    sql = match.group(1)
                    # Unescape JSON escapes
                    sql = sql.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                    if _is_valid_sql(sql):
                        return _clean_sql(sql) if clean else sql
                else:
                    # Full JSON object
                    json_str = match.group(1) if match.group(1).startswith('{') else match.group(0)
                    json_obj = json.loads(json_str)
                    if "sql" in json_obj and json_obj["sql"]:
                        sql = json_obj["sql"]
                        if _is_valid_sql(sql):
                            return _clean_sql(sql) if clean else sql
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    # Step 3: Try code blocks (markdown/other formats)
    code_block_patterns = [
        r'```sql\s*(.*?)```',
        r'```SQL\s*(.*?)```',
        r'```\s*(SELECT[\s\S]*?)```',  # Any code block starting with SELECT
        r'```\s*(.*?)```',  # Generic code block
    ]

    for pattern in code_block_patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            sql_candidate = match.group(1).strip()
            if _is_valid_sql(sql_candidate):
                return _clean_sql(sql_candidate) if clean else sql_candidate

    # Step 4: Try XML-style tags
    xml_patterns = [
        r'<sql>(.*?)</sql>',
        r'<query>(.*?)</query>',
        r'<SQL>(.*?)</SQL>',
    ]

    for pattern in xml_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sql_candidate = match.group(1).strip()
            if _is_valid_sql(sql_candidate):
                return _clean_sql(sql_candidate) if clean else sql_candidate

    # Step 5: Look for SQL with introductory phrases
    intro_patterns = [
        r'(?:SQL query|query|SQL|answer):\s*["\']?(SELECT[\s\S]+?)(?:["\']?\s*(?:\n\n|$))',
        r'(?:The SQL (?:query )?is|Here\'s the SQL|Generated SQL):\s*["\']?(SELECT[\s\S]+?)(?:["\']?\s*(?:\n\n|$))',
        r'(?:Result|Output):\s*["\']?(SELECT[\s\S]+?)(?:["\']?\s*(?:\n\n|$))',
    ]

    for pattern in intro_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sql_candidate = match.group(1).strip().strip('"\'')
            if _is_valid_sql(sql_candidate):
                return _clean_sql(sql_candidate) if clean else sql_candidate

    # Step 6: Direct SQL pattern matching (last resort)
    # Look for SQL statements with proper boundaries
    sql_statement_patterns = [
        r'\b(SELECT\s+(?:DISTINCT\s+)?[\s\S]+?FROM\s+[\s\S]+?)(?:;|\n\n|$)',
        r'\b(WITH\s+[\s\S]+?SELECT\s+[\s\S]+?)(?:;|\n\n|$)',  # CTE queries
        r'\b(INSERT\s+INTO\s+[\s\S]+?)(?:;|\n\n|$)',
        r'\b(UPDATE\s+[\s\S]+?SET\s+[\s\S]+?)(?:;|\n\n|$)',
        r'\b(DELETE\s+FROM\s+[\s\S]+?)(?:;|\n\n|$)',
    ]

    for pattern in sql_statement_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sql_candidate = match.group(1).strip()
            # Make sure it's not part of a larger explanation
            if _is_valid_sql(sql_candidate) and len(sql_candidate) > 10:
                return _clean_sql(sql_candidate) if clean else sql_candidate

    # If nothing found, return empty string
    return ""


def _is_valid_sql(text: str) -> bool:
    """
    Check if text looks like a valid SQL query.

    Args:
        text: Text to validate

    Returns:
        True if text appears to be valid SQL
    """
    if not text or len(text.strip()) < 5:
        return False

    text = text.strip().upper()

    # Must start with a SQL keyword
    sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
    if not any(text.startswith(kw) for kw in sql_keywords):
        return False

    # For SELECT queries, should have FROM (with some exceptions)
    if text.startswith('SELECT'):
        # Allow SELECT without FROM for simple expressions (e.g., SELECT 1)
        if 'FROM' not in text and len(text) > 50:
            return False

    # Should not contain common non-SQL markers
    non_sql_markers = ['PRINT', 'CONSOLE', 'ECHO', 'RETURN', 'FUNCTION', 'CLASS']
    if any(marker in text for marker in non_sql_markers):
        return False

    return True


def _clean_sql(sql: str) -> str:
    """
    Clean extracted SQL query by removing comments, extra whitespace, etc.

    Args:
        sql: Raw SQL query string

    Returns:
        Cleaned SQL query
    """
    if not sql:
        return ""

    # Remove SQL comments (-- style)
    sql = re.sub(r'--[^\n]*', '', sql)

    # Remove SQL comments (/* */ style)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

    # Remove extra whitespace
    sql = re.sub(r'\s+', ' ', sql)

    # Remove leading/trailing whitespace
    sql = sql.strip()

    # Remove trailing semicolon if present (some systems don't want it)
    sql = sql.rstrip(';').strip()

    return sql

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