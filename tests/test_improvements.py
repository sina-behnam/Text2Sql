#!/usr/bin/env python
"""
Simple test script for improved prompting and SQL extraction (no external dependencies)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Test imports only
try:
    from src.utils import _get_prompt_template, _is_valid_sql, _clean_sql
    print("✓ Successfully imported utility functions from src.utils")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

def test_prompt_templates():
    """Test that all model types have prompt templates"""
    print("\n" + "=" * 80)
    print("Testing Prompt Templates")
    print("=" * 80)

    model_types = ["openai", "anthropic", "together_ai", "local", "default"]

    for model_type in model_types:
        try:
            system_msg, user_template = _get_prompt_template(model_type)
            assert isinstance(system_msg, str) and len(system_msg) > 0
            assert isinstance(user_template, str) and len(user_template) > 0
            assert "{question}" in user_template
            assert "{schema}" in user_template
            assert "{examples}" in user_template
            print(f"✓ {model_type:15} - System msg: {len(system_msg):4} chars, User template: {len(user_template):4} chars")
        except Exception as e:
            print(f"✗ {model_type:15} - Error: {e}")

def test_sql_validation():
    """Test SQL validation function"""
    print("\n" + "=" * 80)
    print("Testing SQL Validation")
    print("=" * 80)

    test_cases = [
        ("SELECT * FROM table", True, "Valid SELECT"),
        ("select * from table", True, "Valid SELECT (lowercase)"),
        ("INSERT INTO table VALUES (1)", True, "Valid INSERT"),
        ("UPDATE table SET x=1", True, "Valid UPDATE"),
        ("DELETE FROM table", True, "Valid DELETE"),
        ("WITH cte AS (SELECT * FROM t) SELECT * FROM cte", True, "Valid CTE"),
        ("", False, "Empty string"),
        ("ABC", False, "Not SQL"),
        ("PRINT hello", False, "Non-SQL command"),
        ("SELECT", False, "Incomplete SQL"),
    ]

    for sql, expected, description in test_cases:
        result = _is_valid_sql(sql)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description:30} - Input: '{sql[:40]}...' => {result}")

def test_sql_cleaning():
    """Test SQL cleaning function"""
    print("\n" + "=" * 80)
    print("Testing SQL Cleaning")
    print("=" * 80)

    test_cases = [
        ("SELECT * FROM table;", "SELECT * FROM table", "Remove trailing semicolon"),
        ("SELECT  *   FROM   table", "SELECT * FROM table", "Remove extra spaces"),
        ("SELECT * FROM table -- comment", "SELECT * FROM table", "Remove line comment"),
        ("SELECT * FROM table /* comment */", "SELECT * FROM table", "Remove block comment"),
        ("  SELECT * FROM table  ", "SELECT * FROM table", "Trim whitespace"),
    ]

    for input_sql, expected, description in test_cases:
        result = _clean_sql(input_sql)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description:30}")
        if result != expected:
            print(f"    Expected: '{expected}'")
            print(f"    Got:      '{result}'")

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Improved Text2SQL Prompting & Extraction")
    print("=" * 80)

    test_prompt_templates()
    test_sql_validation()
    test_sql_cleaning()

    print("\n" + "=" * 80)
    print("All basic tests completed!")
    print("=" * 80)
    print("\nNote: Full integration tests require all dependencies installed.")
    print("Run 'pip install -r requirements.txt' to install dependencies.")
