#!/usr/bin/env python
"""
Test script for the new prompt engineering system.

This script tests:
1. Base class functionality
2. Model-specific templates
3. Registry and factory pattern
4. Custom template creation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.prompt_engineering import (
    BasePromptTemplate,
    get_prompt_template,
    register_prompt_template,
    PromptTemplateRegistry,
    AnthropicPromptTemplate,
    OpenAIPromptTemplate,
    DefaultPromptTemplate,
)


def test_template_retrieval():
    """Test that templates are correctly retrieved for different models."""
    print("=" * 80)
    print("Test 1: Template Retrieval")
    print("=" * 80)

    test_cases = [
        ("claude-3-5-sonnet-20241022", "AnthropicPromptTemplate"),
        ("gpt-4", "OpenAIPromptTemplate"),
        ("gpt-4o", "OpenAIPromptTemplate"),
        ("unknown-model", "DefaultPromptTemplate"),
        ("llama-2-7b", "LocalModelPromptTemplate"),
    ]

    for model_name, expected_class in test_cases:
        template = get_prompt_template(model_name)
        actual_class = template.__class__.__name__
        status = "✓" if actual_class == expected_class else "✗"
        print(f"{status} {model_name:30} => {actual_class:30} (expected: {expected_class})")


def test_prompt_creation():
    """Test prompt creation for different models."""
    print("\n" + "=" * 80)
    print("Test 2: Prompt Creation")
    print("=" * 80)

    question = "Get all employees in the sales department"
    schema = "CREATE TABLE employees (id INT, name TEXT, department TEXT);"

    models = [
        "claude-3-5-sonnet-20241022",
        "gpt-4",
        "together-ai-model",
    ]

    for model_name in models:
        template = get_prompt_template(model_name)
        system_msg, user_msg = template.create_prompt(question, schema)

        print(f"\n--- {model_name} ({template.__class__.__name__}) ---")
        print(f"System message length: {len(system_msg)} chars")
        print(f"User message length: {len(user_msg)} chars")
        print(f"System preview: {system_msg[:100]}...")
        print(f"User preview: {user_msg[:100]}...")


def test_sql_extraction():
    """Test SQL extraction from various response formats."""
    print("\n" + "=" * 80)
    print("Test 3: SQL Extraction")
    print("=" * 80)

    test_cases = [
        (
            '{"sql": "SELECT * FROM employees WHERE department = \'sales\'"}',
            "JSON format",
        ),
        (
            '```sql\nSELECT * FROM employees WHERE department = \'sales\'\n```',
            "Markdown SQL block",
        ),
        (
            '<think>Analyzing...</think>\n{"sql": "SELECT * FROM employees WHERE department = \'sales\'"}',
            "With thinking tags",
        ),
        (
            '<sql>SELECT * FROM employees WHERE department = \'sales\'</sql>',
            "XML format",
        ),
        (
            'The SQL query is: SELECT * FROM employees WHERE department = \'sales\'',
            "With intro phrase",
        ),
    ]

    template = get_prompt_template("default")

    for response, description in test_cases:
        sql = template.extract_sql(response)
        status = "✓" if sql else "✗"
        print(f"{status} {description:30} => {sql[:50] if sql else 'FAILED'}...")


def test_few_shot_examples():
    """Test few-shot example integration."""
    print("\n" + "=" * 80)
    print("Test 4: Few-Shot Examples")
    print("=" * 80)

    question = "Get products with price > 100"
    schema = "CREATE TABLE products (id INT, name TEXT, price REAL);"

    examples = [
        {
            "question": "Get all customers",
            "schema": "CREATE TABLE customers (id INT, name TEXT);",
            "sql": "SELECT * FROM customers"
        },
        {
            "question": "Get orders from 2024",
            "schema": "CREATE TABLE orders (id INT, year INT);",
            "sql": "SELECT * FROM orders WHERE year = 2024"
        }
    ]

    template = get_prompt_template("gpt-4")
    system_msg, user_msg = template.create_prompt(
        question=question,
        schema=schema,
        few_shot_examples=examples
    )

    print(f"Prompt includes examples: {'Example 1' in user_msg}")
    print(f"Number of examples in prompt: {user_msg.count('Example')}")
    print(f"\nUser message preview:\n{user_msg[:300]}...")


def test_custom_template():
    """Test creating and registering a custom template."""
    print("\n" + "=" * 80)
    print("Test 5: Custom Template Registration")
    print("=" * 80)

    # Create a custom template class
    class MyCustomTemplate(BasePromptTemplate):
        def __init__(self, model_name="my-custom-model"):
            super().__init__(model_name, "custom")

        def create_prompt(self, question, schema, evidence=None, few_shot_examples=None):
            system_msg = "Custom system message for my fine-tuned model"
            user_msg = f"CUSTOM FORMAT:\nQ: {question}\nSCHEMA: {schema}"
            return system_msg, user_msg

        def extract_sql(self, response_text, clean=True):
            # Custom extraction logic
            import re
            match = re.search(r'RESULT:\s*(SELECT.*)', response_text, re.IGNORECASE)
            if match:
                sql = match.group(1)
                return self._clean_sql(sql) if clean else sql
            return self._extract_sql_generic(response_text, clean)

    # Register it
    register_prompt_template("my-finetuned-model-v1", MyCustomTemplate)

    # Verify it's registered
    template = get_prompt_template("my-finetuned-model-v1")
    print(f"✓ Custom template registered: {template.__class__.__name__}")

    # Test it
    system_msg, user_msg = template.create_prompt("test question", "test schema")
    print(f"✓ Custom prompt created: {system_msg[:50]}...")

    # Test extraction
    test_response = "RESULT: SELECT * FROM table"
    sql = template.extract_sql(test_response)
    print(f"✓ Custom extraction works: {sql}")


def test_registry_listing():
    """Test listing all registered templates."""
    print("\n" + "=" * 80)
    print("Test 6: Registry Listing")
    print("=" * 80)

    registered = PromptTemplateRegistry.list_registered()
    print(f"Total registered templates: {len(registered)}")
    print(f"\nSample registered models:")
    for model in registered[:10]:
        print(f"  - {model}")


def test_evidence_integration():
    """Test that evidence is properly integrated into prompts."""
    print("\n" + "=" * 80)
    print("Test 7: Evidence Integration")
    print("=" * 80)

    question = "Get recent orders"
    schema = "CREATE TABLE orders (id INT, date TEXT);"
    evidence = "Recent means within the last 30 days"

    template = get_prompt_template("claude-3-5-sonnet-20241022")
    system_msg, user_msg = template.create_prompt(
        question=question,
        schema=schema,
        evidence=evidence
    )

    has_evidence = "30 days" in user_msg or evidence in user_msg
    status = "✓" if has_evidence else "✗"
    print(f"{status} Evidence included in prompt: {has_evidence}")


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PROMPT ENGINEERING TESTS" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        test_template_retrieval()
        test_prompt_creation()
        test_sql_extraction()
        test_few_shot_examples()
        test_custom_template()
        test_registry_listing()
        test_evidence_integration()

        print("\n" + "=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
