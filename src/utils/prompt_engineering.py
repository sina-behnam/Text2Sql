"""
Prompt Engineering Module for Text2SQL

This module provides a flexible, extensible system for creating model-specific
prompts and extracting SQL queries from model responses.

Design Pattern:
- Base class (BasePromptTemplate) defines the interface
- Specific implementations for different models
- Registry pattern for easy model-to-template mapping
- Factory pattern for retrieving the right template

Usage:
    # Use existing template
    template = get_prompt_template("claude-3-5-sonnet-20241022")
    system_msg, user_msg = template.create_prompt(question, schema)
    sql = template.extract_sql(response)

    # Create custom template for fine-tuned model
    class MyFineTunedTemplate(BasePromptTemplate):
        def create_prompt(self, question, schema, evidence=None, few_shot_examples=None):
            # Custom implementation
            pass

        def extract_sql(self, response_text, clean=True):
            # Custom implementation
            pass

    # Register it
    register_prompt_template("my-finetuned-model", MyFineTunedTemplate)
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


class BasePromptTemplate(ABC):
    """
    Abstract base class for SQL prompt templates.

    This class defines the interface that all prompt templates must implement.
    Extend this class to create custom prompt templates for specific models.

    Attributes:
        model_name: Name of the model this template is designed for
        model_type: General type/family of the model (e.g., "anthropic", "openai")
    """

    def __init__(self, model_name: str = "default", model_type: str = "default"):
        """
        Initialize the prompt template.

        Args:
            model_name: Specific model name (e.g., "claude-3-5-sonnet-20241022")
            model_type: General model type (e.g., "anthropic", "openai")
        """
        self.model_name = model_name
        self.model_type = model_type

    @abstractmethod
    def create_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """
        Create a model-specific prompt for SQL generation.

        Args:
            question: Natural language question to generate SQL for
            schema: Database schema information
            evidence: Additional evidence/context (optional)
            few_shot_examples: List of example dicts with 'question', 'schema', 'sql' keys

        Returns:
            Tuple of (system_message, user_message)
        """
        pass

    @abstractmethod
    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """
        Extract SQL query from model response.

        Args:
            response_text: Raw text output from the model
            clean: Whether to clean the extracted SQL (remove comments, whitespace)

        Returns:
            Extracted SQL query as a string, or empty string if extraction fails
        """
        pass

    @staticmethod
    def _is_valid_sql(text: str) -> bool:
        """
        Check if text looks like a valid SQL query.

        Args:
            text: Text to validate

        Returns:
            True if text appears to be valid SQL
        """
        if not text or not isinstance(text, str):
            return False
        
        if len(text.strip()) < 5:
            return False 

        text_upper = text.strip().upper()

        # Must start with a SQL keyword
        sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
        if not any(text_upper.startswith(kw) for kw in sql_keywords):
            return False

        # For SELECT queries, should have FROM (with some exceptions)
        if text_upper.startswith('SELECT'):
            # Allow SELECT without FROM for simple expressions (e.g., SELECT 1)
            if 'FROM' not in text_upper and len(text) > 50:
                return False

        # Should not contain common non-SQL markers
        non_sql_markers = ['PRINT', 'CONSOLE', 'ECHO', 'RETURN', 'FUNCTION', 'CLASS']
        if any(marker in text_upper for marker in non_sql_markers):
            return False

        return True

    @staticmethod
    def _clean_sql(sql: str) -> str:
        """
        Clean extracted SQL query.

        Args:
            sql: Raw SQL query string

        Returns:
            Cleaned SQL query
        """
        if not sql:
            return ""

        # Remove SQL comments (-- style)
        # sql = re.sub(r'--[^\n]*', '', sql) # for example : SELECT * FROM table -- this is a comment # ! It is better to keep inline comments for clarity, FOR NOW

        # Remove SQL comments (/* */ style)
        # sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL) # for example : SELECT /* comment */ * FROM table # ! It is better to keep inline comments for clarity, FOR NOW

        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql) 

        # Remove leading/trailing whitespace
        sql = sql.strip()

        # Remove trailing semicolon if present
        sql = sql.rstrip(';').strip()

        return sql


class DefaultPromptTemplate(BasePromptTemplate):
    """Default prompt template for generic models."""

    def __init__(self, model_name: str = "default"):
        super().__init__(model_name, "default")

    def create_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
        dialect: str = "SQL",
    ) -> Tuple[str, str]:
        """Create a default prompt."""
        # Combine question with evidence
        full_question = question
        if evidence:
            full_question = f"{question}\n\nAdditional Context: {evidence}"

        system_message = (
            """
            You are a database expert generating SQL queries from natural language. 

                SCHEMA FORMAT:
                Each table is a dictionary with:
                - table_name: string
                - description: string (it can be empty)
                - ddl: CREATE TABLE statement (string)
            
                REQUIREMENTS:
                1. Generate valid SQL for the specified dialect (provide dialect in context)
                2. Return JSON: {"sql": "query_string", "explanation": "brief rationale"}

                EXAMPLE RESPONSE:
                {
                    "sql": "SELECT name FROM employees WHERE age > 30",
                    "explanation": "Selects names of employees older than 30"
                }
            """
        )

        user_message = (
            f"{full_question}\n\n"
            f"Database schema (with {dialect} dialect ) :\n```\n{schema}\n```"
        )

        return system_message, user_message

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """Extract SQL using common patterns."""
        return self._extract_sql_generic(response_text, clean)

    def _extract_sql_generic(self, text: str, clean: bool = True) -> str:
        """
        Generic SQL extraction method that can be used by subclasses.

        This method tries multiple extraction strategies in order:
        1. JSON format
        2. Code blocks (markdown)
        3. XML-style tags
        4. Introductory phrases
        5. Direct SQL pattern matching
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Remove thinking tags
        text = self._remove_thinking_tags(text)

        sql_function_set = [self._try_json_extraction, 
                            self._try_code_block_extraction,
                            self._try_xml_extraction,
                            self._try_intro_phrase_extraction,
                            self._try_direct_sql_extraction]
        
        for func in sql_function_set:
            sql = func(text)
            if sql and self._is_valid_sql(sql):
                return self._clean_sql(sql) if clean else sql

        return ""

    @staticmethod
    def _remove_thinking_tags(text: str) -> str:
        """Remove thinking tags from text."""
        # Remove <think>...</think> tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove orphaned tags
        text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
        # Remove <thinking>...</thinking> tags
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return text

    @staticmethod
    def _try_json_extraction(text: str) -> str:
        """Try to extract SQL from JSON format."""
        json_patterns = [
            r'\{[^{}]*"sql"\s*:\s*"([^"]*(?:\\.[^"]*)*)"[^{}]*\}',
            r'\{[^{}]*\'sql\'\s*:\s*\'([^\']*(?:\\.[^\']*)*)\'[^{}]*\}',
            r'(\{[^{}]*"sql"[^{}]*\})',
        ]

        for pattern in json_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    if len(match.groups()) == 1 and not match.group(1).startswith('{'):
                        # Direct SQL extraction
                        sql = match.group(1)
                        sql = sql.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                        return sql
                    else:
                        # Full JSON object
                        json_str = match.group(1) if match.group(1).startswith('{') else match.group(0)
                        json_obj = json.loads(json_str)
                        if "sql" in json_obj and json_obj["sql"]:
                            return json_obj["sql"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        return ""

    @staticmethod
    def _try_code_block_extraction(text: str) -> str:
        """Try to extract SQL from code blocks."""
        patterns = [
            r'```sql\s*(.*?)```',
            r'```SQL\s*(.*?)```',
            r'```\s*(SELECT[\s\S]*?)```',
            r'```\s*(WITH[\s\S]*?)```',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _try_xml_extraction(text: str) -> str:
        """Try to extract SQL from XML-style tags."""
        patterns = [
            r'<sql>(.*?)</sql>',
            r'<query>(.*?)</query>',
            r'<SQL>(.*?)</SQL>',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _try_intro_phrase_extraction(text: str) -> str:
        """Try to extract SQL after introductory phrases."""
        patterns = [
            r'(?:SQL query|query|SQL|answer):\s*["\']?(SELECT[\s\S]+?)(?:["\']?\s*(?:\n\n|$))',
            r'(?:The SQL (?:query )?is|Here\'s the SQL|Generated SQL):\s*["\']?(SELECT[\s\S]+?)(?:["\']?\s*(?:\n\n|$))',
            r'(?:Result|Output):\s*["\']?(SELECT[\s\S]+?)(?:["\']?\s*(?:\n\n|$))',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().strip('"\'')
        return ""

    @staticmethod
    def _try_direct_sql_extraction(text: str) -> str:
        """Try to extract SQL directly from text."""
        patterns = [
            r'\b(SELECT\s+(?:DISTINCT\s+)?[\s\S]+?FROM\s+[\s\S]+?)(?:;|\n\n|$)',
            r'\b(WITH\s+[\s\S]+?SELECT\s+[\s\S]+?)(?:;|\n\n|$)',
            r'\b(INSERT\s+INTO\s+[\s\S]+?)(?:;|\n\n|$)',
            r'\b(UPDATE\s+[\s\S]+?SET\s+[\s\S]+?)(?:;|\n\n|$)',
            r'\b(DELETE\s+FROM\s+[\s\S]+?)(?:;|\n\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(1).strip()
                if len(sql) > 10:
                    return sql
        return ""


class PromptTemplateRegistry:
    """
    Registry for managing prompt templates.

    This class maintains a mapping of model names to their corresponding
    prompt template classes, allowing for easy lookup and extension.
    """

    _registry: Dict[str, type] = {}
    _default_templates: Dict[str, type] = {
        "default": DefaultPromptTemplate
    } 

    @classmethod
    def register(cls, model_name: str, template_class: type):
        """
        Register a custom prompt template for a model.

        Args:
            model_name: Name or identifier of the model
            template_class: Class that extends BasePromptTemplate

        Raises:
            ValueError: If template_class doesn't extend BasePromptTemplate
        """
        if not issubclass(template_class, BasePromptTemplate):
            raise ValueError(
                f"Template class must extend BasePromptTemplate, got {template_class}"
            )
        cls._registry[model_name] = template_class

    @classmethod
    def get(cls, model_name: str) -> BasePromptTemplate:
        """
        Get a prompt template instance for a model.

        Args:
            model_name: Name of the model

        Returns:
            Instance of the appropriate prompt template class
        """
        # First, check custom registry
        if model_name in cls._registry:
            return cls._registry[model_name](model_name)
        else:
            print(f"No custom template found for model '{model_name}', using default.")
        # Otherwise, return default
        return DefaultPromptTemplate(model_name)

    @classmethod
    def list_registered(cls) -> List[str]:
        """
        List all registered model names.

        Returns:
            List of registered model names
        """
        all_models = set(cls._default_templates.keys()) | set(cls._registry.keys())
        return sorted(list(all_models))


# Convenience functions
def get_prompt_template(model_name: str) -> BasePromptTemplate:
    """
    Get a prompt template instance for a model.

    Args:
        model_name: Name of the model

    Returns:
        Instance of the appropriate prompt template class

    Example:
        >>> template = get_prompt_template("arctic_text2sql_R1")
        >>> system_msg, user_msg = template.create_prompt("Get all users", schema)
    """
    return PromptTemplateRegistry.get(model_name)


def register_prompt_template(model_name: str, template_class: type):
    """
    Register a custom prompt template for a model.

    Args:
        model_name: Name or identifier of the model
        template_class: Class that extends BasePromptTemplate

    Example:
        >>> class MyCustomTemplate(BasePromptTemplate):
        ...     def create_prompt(self, question, schema, evidence=None, few_shot_examples=None):
        ...         # Custom implementation
        ...         pass
        ...     def extract_sql(self, response_text, clean=True):
        ...         # Custom implementation
        ...         pass
        >>> register_prompt_template("my-model", MyCustomTemplate)
    """
    PromptTemplateRegistry.register(model_name, template_class)
