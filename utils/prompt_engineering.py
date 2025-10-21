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

    def _build_few_shot_examples(self, examples: Optional[List[Dict]]) -> str:
        """
        Helper method to format few-shot examples.

        Args:
            examples: List of example dicts with 'question', 'schema', 'sql' keys

        Returns:
            Formatted examples string
        """
        if not examples:
            return ""

        examples_text = "\n\nHere are some examples:\n\n"
        for i, example in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Question: {example.get('question', '')}\n"
            examples_text += f"Schema: {example.get('schema', '')}\n"
            examples_text += f"SQL: {example.get('sql', '')}\n\n"

        return examples_text

    @staticmethod
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
        sql = re.sub(r'--[^\n]*', '', sql)

        # Remove SQL comments (/* */ style)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

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
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Create a default prompt."""
        # Combine question with evidence
        full_question = question
        if evidence:
            full_question = f"{question}\n\nAdditional Context: {evidence}"

        system_message = (
            "You are a database expert. "
            "Generate a SQL query based on the user's question and the provided database schema. "
            "Your response must be in JSON format with a field named 'sql' containing the generated SQL query. "
            "Example response format: {\"sql\": \"SELECT * FROM table WHERE condition\"}"
        )

        examples_text = self._build_few_shot_examples(few_shot_examples)

        user_message = (
            f"{examples_text}"
            f"{full_question}\n\n"
            f"Database schema:\n```\n{schema}\n```"
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

        # Step 2: Try JSON extraction
        sql = self._try_json_extraction(text)
        if sql and self._is_valid_sql(sql):
            return self._clean_sql(sql) if clean else sql

        # Step 3: Try code blocks
        sql = self._try_code_block_extraction(text)
        if sql and self._is_valid_sql(sql):
            return self._clean_sql(sql) if clean else sql

        # Step 4: Try XML tags
        sql = self._try_xml_extraction(text)
        if sql and self._is_valid_sql(sql):
            return self._clean_sql(sql) if clean else sql

        # Step 5: Try introductory phrases
        sql = self._try_intro_phrase_extraction(text)
        if sql and self._is_valid_sql(sql):
            return self._clean_sql(sql) if clean else sql

        # Step 6: Direct SQL pattern matching
        sql = self._try_direct_sql_extraction(text)
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


class AnthropicPromptTemplate(BasePromptTemplate):
    """Prompt template optimized for Anthropic Claude models."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model_name, "anthropic")

    def create_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Create Anthropic-optimized prompt."""
        full_question = question
        if evidence:
            full_question = f"{question}\n\nAdditional Context: {evidence}"

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

        examples_text = self._build_few_shot_examples(few_shot_examples)

        user_message = (
            f"{examples_text}"
            f"Question: {full_question}\n\n"
            f"Database Schema:\n"
            f"```sql\n{schema}\n```\n\n"
            f"Generate the SQL query in JSON format: {{\"sql\": \"YOUR_QUERY\"}}"
        )

        return system_message, user_message

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """Extract SQL from Anthropic response."""
        # Anthropic typically returns well-formatted JSON
        return self._extract_sql_generic(response_text, clean)


class OpenAIPromptTemplate(BasePromptTemplate):
    """Prompt template optimized for OpenAI GPT models."""

    def __init__(self, model_name: str = "gpt-4"):
        super().__init__(model_name, "openai")

    def create_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Create OpenAI-optimized prompt."""
        full_question = question
        if evidence:
            full_question = f"{question}\n\nAdditional Context: {evidence}"

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

        examples_text = self._build_few_shot_examples(few_shot_examples)

        user_message = (
            f"{examples_text}"
            f"Natural Language Query:\n{full_question}\n\n"
            f"Database Schema:\n```\n{schema}\n```\n\n"
            f"Provide the SQL query as JSON:"
        )

        return system_message, user_message

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """Extract SQL from OpenAI response."""
        return self._extract_sql_generic(response_text, clean)


class TogetherAIPromptTemplate(BasePromptTemplate):
    """Prompt template optimized for Together.ai models."""

    def __init__(self, model_name: str = "together-ai-model"):
        super().__init__(model_name, "together_ai")

    def create_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Create Together.ai-optimized prompt."""
        full_question = question
        if evidence:
            full_question = f"{question}\n\nAdditional Context: {evidence}"

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

        examples_text = self._build_few_shot_examples(few_shot_examples)

        user_message = (
            f"{examples_text}"
            f"Task: Convert this question to SQL\n\n"
            f"Question: {full_question}\n\n"
            f"Schema:\n```\n{schema}\n```\n\n"
            f"Output JSON:"
        )

        return system_message, user_message

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """Extract SQL from Together.ai response."""
        return self._extract_sql_generic(response_text, clean)


class LocalModelPromptTemplate(BasePromptTemplate):
    """Prompt template optimized for local/open-source models."""

    def __init__(self, model_name: str = "local-model"):
        super().__init__(model_name, "local")

    def create_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Create local model-optimized prompt (simpler, more direct)."""
        full_question = question
        if evidence:
            full_question = f"{question}\n\nAdditional Context: {evidence}"

        system_message = (
            "You are a database expert. Generate SQL queries from natural language questions.\n"
            "Output format: {\"sql\": \"YOUR_SQL_QUERY\"}\n"
            "Only output the JSON, nothing else."
        )

        examples_text = self._build_few_shot_examples(few_shot_examples)

        user_message = (
            f"{examples_text}"
            f"Question: {full_question}\n\n"
            f"Schema:\n{schema}\n\n"
            f"SQL (as JSON):"
        )

        return system_message, user_message

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """Extract SQL from local model response."""
        return self._extract_sql_generic(response_text, clean)


class PromptTemplateRegistry:
    """
    Registry for managing prompt templates.

    This class maintains a mapping of model names to their corresponding
    prompt template classes, allowing for easy lookup and extension.
    """

    _registry: Dict[str, type] = {}
    _default_templates: Dict[str, type] = {
        # Anthropic models
        "claude-3-opus-20240229": AnthropicPromptTemplate,
        "claude-3-sonnet-20240229": AnthropicPromptTemplate,
        "claude-3-haiku-20240307": AnthropicPromptTemplate,
        "claude-3-5-sonnet-20241022": AnthropicPromptTemplate,
        "claude-3-5-sonnet-20240620": AnthropicPromptTemplate,
        "claude-3-5-haiku-20241022": AnthropicPromptTemplate,

        # OpenAI models
        "gpt-4": OpenAIPromptTemplate,
        "gpt-4-turbo": OpenAIPromptTemplate,
        "gpt-4o": OpenAIPromptTemplate,
        "gpt-4o-mini": OpenAIPromptTemplate,
        "gpt-3.5-turbo": OpenAIPromptTemplate,

        # Model types (fallback)
        "anthropic": AnthropicPromptTemplate,
        "openai": OpenAIPromptTemplate,
        "together_ai": TogetherAIPromptTemplate,
        "local": LocalModelPromptTemplate,
        "default": DefaultPromptTemplate,
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
    def get(cls, model_name: str, model_type: str = None) -> BasePromptTemplate:
        """
        Get a prompt template instance for a model.

        Args:
            model_name: Name of the model
            model_type: Type of the model (fallback if model_name not found)

        Returns:
            Instance of the appropriate prompt template class
        """
        # First, check custom registry
        if model_name in cls._registry:
            return cls._registry[model_name](model_name)

        # Then, check default templates by model name
        if model_name in cls._default_templates:
            return cls._default_templates[model_name](model_name)

        # Then, try model type
        if model_type and model_type in cls._default_templates:
            return cls._default_templates[model_type](model_name)

        # Check if model name contains known patterns
        model_name_lower = model_name.lower()
        if "claude" in model_name_lower or "anthropic" in model_name_lower:
            return AnthropicPromptTemplate(model_name)
        elif "gpt" in model_name_lower or "openai" in model_name_lower:
            return OpenAIPromptTemplate(model_name)
        elif "llama" in model_name_lower or "mistral" in model_name_lower or "qwen" in model_name_lower:
            return LocalModelPromptTemplate(model_name)

        # Default fallback
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
def get_prompt_template(model_name: str, model_type: str = None) -> BasePromptTemplate:
    """
    Get a prompt template instance for a model.

    Args:
        model_name: Name of the model
        model_type: Type of the model (optional fallback)

    Returns:
        Instance of the appropriate prompt template class

    Example:
        >>> template = get_prompt_template("claude-3-5-sonnet-20241022")
        >>> system_msg, user_msg = template.create_prompt("Get all users", schema)
    """
    return PromptTemplateRegistry.get(model_name, model_type)


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
