"""
LangChain-based Prompt Engineering for Text2SQL

This module provides prompt templates and output parsers using LangChain components.
Simplifies SQL generation and extraction with structured prompts and parsing.
"""

import re
import json
from typing import Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException


class SQLOutputParser(BaseOutputParser[str]):
    """
    Custom LangChain output parser for extracting SQL from model responses.

    Tries multiple extraction strategies:
    1. JSON format with 'sql' field
    2. Code blocks (```sql ... ```)
    3. Direct SQL detection
    """

    def parse(self, text: str) -> str:
        """Parse model output to extract SQL query"""
        if not text or not isinstance(text, str):
            return ""

        # Try different extraction methods
        sql = (
            self._try_json_extraction(text)
            or self._try_code_block_extraction(text)
            or self._try_direct_sql_extraction(text)
        )

        if sql and self._is_valid_sql(sql):
            return self._clean_sql(sql)

        return ""

    @staticmethod
    def _try_json_extraction(text: str) -> str:
        """Extract SQL from JSON format"""
        patterns = [
            r'\{[^{}]*"sql"\s*:\s*"([^"]*(?:\\.[^"]*)*)"[^{}]*\}',
            r'(\{[^{}]*"sql"[^{}]*\})',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    if len(match.groups()) == 1 and not match.group(1).startswith('{'):
                        sql = match.group(1)
                        return sql.replace('\\"', '"').replace('\\n', '\n')
                    else:
                        json_str = match.group(1) if match.group(1).startswith('{') else match.group(0)
                        json_obj = json.loads(json_str)
                        if "sql" in json_obj and json_obj["sql"]:
                            return json_obj["sql"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        return ""

    @staticmethod
    def _try_code_block_extraction(text: str) -> str:
        """Extract SQL from markdown code blocks"""
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
    def _try_direct_sql_extraction(text: str) -> str:
        """Extract SQL directly from text"""
        patterns = [
            r'\b(SELECT\s+(?:DISTINCT\s+)?[\s\S]+?FROM\s+[\s\S]+?)(?:;|\n\n|$)',
            r'\b(WITH\s+[\s\S]+?SELECT\s+[\s\S]+?)(?:;|\n\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(1).strip()
                if len(sql) > 10:
                    return sql
        return ""

    @staticmethod
    def _is_valid_sql(text: str) -> bool:
        """Check if text looks like valid SQL"""
        if not text or len(text.strip()) < 5:
            return False

        text_upper = text.strip().upper()
        sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']
        return any(text_upper.startswith(kw) for kw in sql_keywords)

    @staticmethod
    def _clean_sql(sql: str) -> str:
        """Clean extracted SQL query"""
        if not sql:
            return ""

        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql)
        # Remove trailing semicolon
        sql = sql.strip().rstrip(';').strip()

        return sql


class Text2SQLPromptTemplate:
    """
    LangChain prompt template for Text2SQL generation.

    Creates structured prompts for SQL generation from natural language questions.
    """

    def __init__(self, dialect: str = "SQL"):
        """
        Initialize prompt template.

        Args:
            dialect: SQL dialect (e.g., "SQLite", "PostgreSQL", "MySQL")
        """
        self.dialect = dialect
        self._build_prompt()

    def _build_prompt(self):
        """Build the LangChain prompt template"""
        system_template = """You are an expert SQL query generator. Your task is to convert natural language questions into valid {dialect} queries.

IMPORTANT INSTRUCTIONS:
1. Generate syntactically correct {dialect} queries
2. Use the provided database schema information
3. Consider any additional context or evidence provided
4. Return your response in JSON format with 'sql' and 'explanation' fields

SCHEMA FORMAT:
Each table includes:
- table_name: Name of the table
- description: Description of the table (may be empty)
- ddl: CREATE TABLE statement

RESPONSE FORMAT (JSON):
{{
    "sql": "your SQL query here",
    "explanation": "brief explanation of the query logic"
}}

Example response:
{{
    "sql": "SELECT name FROM employees WHERE age > 30",
    "explanation": "Retrieves names of all employees older than 30"
}}
"""

        human_template = """Question: {question}

Database Schema ({dialect}):
```
{schema}
```
{evidence}

Generate the SQL query in JSON format."""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def format_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None
    ) -> ChatPromptTemplate:
        """
        Format the prompt with the given inputs.

        Args:
            question: Natural language question
            schema: Database schema information
            evidence: Additional context/evidence (optional)

        Returns:
            Formatted ChatPromptTemplate
        """
        evidence_text = f"\nAdditional Context:\n{evidence}" if evidence else ""

        return self.prompt.partial(
            dialect=self.dialect,
            question=question,
            schema=schema,
            evidence=evidence_text
        )


class SemanticEquivalencePromptTemplate:
    """
    LangChain prompt template for checking semantic equivalence of SQL queries.
    """

    def __init__(self):
        """Initialize semantic equivalence prompt template"""
        self._build_prompt()

    def _build_prompt(self):
        """Build the LangChain prompt template"""
        system_template = """You are a SQL expert tasked with determining if two SQL queries are semantically equivalent.

Semantically equivalent means the queries would return the same results on the same database, even if they differ syntactically.

ACCEPTABLE DIFFERENCES:
- Column ordering in SELECT statements
- Presence/absence of column aliases (AS)
- Formatting, spacing, capitalization
- Quote styles around identifiers
- Simple condition reordering (when logically equivalent)

RESPONSE FORMAT (JSON):
{{
    "equivalent": true/false,
    "explanation": "brief explanation of your judgment"
}}
"""

        human_template = """Question: {question}

Ground Truth SQL:
```sql
{ground_truth_sql}
```

Generated SQL:
```sql
{predicted_sql}
```

Are these two SQL queries semantically equivalent? Respond in JSON format."""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def format_prompt(
        self,
        question: str,
        ground_truth_sql: str,
        predicted_sql: str
    ) -> ChatPromptTemplate:
        """
        Format the prompt for semantic equivalence checking.

        Args:
            question: Original natural language question
            ground_truth_sql: Expected SQL query
            predicted_sql: Generated SQL query

        Returns:
            Formatted ChatPromptTemplate
        """
        return self.prompt.partial(
            question=question,
            ground_truth_sql=ground_truth_sql,
            predicted_sql=predicted_sql
        )


def create_sql_generation_chain(llm, dialect: str = "SQL"):
    """
    Create a LangChain chain for SQL generation.

    Args:
        llm: LangChain LLM instance
        dialect: SQL dialect

    Returns:
        Runnable chain: prompt | llm | parser
    """
    from langchain_core.runnables import RunnablePassthrough

    prompt_template = Text2SQLPromptTemplate(dialect)
    parser = SQLOutputParser()

    # Create the chain using LCEL (LangChain Expression Language)
    # This is a simple chain: we'll format the prompt manually and pass to LLM
    return {
        "prompt_template": prompt_template,
        "llm": llm,
        "parser": parser
    }


def create_semantic_equivalence_chain(llm):
    """
    Create a LangChain chain for semantic equivalence checking.

    Args:
        llm: LangChain LLM instance

    Returns:
        Runnable chain: prompt | llm | parser
    """
    prompt_template = SemanticEquivalencePromptTemplate()
    parser = JsonOutputParser()

    return {
        "prompt_template": prompt_template,
        "llm": llm,
        "parser": parser
    }
