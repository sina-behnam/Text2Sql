import re
import json
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

from collections import defaultdict, namedtuple, Counter, OrderedDict, deque, ChainMap, UserDict, UserList

class BasePromptTemplate(ABC):
    """
    Abstract base class for SQL prompt templates.

    This class defines the interface for creating prompts and extracting SQL
    from model responses. It also includes a helper class for SQL extraction
    using various strategies.
    """
    class SQLExtractorHelper(ABC):
        """Helper class for extracting SQL from model responses using various strategies."""

        @staticmethod
        def _try_extraction_methods(functino_set: List, text: str) -> str:
            """Try multiple extraction methods in order."""
            for func in functino_set:
                sql = func(text)
                if sql:
                    return sql
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

    def __init__(self):
        """
        Initialize the prompt template.
        """
        # to return
        self.system_message = ""
        self.user_message = ""
        self.assistant_message = "" # optional 

    def get_user_message(self) -> str:
        return self.user_message
    
    def get_system_message(self) -> str:
        return self.system_message

    def get_assistant_message(self) -> str:
        return self.assistant_message

    @abstractmethod
    def create_prompt(self, question:str, schema:str, dialect:str, evidence:str = None) -> Tuple[str, str, str]:
        """
        Create a model-specific prompt for SQL generation.
        Args:
            question: Natural language question
            schema: Database schema
            dialect: SQL dialect
            evidence: Optional additional context or evidence

        Returns:
            Tuple of (system_message, user_message) or (system_message, user_message, assistant_message)
        """
        return self.system_message, self.user_message, self.assistant_message

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