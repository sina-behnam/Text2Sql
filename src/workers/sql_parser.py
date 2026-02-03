# ─────────────────────────────────────────────────────────────────
# SQL Extraction with sqlglot validation and timeout
# ─────────────────────────────────────────────────────────────────
import re
import json
from loguru import logger
from typing import Optional, List
import signal
from sqlglot import parse as sqlglot_parse
from sqlglot.errors import ParseError as SqlglotParseError


class ExtractionTimeoutError(Exception):
    """Raised when SQL extraction times out."""
    pass


def _timeout_handler(signum, frame):
    raise ExtractionTimeoutError("SQL extraction timed out")


class SQLExtractor:
    """
    Extract SQL from raw model responses using multiple strategies.

    Uses sqlglot for validation instead of regex.
    Includes timeout protection for long-running extractions.
    """

    EXTRACTION_TIMEOUT = 5  # seconds

    def __init__(self, dialect: str = "sqlite"):
        self.target_dialect = dialect
        self.known_dialects = [
            "sqlite", "mysql", "postgres", "bigquery",
            "snowflake", "mssql"
        ]


    def is_valid_sql(self, sql: str) -> bool:
        """Validate SQL using sqlglot parsing."""
        if not sql or not isinstance(sql, str):
            return False
        sql = sql.strip()
        if len(sql) < 6:  # Minimum: "SELECT"
            return False
        try:
            result = sqlglot_parse(sql, dialect=self.target_dialect)
            return len(result) > 0 and result[0] is not None
        except SqlglotParseError:
            # Try other known dialects as fallback
            for dialect in self.known_dialects:
                if dialect == self.target_dialect:
                    continue
                try:
                    result = sqlglot_parse(sql, dialect=dialect)
                    if len(result) > 0 and result[0] is not None:
                        logger.debug(
                            f"SQL valid for '{dialect}' but not for target '{self.target_dialect}': {sql[:100]}..."
                        )
                        return True
                except SqlglotParseError:
                    continue
            # No dialect could parse it
            logger.debug(f"Invalid SQL - no dialect could parse: {sql[:100]}...")
            return False
        except Exception as e:
            logger.debug(f"Invalid SQL detected: {sql[:100]}... because {e}")
            return False

    def is_valid_sql_2(self, sql: str) -> bool:
        if not sql or not isinstance(sql, str):
            return False
        sql = sql.strip()
        if len(sql) < 6:  # Minimum: "SELECT"
            return False
        return True

    def clean_sql(self, sql: str) -> str:
        """Clean extracted SQL string."""
        if not sql:
            return ""
        # Unescape common escape sequences
        sql = sql.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
        # Remove leading/trailing whitespace and semicolons
        sql = sql.strip().rstrip(';').strip()
        return sql

    def extract(self, raw_response: str, timeout: int = None) -> Optional[str]:
        """
        Extract SQL from raw_response using multiple strategies with validation.

        Tries extraction methods in order, validates each with sqlglot.
        Returns first valid SQL found, or None if all methods fail.

        Args:
            raw_response: Raw model response text
            timeout: Extraction timeout in seconds (default: EXTRACTION_TIMEOUT)

        Returns:
            Extracted SQL string or None
        """
        if not raw_response:
            return None

        timeout = timeout or self.EXTRACTION_TIMEOUT

        # Set up timeout
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

        try:
            # Preprocess: remove thinking tags
            text = self._remove_thinking_tags(raw_response)

            # Extraction methods in priority order
            extraction_methods = [
                self._try_json_extraction,
                self._try_code_block_extraction,
                self._try_xml_extraction,
                self._try_intro_phrase_extraction,
                self._try_direct_sql_extraction,
            ]

            for method in extraction_methods:
                try:
                    candidates = method(text)
                    # Method can return single string or list of candidates
                    if isinstance(candidates, str):
                        candidates = [candidates] if candidates else []

                    for sql in candidates:
                        sql = self.clean_sql(sql)
                        if sql and self.is_valid_sql_2(sql):
                            return sql
                except Exception:
                    # Method failed, try next
                    continue

            return None

        except ExtractionTimeoutError:
            logger.warning(f"SQL extraction timed out after {timeout}s")
            return None
        finally:
            # Cancel timeout
            signal.alarm(0)

    @staticmethod
    def _remove_thinking_tags(text: str) -> str:
        """Remove thinking/reasoning tags from text."""
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<reasoning>.*?</reasoning>',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove orphaned tags
        text = re.sub(r'</?(?:think|thinking|reasoning)>', '', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _try_json_extraction(text: str) -> List[str]:
        """Extract SQL from JSON responses."""
        candidates = []

        # Try full JSON parse first
        try:
            data = json.loads(text)
            for key in ['sql_query', 'sql', 'query', 'SQL', 'generated_sql']:
                if key in data and data[key]:
                    candidates.append(data[key])
        except json.JSONDecodeError:
            pass

        # Try to find JSON objects in text - handle nested braces with balanced matching
        # Find potential JSON start positions
        for i, char in enumerate(text):
            if char == '{':
                # Try to extract balanced JSON object
                depth = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = text[start:j+1]
                            try:
                                obj = json.loads(json_str)
                                if isinstance(obj, dict):
                                    for key in ['sql_query', 'sql', 'query', 'SQL', 'generated_sql']:
                                        if key in obj and obj[key]:
                                            candidates.append(obj[key])
                            except json.JSONDecodeError:
                                pass
                            break

        return candidates

    @staticmethod
    def _try_code_block_extraction(text: str) -> List[str]:
        """Extract SQL from markdown code blocks."""
        candidates = []

        # SQL code blocks FIRST (higher priority than JSON blocks)
        patterns = [
            r'```sql\s*(.*?)```',
            r'```SQL\s*(.*?)```',
            r'```\s*(SELECT\b.*?)```',
            r'```\s*(WITH\b.*?)```',
            r'```\s*(INSERT\b.*?)```',
            r'```\s*(UPDATE\b.*?)```',
            r'```\s*(DELETE\b.*?)```',
            r'```\s*(CREATE\b.*?)```',
            r'```\s*(ALTER\b.*?)```',
            r'```\s*(DROP\b.*?)```',
            r'```\s*(MERGE\b.*?)```',
            r'```\s*(TRUNCATE\b.*?)```',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                candidates.append(match.group(1).strip())

        # Inline backticks (single backticks)
        inline_patterns = [
            r'`((?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b[^`]+)`',
        ]
        for pattern in inline_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidates.append(match.group(1).strip())

        # Handle ```json blocks LAST (lower priority - only if no SQL blocks found)
        json_block_pattern = r'```json\s*(.*?)```'
        for match in re.finditer(json_block_pattern, text, re.DOTALL | re.IGNORECASE):
            json_content = match.group(1).strip()
            try:
                obj = json.loads(json_content)
                if isinstance(obj, dict):
                    for key in ['sql_query', 'sql', 'query', 'SQL', 'generated_sql']:
                        if key in obj and obj[key]:
                            candidates.append(obj[key])
            except json.JSONDecodeError:
                # Fallback: LLMs often output invalid JSON with literal newlines in strings
                sql_keys = ['sql_query', 'sql', 'query', 'generated_sql']
                for key in sql_keys:
                    pattern = rf'"{key}"\s*:\s*"(.*?)"(?:\s*[,}}\]]|\s*$)'
                    sql_match = re.search(pattern, json_content, re.DOTALL | re.IGNORECASE)
                    if sql_match:
                        candidates.append(sql_match.group(1).strip())

        return candidates

    @staticmethod
    def _try_xml_extraction(text: str) -> List[str]:
        """Extract SQL from XML-style tags."""
        candidates = []

        patterns = [
            r'<sql>(.*?)</sql>',
            r'<query>(.*?)</query>',
            r'<SQL>(.*?)</SQL>',
            r'<answer>(.*?)</answer>',
            r'<result>(.*?)</result>',
            r'<sql_query>(.*?)</sql_query>',
            r'<generated_sql>(.*?)</generated_sql>',
            r'<SQL_QUERY>(.*?)</SQL_QUERY>',
            r'<output>(.*?)</output>',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                candidates.append(match.group(1).strip())

        return candidates

    @staticmethod
    def _try_intro_phrase_extraction(text: str) -> List[str]:
        """Extract SQL after common introductory phrases."""
        candidates = []

        # SQL keywords to match after intro phrases
        sql_start = r'(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|MERGE|TRUNCATE)'

        # Match until double newline, semicolon+newline, or end - allows quoted strings inside
        patterns = [
            rf'(?:final\s+)?(?:SQL\s+)?query\s*:\s*[`"\']?({sql_start}\b.*?)(?:[`"\']?\s*(?:\n\n|;\s*\n|\Z))',
            rf'(?:the\s+)?SQL\s+(?:query\s+)?is\s*:\s*[`"\']?({sql_start}\b.*?)(?:[`"\']?\s*(?:\n\n|;\s*\n|\Z))',
            rf'(?:here\'?s?\s+(?:the\s+)?)?(?:generated\s+)?SQL\s*:\s*[`"\']?({sql_start}\b.*?)(?:[`"\']?\s*(?:\n\n|;\s*\n|\Z))',
            rf'answer\s*:\s*[`"\']?({sql_start}\b.*?)(?:[`"\']?\s*(?:\n\n|;\s*\n|\Z))',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                candidates.append(match.group(1).strip())

        return candidates

    @staticmethod
    def _try_direct_sql_extraction(text: str) -> List[str]:
        """Extract SQL statements directly from text."""
        candidates = []

        # Match SQL statements - bounded by semicolon, double newline, or end
        # Using (?:;|\n\n|\Z) as terminator, with lookahead to not consume
        patterns = [
            r'\b(SELECT\s+[\s\S]+?FROM\s+[\s\S]+?)(?=;|\n\n|\Z)',
            r'\b(WITH\s+[\s\S]+?SELECT\s+[\s\S]+?)(?=;|\n\n|\Z)',
            r'\b(INSERT\s+INTO\s+[\s\S]+?)(?=;|\n\n|\Z)',
            r'\b(UPDATE\s+\w+\s+SET\s+[\s\S]+?)(?=;|\n\n|\Z)',
            r'\b(DELETE\s+FROM\s+[\s\S]+?)(?=;|\n\n|\Z)',
            # DDL statements
            r'\b(CREATE\s+(?:TABLE|VIEW|INDEX|DATABASE|SCHEMA)\s+[\s\S]+?)(?=;|\n\n|\Z)',
            r'\b(ALTER\s+(?:TABLE|VIEW|INDEX|DATABASE|SCHEMA)\s+[\s\S]+?)(?=;|\n\n|\Z)',
            r'\b(DROP\s+(?:TABLE|VIEW|INDEX|DATABASE|SCHEMA)\s+[\s\S]+?)(?=;|\n\n|\Z)',
            r'\b(MERGE\s+INTO\s+[\s\S]+?)(?=;|\n\n|\Z)',
            r'\b(TRUNCATE\s+TABLE\s+[\s\S]+?)(?=;|\n\n|\Z)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                sql = match.group(1).strip()
                if len(sql) > 10:
                    candidates.append(sql)

        return candidates
