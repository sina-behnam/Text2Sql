from typing import Optional, Tuple, List, Dict
from src.templates.base import BasePromptTemplate
import re
import json


SYSTEM_PROMPT = """You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question. You must respond in valid JSON format."""

USER_PROMPT_TEMPLATE = """Database Engine:
{dialect}

Database Schema: {schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{evidence} {question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
You must respond with a valid JSON object with the following structure:
{{
    "reasoning": "Step-by-step reasoning process, including self-reflection and corrections if necessary",
    "sql_query": "The final SQL query as a string"
}}"""

class JSONText2SQLTemplate(BasePromptTemplate):
    """Prompt template for JSON-formatted Text2SQL responses."""
    
    def __init__(self):
        super().__init__()
        self.response_format = {"type": "json_object"}
    
    def create_prompt(self, question: str, schema: str, dialect: str,
                     evidence: Optional[str] = None,
                     few_shot_examples: Optional[List[Dict]] = None) -> Tuple[str, str, Dict]:
        """Create JSON-formatted prompt."""
        
        user_message = USER_PROMPT_TEMPLATE.format(
            dialect=dialect,
            schema=schema,
            question=question,
            evidence=evidence if evidence else ""
        )

        return SYSTEM_PROMPT, user_message, self.response_format
    
    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        
        if not response_text or not isinstance(response_text, str):
            return ""
        
        # Try to parse JSON
        sql = self.extract_sql_from_json(response_text)
        if sql and self._is_valid_sql(sql):
            return self._clean_sql(sql)

        
        # remove thinking tags
        extractor = self.SQLExtractorHelper()
        response_text = extractor._remove_thinking_tags(response_text)

        sql_function_set = [extractor._try_json_extraction]
        
        for func in sql_function_set:
            sql = func(response_text)
            if sql and self._is_valid_sql(sql):
                return self._clean_sql(sql)
        
        return ""
    
    def extract_sql_from_json(self,response_text: str) -> str:
        """Extract SQL query from JSON-formatted response."""

        if not response_text or not isinstance(response_text, str):
            return ""

        try:
            # Parse JSON response
            response_json = json.loads(response_text)

            # Extract sql_query field
            sql_query = response_json.get("sql_query", "")

            if not sql_query:
                return ""

            return sql_query.strip()

        except json.JSONDecodeError:
            # Fallback: try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    response_json = json.loads(json_match.group(1))
                    sql_query = response_json.get("sql_query", "")
                    return sql_query.strip()
                except json.JSONDecodeError:
                    pass
                
            # Last resort: try to find sql_query field directly in text
            sql_match = re.search(r'"sql_query"\s*:\s*"([^"]*)"', response_text, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1)
                # Unescape JSON string
                sql_query = sql_query.replace('\\n', '\n').replace('\\"', '"')
                return sql_query.strip()

            return ""

        except Exception as e:
            print(f"Error extracting SQL: {e}")
            return ""
