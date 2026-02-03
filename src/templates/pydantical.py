from typing import Optional, Tuple, List, Dict
from pydantic import BaseModel, Field
from src.templates.base import BasePromptTemplate


class Text2SQLResponse(BaseModel):
    """Pydantic model for structured SQL generation response."""
    reasoning: str = Field(description="Step-by-step reasoning process")
    sql_query: str = Field(description="The final SQL query")


SYSTEM_PROMPT = """You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

You must respond with a JSON object matching this exact schema:
{schema}"""

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

Respond with valid JSON only."""


class PydanticText2SQLTemplate(BasePromptTemplate):
    """Prompt template using Pydantic schema-enforced structured outputs."""

    def __init__(self):
        super().__init__()
        self.response_model = Text2SQLResponse
        self.response_format = {
            "type": "json_schema",
            "schema": Text2SQLResponse.model_json_schema(),
        }

    def create_prompt(self, question: str, schema: str, dialect: str,
                     evidence: Optional[str] = None,
                     few_shot_examples: Optional[List[Dict]] = None) -> Tuple[str, str, str]:
        """Create prompt with Pydantic schema in system message.

        Returns:
            Tuple of (system_message, user_message, assistant_prefix).
            Note: response_format is available via self.response_format attribute.
        """
        # Include schema definition in system prompt (recommended by Together.ai)
        response_schema_str = Text2SQLResponse.model_json_schema()
        system_message = SYSTEM_PROMPT.format(schema=response_schema_str)

        user_message = USER_PROMPT_TEMPLATE.format(
            dialect=dialect,
            schema=schema,
            question=question,
            evidence=evidence if evidence else ""
        )

        # Return empty assistant_prefix (no prefill with structured outputs)
        return system_message, user_message, ""

    def parse_response(self, response_text: str) -> Text2SQLResponse:
        """Parse response directly to Pydantic model."""
        if not response_text:
            return Text2SQLResponse(reasoning="", sql_query="")

        try:
            return Text2SQLResponse.model_validate_json(response_text)
        except Exception:
            # Fallback: try to extract SQL manually
            sql = self.extract_sql(response_text)
            return Text2SQLResponse(reasoning="", sql_query=sql)

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """Extract SQL from response (fallback method)."""
        if not response_text or not isinstance(response_text, str):
            return ""

        # Try Pydantic parsing first
        try:
            parsed = Text2SQLResponse.model_validate_json(response_text)
            sql = parsed.sql_query
            if sql and self._is_valid_sql(sql):
                return self._clean_sql(sql) if clean else sql
        except Exception:
            pass

        # Fallback to helper methods
        extractor = self.SQLExtractorHelper()
        response_text = extractor._remove_thinking_tags(response_text)

        for func in [extractor._try_json_extraction,
                     extractor._try_code_block_extraction,
                     extractor._try_direct_sql_extraction]:
            sql = func(response_text)
            if sql and self._is_valid_sql(sql):
                return self._clean_sql(sql) if clean else sql

        return ""
