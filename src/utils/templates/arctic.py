from typing import Optional, Tuple, List, Dict
from src.utils.templates.base import BasePromptTemplate
import re

SYSTEM_PROMPT = """You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question."""

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
Please provide a detailed chain-of-thought reasoning process and include your thought process within '<think>' tags. Your final answer should be enclosed within '<answer>' tags.
Ensure that your SQL query follows the correct syntax and is formatted as follows:
```sql
-- Your SQL query here
```

Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]
```sql
Correct SQL query here
```
</answer>"""

ASSISTANT_PREFIX = "Let me solve this step by step.\n<think>"

class ArcticText2SQLTemplate(BasePromptTemplate):
    """Prompt template for Snowflake Arctic Text2SQL fine-tuned model."""
    
    def __init__(self):
        super().__init__()
    
    def create_prompt(self, question: str, schema: str, dialect: str,
                     evidence: Optional[str] = None,
                     few_shot_examples: Optional[List[Dict]] = None) -> Tuple[str, str]:
        """Create Arctic-specific prompt format."""
        
        user_message = USER_PROMPT_TEMPLATE.format(
            dialect=dialect,
            schema=schema,
            question=question,
            evidence= evidence if evidence else ""
        )

        return SYSTEM_PROMPT, user_message, ASSISTANT_PREFIX
    
    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        
        if not response_text or not isinstance(response_text, str):
            return ""
        
        # Try Arctic-specific extraction first
        sql = self.arctic_extract_sql(response_text)
        if sql:
            return sql.strip()
        

        # remove thinking tags
        extractor = self.SQLExtractorHelper()
        response_text = extractor._remove_thinking_tags(response_text)

        sql_function_set = [extractor._try_code_block_extraction,
                            extractor._try_direct_sql_extraction]
        
        for func in sql_function_set:
            sql = func(response_text)
            if sql and self._is_valid_sql(sql):
                return self._clean_sql(sql)
            
        
        return ""
    
    def arctic_extract_sql(self, response_text: str) -> str:
        answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1)
            # Extract SQL from code block
            sql_match = re.search(r'```sql\s*(.*?)\s*```', answer_content, re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()

        return None