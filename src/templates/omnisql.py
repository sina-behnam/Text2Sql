"""
OmniSQL Prompt Template
"""
from src.templates.base import BasePromptTemplate

input_prompt_template = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}
{evidence}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.'''


class OmniSQLPromptTemplate(BasePromptTemplate):

    def create_prompt(self, question, schema, dialect=None, evidence = None):
        
        usr_msg = input_prompt_template.format(
            db_details = schema,
            question = question,
            evidence = '' if evidence is None else f'(PS :\n{evidence})'
        )

        return '', usr_msg, ''
    
    def extract_sql(self, response_text, clean = True):
        
        extractor = self.SQLExtractorHelper()

        sql = extractor._try_extraction_methods([
            extractor._try_code_block_extraction,
        ],response_text)
        
        if self._is_valid_sql(sql):
            return self._clean_sql(sql)
        
        return None