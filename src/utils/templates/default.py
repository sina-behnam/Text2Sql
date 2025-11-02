from typing import Optional, Tuple
from src.utils.templates.base import BasePromptTemplate

class DefaultPromptTemplate(BasePromptTemplate):
    """Default prompt template for generic models."""

    def __init__(self):
        super().__init__()

        self.user_message_template = '''
        Question : {full_question}

        Database schema (with {dialect} dialect ) :
        ```
        {schema}
        ```
        '''
        
    def create_prompt(
        self,
        question: str,
        schema: str,
        dialect: str,
        evidence: Optional[str] = None,
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

            Example:

            Question: List the names of all employees hired after 2020.

            Database schema (with sqlite dialect):
            ```
            [
                {'table_name': 'employees', 'description': 'Employee details', 'ddl': 'CREATE TABLE employees (id INT, name VARCHAR, hire_date DATE);'}
            ]
            ```

            Response:
            {
                "sql": "SELECT name FROM employees WHERE hire_date > '2020-12-31'",
                "explanation": "Selects names of employees hired after December 31, 2020"
            }
            """
        )

        user_message = self.user_message_template.format(
            full_question=full_question,
            schema=schema,
            dialect=dialect
        )

        return system_message, user_message, ""

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """
        Generic SQL extraction method that can be used by subclasses.

        This method tries multiple extraction strategies in order:
        1. JSON format
        2. Code blocks (markdown)
        3. XML-style tags
        4. Introductory phrases
        5. Direct SQL pattern matching
        """
        if not response_text or not isinstance(response_text, str):
            return ""
        
        extractor = self.SQLExtractorHelper()

        # Step 1: Remove thinking tags
        response_text = extractor._remove_thinking_tags(response_text)

        sql_function_set = [extractor._try_json_extraction, 
                            extractor._try_code_block_extraction,
                            extractor._try_xml_extraction,
                            extractor._try_intro_phrase_extraction,
                            extractor._try_direct_sql_extraction]
        
        sql = extractor._try_extraction_methods(sql_function_set, response_text)
        if sql and extractor._is_valid_sql(sql):
            return extractor._clean_sql(sql) if clean else sql
    

        return ""
    
    def __str__(self):
        example_system, example_user = self.create_prompt(
            question="List all employees hired after 2020.",
            schema="[{'table_name': 'employees', 'description': 'Employee details', 'ddl': 'CREATE TABLE employees (id INT, name VARCHAR, hire_date DATE);'}]",
            dialect="PostgreSQL"
        )
        return f"DefaultPromptTemplate:\nSystem Message:\n{example_system}\n\nUser Message:\n{example_user}"