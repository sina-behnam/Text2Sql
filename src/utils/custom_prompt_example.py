"""
Example of how to create and register custom prompt templates for fine-tuned models.

This file demonstrates:
1. How to extend BasePromptTemplate for a custom/fine-tuned model
2. How to register the custom template
3. How to use the custom template in the pipeline
"""

from typing import Dict, List, Optional, Tuple
from utils.prompt_engineering import BasePromptTemplate, register_prompt_template


# Example 1: Custom prompt for a fine-tuned model with special formatting
class CodeLlamaSQL_FineTunedTemplate(BasePromptTemplate):
    """
    Custom prompt template for a Code Llama model fine-tuned on SQL tasks.

    This model expects a specific format with [INST] tags and prefers
    direct SQL output without JSON wrapping.
    """

    def __init__(self, model_name: str = "codellama-sql-finetuned"):
        super().__init__(model_name, "local")

    def create_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """
        Create a prompt in the format expected by this fine-tuned model.
        """
        full_question = question
        if evidence:
            full_question = f"{question} [CONTEXT: {evidence}]"

        # This model was fine-tuned with a specific system message
        system_message = (
            "You are CodeLlama-SQL, a specialized model for SQL generation. "
            "Generate only the SQL query, no explanations."
        )

        # Build examples in the model's expected format
        examples_text = ""
        if few_shot_examples:
            examples_text = "### EXAMPLES ###\n"
            for ex in few_shot_examples:
                examples_text += f"Q: {ex['question']}\n"
                examples_text += f"SCHEMA: {ex['schema']}\n"
                examples_text += f"SQL: {ex['sql']}\n\n"
            examples_text += "### YOUR TASK ###\n"

        # Format: [INST] instruction [/INST]
        user_message = (
            f"[INST]\n"
            f"{examples_text}"
            f"Generate SQL for:\n"
            f"Question: {full_question}\n\n"
            f"Database Schema:\n{schema}\n"
            f"[/INST]\n"
            f"SQL:"
        )

        return system_message, user_message

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """
        Extract SQL from this model's output format.

        This model outputs SQL directly without JSON wrapping,
        but may include explanation text afterwards.
        """
        if not response_text:
            return ""

        # This model outputs "SQL: <query>" format
        import re

        # Try to find SQL after "SQL:" marker
        sql_match = re.search(r'SQL:\s*(.*?)(?:\n\n|$)', response_text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql = sql_match.group(1).strip()
            if self._is_valid_sql(sql):
                return self._clean_sql(sql) if clean else sql

        # Fallback to generic extraction
        return self._extract_sql_generic(response_text, clean)


# Example 2: Custom prompt for a model that prefers XML output
class XMLFormatModelTemplate(BasePromptTemplate):
    """
    Custom template for a model that was trained to output XML format.
    """

    def __init__(self, model_name: str = "xml-sql-model"):
        super().__init__(model_name, "custom")

    def create_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Create prompt requesting XML output."""
        full_question = question
        if evidence:
            full_question = f"{question}\n\nContext: {evidence}"

        system_message = (
            "You are an AI that generates SQL queries. "
            "Always output in XML format: <response><sql>YOUR_QUERY</sql></response>"
        )

        examples_text = self._build_few_shot_examples(few_shot_examples)

        user_message = (
            f"{examples_text}"
            f"Question: {full_question}\n\n"
            f"Schema:\n{schema}\n\n"
            f"Output the SQL query in XML format:"
        )

        return system_message, user_message

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """Extract SQL from XML format."""
        import re

        # Try XML extraction first
        xml_pattern = r'<sql>(.*?)</sql>'
        match = re.search(xml_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            if self._is_valid_sql(sql):
                return self._clean_sql(sql) if clean else sql

        # Fallback to generic
        return self._extract_sql_generic(response_text, clean)


# Example 3: Custom prompt for a domain-specific model (e.g., healthcare SQL)
class HealthcareSQLTemplate(BasePromptTemplate):
    """
    Template for a model fine-tuned on healthcare database queries.
    Includes domain-specific guidelines and terminology.
    """

    def __init__(self, model_name: str = "healthcare-sql-model"):
        super().__init__(model_name, "domain-specific")

    def create_prompt(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Create healthcare-domain specific prompt."""
        full_question = question
        if evidence:
            full_question = f"{question}\n\nAdditional Context: {evidence}"

        system_message = (
            "You are a healthcare data specialist. Generate SQL queries for medical databases.\n\n"
            "Important Guidelines:\n"
            "- Follow HIPAA compliance: never expose PII without proper filtering\n"
            "- Use proper medical terminology\n"
            "- Consider temporal aspects (admission dates, discharge dates)\n"
            "- Apply appropriate aggregations for patient data\n\n"
            "Output format: {\"sql\": \"YOUR_QUERY\"}"
        )

        examples_text = self._build_few_shot_examples(few_shot_examples)

        user_message = (
            f"{examples_text}"
            f"Medical Query: {full_question}\n\n"
            f"Database Schema:\n{schema}\n\n"
            f"Generate the SQL query as JSON:"
        )

        return system_message, user_message

    def extract_sql(self, response_text: str, clean: bool = True) -> str:
        """Extract SQL from response."""
        return self._extract_sql_generic(response_text, clean)




