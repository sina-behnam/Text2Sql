"""
LangChain-based Text2SQL Inference Pipeline

Enhanced pipeline using LangChain for SQL generation, evaluation, and semantic analysis.
Supports multiple model providers with unified interface.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

# Snowflake support
try:
    import snowflake.connector
    HAS_SNOWFLAKE = True
except ImportError:
    HAS_SNOWFLAKE = False

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage

# Local imports
from src.dataloader import DatasetInstance
from src.models import get_model_provider
from src.utils.utils import check_exact_match, check_execution_accuracy_2
from src.utils.prompt_engineering import (
    SQLOutputParser,
    Text2SQLPromptTemplate,
    SemanticEquivalencePromptTemplate,
    create_sql_generation_chain,
    create_semantic_equivalence_chain
)

# Setup logging
if not os.path.exists('./logs'):
    os.makedirs('./logs')

log_filename = f"./logs/text2sql_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
    ],
    force=True
)


class Text2SQLInferencePipeline:
    """
    LangChain-based pipeline for Text2SQL inference.

    Features:
    - Unified LLM interface via LangChain
    - Structured prompt templates
    - Robust SQL extraction with output parsers
    - Semantic equivalence checking
    - Comprehensive evaluation metrics
    """

    def __init__(
        self,
        model_config: Dict,
        snowflake_config: Optional[Dict] = None,
        sql_dialect: str = "SQLite"
    ):
        """
        Initialize the Text2SQL pipeline.

        Args:
            model_config: Model configuration with keys:
                - type: "openai", "anthropic", "together_ai", or "local"
                - name/path: Model identifier
                - api_key: API key (for API models)
                - temperature, max_tokens, etc.
            snowflake_config: Snowflake connection config (optional)
            sql_dialect: SQL dialect for query generation
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing LangChain-based Text2SQL Pipeline...")

        self.model_config = model_config
        self.snowflake_config = snowflake_config
        self.sql_dialect = sql_dialect

        # Initialize model provider using LangChain
        self.model_provider = get_model_provider(model_config)
        self.llm = self.model_provider.get_llm()

        # Initialize chains
        self.sql_chain = create_sql_generation_chain(self.llm, sql_dialect)
        self.equivalence_chain = create_semantic_equivalence_chain(self.llm)

        # Store model info for tracking
        self.model_info = {
            "model_name": model_config.get("name") or model_config.get("path"),
            "model_type": model_config.get("type"),
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"Initialized {self.model_info['model_type']} model: {self.model_info['model_name']}")

    def generate_sql(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate SQL query using LangChain chain.

        Args:
            question: Natural language question
            schema: Database schema
            evidence: Additional context (optional)

        Returns:
            Tuple of (generated_sql, raw_response)
        """
        try:
            # Get prompt template and parser from chain
            prompt_template = self.sql_chain["prompt_template"]
            parser = self.sql_chain["parser"]

            # Format prompt
            formatted_prompt = prompt_template.format_prompt(
                question=question,
                schema=schema,
                evidence=evidence
            )

            # Invoke LLM
            messages = formatted_prompt.format_messages(
                dialect=self.sql_dialect,
                question=question,
                schema=schema,
                evidence=f"\nAdditional Context:\n{evidence}" if evidence else ""
            )

            response = self.llm.invoke(messages)
            raw_response = response.content

            # Parse SQL from response
            generated_sql = parser.parse(raw_response)

            return generated_sql, raw_response

        except Exception as e:
            self.logger.error(f"Error generating SQL: {e}")
            raise

    def check_semantic_equivalence(
        self,
        predicted_sql: str,
        ground_truth_sql: str,
        question: str
    ) -> Tuple[bool, str]:
        """
        Check if two SQL queries are semantically equivalent using LangChain.

        Args:
            predicted_sql: Generated SQL query
            ground_truth_sql: Ground truth SQL query
            question: Original question

        Returns:
            Tuple of (is_equivalent, explanation)
        """
        try:
            # Get prompt template from chain
            prompt_template = self.equivalence_chain["prompt_template"]

            # Format prompt
            formatted_prompt = prompt_template.format_prompt(
                question=question,
                ground_truth_sql=ground_truth_sql,
                predicted_sql=predicted_sql
            )

            # Invoke LLM
            messages = formatted_prompt.format_messages(
                question=question,
                ground_truth_sql=ground_truth_sql,
                predicted_sql=predicted_sql
            )

            response = self.llm.invoke(messages)
            raw_response = response.content

            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return result.get("equivalent", False), result.get("explanation", "")
                except json.JSONDecodeError:
                    pass

            # Fallback: look for yes/no
            if re.search(r'\b(yes|equivalent|true)\b', raw_response, re.IGNORECASE):
                return True, "Model indicated equivalence"
            elif re.search(r'\b(no|not equivalent|false)\b', raw_response, re.IGNORECASE):
                return False, "Model indicated non-equivalence"

            return False, "Could not parse model response"

        except Exception as e:
            self.logger.error(f"Error checking semantic equivalence: {e}")
            return False, f"Error: {str(e)}"

    def get_db_connection(
        self,
        instance: DatasetInstance,
        instance_path: str = None
    ) -> Tuple:
        """
        Get database connection based on instance configuration.

        Args:
            instance: Dataset instance
            instance_path: Path to instance file

        Returns:
            Tuple of (connection, db_type)
        """
        database_info = instance.database
        db_type = database_info.get('type', 'sqlite').lower()

        if db_type == 'sqlite' and instance.dataset != 'spider2-lite':
            db_name = database_info['name']
            db_file = database_info['path'][0].split('/')[-1]
            database_path = os.path.join(
                os.path.dirname(instance_path),
                'databases',
                db_name,
                db_file
            )
            return sqlite3.connect(database_path), 'sqlite'

        elif db_type == 'sqlite' and instance.dataset == 'spider2-lite':
            db_file = database_info['path'][0]
            database_path = os.path.join(os.path.dirname(instance_path), db_file)
            return sqlite3.connect(database_path), 'sqlite'

        elif db_type == 'snowflake':
            if not HAS_SNOWFLAKE:
                raise ImportError("Install snowflake-connector-python")

            conn = snowflake.connector.connect(
                database=database_info['name'],
                **self.snowflake_config
            )
            return conn, 'snowflake'

        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def evaluate_instance(
        self,
        instance: DatasetInstance,
        generated_sql: str,
        instance_path: str
    ) -> Dict:
        """
        Evaluate generated SQL against ground truth.

        Args:
            instance: Dataset instance
            generated_sql: Generated SQL query
            instance_path: Path to instance file

        Returns:
            Evaluation results dictionary
        """
        # Get database connection
        db_connection, db_type = self.get_db_connection(instance, instance_path)

        # Check execution accuracy
        exec_correct, exec_error = check_execution_accuracy_2(
            generated_sql, instance.sql, db_connection
        )

        # Check exact match
        exact_match = check_exact_match(generated_sql, instance.sql)

        # Determine semantic equivalence
        semantic_equivalent = None
        semantic_explanation = None

        if exact_match:
            semantic_equivalent = True
            semantic_explanation = "Exact match found"
        elif exec_correct:
            semantic_equivalent = True
            semantic_explanation = "Execution correct"
        elif exec_error and exec_error.strip():
            semantic_equivalent = False
            semantic_explanation = f"Execution failed: {exec_error}"
        else:
            # Use LLM to check semantic equivalence
            semantic_equivalent, semantic_explanation = self.check_semantic_equivalence(
                generated_sql, instance.sql, instance.question
            )

        # Close connection
        db_connection.close()

        return {
            'instance': instance,
            'has_prediction': True,
            'predicted_output': {
                'generated_sql': generated_sql,
                'execution_correct': exec_correct,
                'execution_error': exec_error,
                'exact_match': exact_match,
                'semantic_equivalent': semantic_equivalent,
                'semantic_explanation': semantic_explanation
            }
        }

    def run_pipeline(
        self,
        instances: List[Tuple[DatasetInstance, str]],
        save_updated_files: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run the complete inference pipeline.

        Args:
            instances: List of (DatasetInstance, file_path) tuples
            save_updated_files: Whether to save results to files
            output_dir: Output directory (if None, updates in place)

        Returns:
            Evaluation metrics dictionary
        """
        results = []

        # Process each instance
        for instance, file_path in tqdm(instances, desc="Processing instances", unit="instance"):
            self.logger.info(f"Processing instance {instance.id}...")

            # Extract instance data
            question = instance.question
            schema = json.dumps(instance.schemas, indent=2)  # Convert to string
            evidence = instance.evidence

            try:
                # Generate SQL using LangChain
                generated_sql, raw_response = self.generate_sql(
                    question=question,
                    schema=schema,
                    evidence=evidence
                )

                if generated_sql:
                    # Evaluate the generated SQL
                    evaluation = self.evaluate_instance(instance, generated_sql, file_path)
                    evaluation['model'] = self.model_info
                    results.append(evaluation)

                    # Update instance with results
                    existing_results = instance.inference_results if instance.inference_results else []
                    if not isinstance(existing_results, list):
                        existing_results = [existing_results]

                    existing_results.append({
                        'has_prediction': True,
                        'model': self.model_info,
                        'predicted_output': {
                            'generated_sql': generated_sql,
                            'execution_correct': evaluation['predicted_output']['execution_correct'],
                            'execution_error': evaluation['predicted_output']['execution_error'],
                            'exact_match': evaluation['predicted_output']['exact_match'],
                            'semantic_equivalent': evaluation['predicted_output'].get('semantic_equivalent'),
                            'semantic_explanation': evaluation['predicted_output'].get('semantic_explanation', ''),
                            'raw_response': raw_response
                        }
                    })

                    instance.inference_results = existing_results

                    self.logger.info(f"Execution correct: {evaluation['predicted_output']['execution_correct']}")
                    self.logger.info(f"Exact match: {evaluation['predicted_output']['exact_match']}")
                    self.logger.info(f"Semantic equivalent: {evaluation['predicted_output'].get('semantic_equivalent')}")

                else:
                    # Failed to extract SQL
                    self.logger.warning("Failed to extract SQL from model response")
                    results.append({
                        'instance': instance,
                        'has_prediction': False,
                        'predicted_output': {
                            'sql': None,
                            'raw_response': raw_response
                        },
                        'model': self.model_info
                    })

                    instance.inference_results = {
                        'has_prediction': False,
                        'model': self.model_info,
                        'predicted_output': {
                            'sql': None,
                            'raw_response': raw_response
                        }
                    }

            except Exception as e:
                # Handle errors
                error_message = f"Error: {str(e)}"
                self.logger.error(error_message)

                results.append({
                    'instance': instance,
                    'has_prediction': False,
                    'predicted_output': {
                        'sql': None,
                        'error': error_message
                    },
                    'model': self.model_info
                })

                instance.inference_results = {
                    'has_prediction': False,
                    'model': self.model_info,
                    'predicted_output': {
                        'sql': None,
                        'error': error_message
                    }
                }

            # Save updated instance
            if save_updated_files:
                self._save_updated_instance(instance, file_path, output_dir)

            self.logger.info("-" * 50)

        # Calculate metrics
        metrics = self._calculate_metrics(results)
        self.logger.info(f"Prediction rate: {metrics['prediction_rate']:.2f}")
        self.logger.info(f"Execution accuracy: {metrics['execution_accuracy']:.2f}")
        self.logger.info(f"Exact match accuracy: {metrics['exact_match_accuracy']:.2f}")
        self.logger.info(f"Semantic equivalence accuracy: {metrics['semantic_equivalent_accuracy']:.2f}")

        return metrics

    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics from results"""
        num_eval = len(results)
        num_with_prediction = sum(1 for r in results if r.get('has_prediction', False))

        exec_correct = sum(
            1 for r in results
            if r.get('has_prediction', False) and r['predicted_output'].get('execution_correct', False)
        )

        exact_match = sum(
            1 for r in results
            if r.get('has_prediction', False) and r['predicted_output'].get('exact_match', False)
        )

        semantic_equivalent = sum(
            1 for r in results
            if r.get('has_prediction', False) and r['predicted_output'].get('semantic_equivalent', False)
        )

        return {
            'num_evaluated': num_eval,
            'num_with_prediction': num_with_prediction,
            'prediction_rate': num_with_prediction / num_eval if num_eval > 0 else 0,
            'execution_accuracy': exec_correct / num_with_prediction if num_with_prediction > 0 else 0,
            'exact_match_accuracy': exact_match / num_with_prediction if num_with_prediction > 0 else 0,
            'semantic_equivalent_accuracy': semantic_equivalent / num_with_prediction if num_with_prediction > 0 else 0,
            'model': self.model_info
        }

    def _save_updated_instance(
        self,
        instance: DatasetInstance,
        original_file_path: str,
        output_dir: Optional[str] = None
    ):
        """Save updated instance to JSON file"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_name = os.path.basename(original_file_path)
            new_file_path = os.path.join(output_dir, file_name)
        else:
            new_file_path = original_file_path

        instance_dict = instance.to_dict()
        with open(new_file_path, 'w') as f:
            json.dump(instance_dict, f, indent=2)
