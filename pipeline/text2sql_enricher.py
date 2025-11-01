import os
import re
import glob
import sqlite3
import pandas as pd
from typing import Dict, Tuple, List, Optional, Literal
import sqlparse
import json
from datetime import datetime
import logging
from tqdm import tqdm
import time
from enum import Enum

from src.dataloader import DatasetInstance

from src.utils.utils import (
                        check_exact_match,
                        check_execution_accuracy_2,
                        get_db_connection,
                        check_sql_semantic_equivalence
                        )


from src.utils.templates.base import BasePromptTemplate


from src.models.models import ModelProvider


class PipelineTask(str, Enum):
    """Available pipeline tasks"""
    GENERATE = "generate"
    EXTRACT = "extract"
    EVALUATE = "evaluate"
    SAVE = "save"

# Conditional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for remote logging.")

# make dir logs and remove old logs
if not os.path.exists('./logs'):
    os.makedirs('./logs')

# setup logging and create the logger
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
    Pipeline for Text2SQL Inferencing task: loading data, generating SQL queries, and evaluating results.
    Supports both API-based and local models with optional WandB logging.
    """
    
    def __init__(
        self,
        model_class: ModelProvider,
        prompt_template_class: BasePromptTemplate,
        judge_api_key: str,
        snowflake_config: Dict = None,
        wandb_config: Optional[Dict] = None
    ):
        """
        Initialize the pipeline with dataset paths and model configuration.

        Args:
            model_class: An instance of ModelProvider for SQL generation and evaluation
            prompt_template_class: An instance of BasePromptTemplate for prompt creation and SQL extraction
            judge_api_key: API key for the judge model used in semantic equivalence checking
            snowflake_config: Optional dictionary with Snowflake connection parameters:
                - user: Snowflake username
                - password: Snowflake password
                - account: Snowflake account identifier
                - warehouse: Snowflake warehouse name
            wandb_config: Optional dictionary with WandB configuration:
                - project: Project name (default: 'text2sql-inference')
                - entity: WandB entity/team name
                - name: Run name (default: auto-generated)
                - tags: List of tags for the run
                - notes: Notes about the run
                - enabled: Whether to enable WandB logging (default: True if wandb_config provided)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Text2SQLInferencePipeline...")

        # Use provided config or default
        self.model_provider = model_class

        # Store model info for later use
        self.model_info = {
            'model_name': self.model_provider.model_name,
            'model_config': self.model_provider.config.model_dump()
        }

        # Initialize Snowflake credentials if provided
        self.creds = snowflake_config if snowflake_config else None

        self.judge_api_key = judge_api_key

        # Initialize the prompt template
        if prompt_template_class:
            self.prompt_template = prompt_template_class
        else:
            raise ValueError("A prompt_template_class must be provided to the pipeline. The current available options are ArcticText2SQLTemplate and DefaultPromptTemplate.")

        # Initialize WandB configuration
        self.wandb_config = wandb_config
        self.wandb_enabled = (
            wandb_config is not None and
            wandb_config.get('enabled', True) and
            WANDB_AVAILABLE
        )
        self.wandb_run = None

        if self.wandb_config and not WANDB_AVAILABLE:
            self.logger.warning("WandB configuration provided but wandb is not installed. Logging disabled.")

        # Add this import at the top
        try:
            import snowflake.connector
        except ImportError:
            self.logger.warning("snowflake-connector-python is not installed. Snowflake connections will not work.")

        # Storage for intermediate results
        self.raw_responses: Dict[str, Dict] = {}  # instance_id -> {raw_response, metadata}
        self.extracted_sqls: Dict[str, str] = {}  # instance_id -> extracted_sql
        self.evaluation_results: Dict[str, Dict] = {}  # instance_id -> evaluation    
        
    def _initialize_wandb(self, num_instances: int):
        """Initialize WandB run with configuration."""
        if not self.wandb_enabled:
            return
        
        # Default project name
        project = self.wandb_config.get('project', 'text2sql-inference')
        
        # Auto-generate run name if not provided
        default_name = f"{self.model_info['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_name = self.wandb_config.get('name', default_name)
        
        # Prepare config for wandb
        config = {
            'model': self.model_info,
            'prompt_template': self.prompt_template.__class__.__name__,
            'num_instances': num_instances,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add any additional config from wandb_config
        if 'config' in self.wandb_config:
            config.update(self.wandb_config['config'])
        
        # Initialize wandb run
        self.wandb_run = wandb.init(
            project=project,
            entity=self.wandb_config.get('entity'),
            name=run_name,
            tags=self.wandb_config.get('tags', []),
            notes=self.wandb_config.get('notes'),
            config=config
        )
        
        self.logger.info(f"WandB run initialized: {self.wandb_run.name}")
    
    def _log_instance_result(self, instance: DatasetInstance, result: Dict, inference_time: float):
        """Log individual instance result to WandB."""
        if not self.wandb_enabled or not self.wandb_run:
            return
        
        log_data = {
            'instance_id': instance.id,
            'inference_time': inference_time,
            'has_prediction': result.get('has_prediction', False)
        }
        
        if result.get('has_prediction', False):
            pred = result['predicted_output']
            log_data.update({
                'execution_correct': pred.get('execution_correct', False),
                'exact_match': pred.get('exact_match', False),
                'semantic_equivalent': pred.get('semantic_equivalent', False),
                'has_execution_error': bool(pred.get('execution_error'))
            })
        
        wandb.log(log_data)
    
    def _create_results_table(self, results: List[Dict]) -> Optional[wandb.Table]:
        """Create a WandB table with detailed results."""
        if not self.wandb_enabled or not self.wandb_run:
            return None
        
        columns = [
            'instance_id', 
            'question', 
            'ground_truth_sql', 
            'generated_sql',
            'has_prediction',
            'execution_correct', 
            'exact_match', 
            'semantic_equivalent',
            'execution_error'
        ]
        
        table_data = []
        for r in results:
            instance = r['instance']
            pred = r.get('predicted_output', {})
            
            row = [
                instance.id,
                instance.question[:100] + '...' if len(instance.question) > 100 else instance.question,
                instance.sql[:200] + '...' if len(instance.sql) > 200 else instance.sql,
                (pred.get('generated_sql', '')[:200] + '...') if pred.get('generated_sql') and len(pred.get('generated_sql', '')) > 200 else pred.get('generated_sql', 'N/A'),
                r.get('has_prediction', False),
                pred.get('execution_correct', False),
                pred.get('exact_match', False),
                pred.get('semantic_equivalent', False),
                pred.get('execution_error', '')[:100] if pred.get('execution_error') else ''
            ]
            table_data.append(row)
        
        return wandb.Table(columns=columns, data=table_data)
    
    def _save_raw_responses(self, output_dir: str):
        """Save raw responses to a JSON file."""
        if not output_dir:
            return

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'raw_responses.json')

        with open(output_path, 'w') as f:
            json.dump(self.raw_responses, f, indent=2)

        self.logger.info(f"Saved raw responses to {output_path}")

    def _load_raw_responses(self, output_dir: str) -> bool:
        """Load raw responses from a JSON file. Returns True if successful."""
        if not output_dir:
            return False

        input_path = os.path.join(output_dir, 'raw_responses.json')
        if not os.path.exists(input_path):
            return False

        try:
            with open(input_path, 'r') as f:
                self.raw_responses = json.load(f)
            self.logger.info(f"Loaded {len(self.raw_responses)} raw responses from {input_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load raw responses: {e}")
            return False

    def _save_extracted_sqls(self, output_dir: str):
        """Save extracted SQL queries to a JSON file."""
        if not output_dir:
            return

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'extracted_sqls.json')

        with open(output_path, 'w') as f:
            json.dump(self.extracted_sqls, f, indent=2)

        self.logger.info(f"Saved extracted SQLs to {output_path}")

    def _load_extracted_sqls(self, output_dir: str) -> bool:
        """Load extracted SQL queries from a JSON file. Returns True if successful."""
        if not output_dir:
            return False

        input_path = os.path.join(output_dir, 'extracted_sqls.json')
        if not os.path.exists(input_path):
            return False

        try:
            with open(input_path, 'r') as f:
                self.extracted_sqls = json.load(f)
            self.logger.info(f"Loaded {len(self.extracted_sqls)} extracted SQLs from {input_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load extracted SQLs: {e}")
            return False

    def _save_evaluation_results(self, output_dir: str):
        """Save evaluation results to a JSON file."""
        if not output_dir:
            return

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'evaluation_results.json')

        # Convert evaluation results to serializable format
        serializable_results = {}
        for instance_id, eval_result in self.evaluation_results.items():
            serializable_result = eval_result.copy()
            # Remove the instance object as it's not serializable
            if 'instance' in serializable_result:
                del serializable_result['instance']
            serializable_results[instance_id] = serializable_result

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Saved evaluation results to {output_path}")

    def _load_evaluation_results(self, output_dir: str) -> bool:
        """Load evaluation results from a JSON file. Returns True if successful."""
        if not output_dir:
            return False

        input_path = os.path.join(output_dir, 'evaluation_results.json')
        if not os.path.exists(input_path):
            return False

        try:
            with open(input_path, 'r') as f:
                self.evaluation_results = json.load(f)
            self.logger.info(f"Loaded {len(self.evaluation_results)} evaluation results from {input_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load evaluation results: {e}")
            return False

    def task_generate(self, instances: List[Tuple[DatasetInstance, str]], output_dir: str = None):
        """
        Task 1: Generate raw SQL responses from the model.

        Args:
            instances: List of tuples containing (DatasetInstance, file_path)
            output_dir: Directory to save raw responses
        """
        self.logger.info("Starting GENERATE task...")

        for instance, file_path in tqdm(instances, desc="Generating SQL", unit="instance"):
            start_time = time.time()
            instance_id = str(instance.id)

            # Set up the data required to generate SQL
            question = instance.question
            schema = instance.schemas
            dialect = instance.database['type']
            evidence = instance.evidence

            # Get the prompt messages using the prompt template
            system_message, user_message, assis_message = self.prompt_template.create_prompt(
                question=question,
                schema=schema,
                dialect=dialect,
                evidence=evidence
            )

            # Generate response using the model provider
            try:
                raw_response = self.model_provider.generate(system_message, user_message, assis_message)
                inference_time = time.time() - start_time

                # Store raw response with metadata
                self.raw_responses[instance_id] = {
                    'raw_response': raw_response,
                    'inference_time': inference_time,
                    'question': question,
                    'schema': schema,
                    'dialect': dialect,
                    'evidence': evidence,
                    'file_path': file_path,
                    'has_error': False
                }

                self.logger.info(f"Generated response for instance {instance_id} in {inference_time:.2f}s")

            except Exception as e:
                error_message = f"Model error: {str(e)}"
                self.logger.warning(f"Failed to generate for instance {instance_id}: {error_message}")

                inference_time = time.time() - start_time
                self.raw_responses[instance_id] = {
                    'raw_response': f"Error generating SQL: {error_message}",
                    'inference_time': inference_time,
                    'question': question,
                    'schema': schema,
                    'dialect': dialect,
                    'evidence': evidence,
                    'file_path': file_path,
                    'has_error': True,
                    'error_message': error_message
                }

        # Save raw responses to file
        self._save_raw_responses(output_dir)

        self.logger.info(f"GENERATE task completed. Processed {len(self.raw_responses)} instances")

    def task_extract(self, instances: List[Tuple[DatasetInstance, str]], output_dir: str = None):
        """
        Task 2: Extract SQL queries from raw responses.

        Args:
            instances: List of tuples containing (DatasetInstance, file_path) - used for validation
            output_dir: Directory to load raw responses from and save extracted SQLs
        """
        self.logger.info("Starting EXTRACT task...")

        # Load raw responses if not already in memory
        if not self.raw_responses and output_dir:
            if not self._load_raw_responses(output_dir):
                raise ValueError("No raw responses found. Please run the GENERATE task first.")

        if not self.raw_responses:
            raise ValueError("No raw responses available. Please run the GENERATE task first.")

        # Extract SQL from each raw response
        for instance_id, response_data in tqdm(self.raw_responses.items(), desc="Extracting SQL", unit="instance"):
            if response_data.get('has_error', False):
                self.extracted_sqls[instance_id] = None
                self.logger.info(f"Skipping extraction for instance {instance_id} due to generation error")
                continue

            raw_response = response_data['raw_response']

            # Extract SQL using the prompt template
            extracted_sql = self.prompt_template.extract_sql(raw_response)

            self.extracted_sqls[instance_id] = extracted_sql

            if extracted_sql:
                self.logger.info(f"Successfully extracted SQL for instance {instance_id}")
            else:
                self.logger.warning(f"Failed to extract SQL for instance {instance_id}")

        # Save extracted SQLs to file
        self._save_extracted_sqls(output_dir)

        successful_extractions = sum(1 for sql in self.extracted_sqls.values() if sql is not None)
        self.logger.info(f"EXTRACT task completed. Successfully extracted {successful_extractions}/{len(self.extracted_sqls)} SQLs")

    def task_evaluate(self, instances: List[Tuple[DatasetInstance, str]], output_dir: str = None):
        """
        Task 3: Evaluate extracted SQL queries.

        Args:
            instances: List of tuples containing (DatasetInstance, file_path)
            output_dir: Directory to load extracted SQLs from and save evaluation results
        """
        self.logger.info("Starting EVALUATE task...")

        # Load extracted SQLs if not already in memory
        if not self.extracted_sqls and output_dir:
            if not self._load_extracted_sqls(output_dir):
                raise ValueError("No extracted SQLs found. Please run the EXTRACT task first.")

        if not self.extracted_sqls:
            raise ValueError("No extracted SQLs available. Please run the EXTRACT task first.")

        # Create a mapping from instance_id to (instance, file_path)
        instance_map = {str(inst.id): (inst, path) for inst, path in instances}

        # Evaluate each extracted SQL
        for instance_id, extracted_sql in tqdm(self.extracted_sqls.items(), desc="Evaluating SQL", unit="instance"):
            if instance_id not in instance_map:
                self.logger.warning(f"Instance {instance_id} not found in provided instances")
                continue

            instance, file_path = instance_map[instance_id]

            if extracted_sql is None:
                # No SQL to evaluate
                self.evaluation_results[instance_id] = {
                    'has_prediction': False,
                    'predicted_output': {
                        'generated_sql': None,
                        'execution_correct': False,
                        'execution_error': 'No SQL extracted',
                        'exact_match': False,
                        'semantic_equivalent': False,
                        'semantic_explanation': 'No SQL extracted'
                    }
                }
                self.logger.info(f"Skipping evaluation for instance {instance_id} (no SQL extracted)")
                continue

            # Evaluate the SQL
            evaluation = self.evaluate_instance(instance, extracted_sql, file_path)
            self.evaluation_results[instance_id] = evaluation

            self.logger.info(f"Evaluated instance {instance_id}: "
                           f"exec={evaluation['predicted_output']['execution_correct']}, "
                           f"exact={evaluation['predicted_output']['exact_match']}, "
                           f"semantic={evaluation['predicted_output'].get('semantic_equivalent', False)}")

        # Save evaluation results to file
        self._save_evaluation_results(output_dir)

        successful_evals = sum(1 for r in self.evaluation_results.values() if r.get('has_prediction', False))
        self.logger.info(f"EVALUATE task completed. Evaluated {successful_evals}/{len(self.evaluation_results)} instances")

    def task_save(self, instances: List[Tuple[DatasetInstance, str]], output_dir: str = None):
        """
        Task 4: Save results to instance files.

        Args:
            instances: List of tuples containing (DatasetInstance, file_path)
            output_dir: Directory to save updated instance files
        """
        self.logger.info("Starting SAVE task...")

        # Load evaluation results if not already in memory
        if not self.evaluation_results and output_dir:
            if not self._load_evaluation_results(output_dir):
                raise ValueError("No evaluation results found. Please run the EVALUATE task first.")

        # Also need raw responses for complete data
        if not self.raw_responses and output_dir:
            self._load_raw_responses(output_dir)

        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Please run the EVALUATE task first.")

        # Create a mapping from instance_id to (instance, file_path)
        instance_map = {str(inst.id): (inst, path) for inst, path in instances}

        # Update and save each instance
        for instance_id, eval_result in tqdm(self.evaluation_results.items(), desc="Saving results", unit="instance"):
            if instance_id not in instance_map:
                self.logger.warning(f"Instance {instance_id} not found in provided instances")
                continue

            instance, file_path = instance_map[instance_id]

            # Get raw response if available
            raw_response = ''
            if instance_id in self.raw_responses:
                raw_response = self.raw_responses[instance_id].get('raw_response', '')

            # Update instance with results
            existing_results = instance.inference_results if instance.inference_results else []
            if not isinstance(existing_results, list):
                existing_results = [existing_results]

            result_entry = {
                'has_prediction': eval_result.get('has_prediction', False),
                'model': self.model_info,
                'predicted_output': {
                    'generated_sql': eval_result['predicted_output'].get('generated_sql'),
                    'execution_correct': eval_result['predicted_output'].get('execution_correct', False),
                    'execution_error': eval_result['predicted_output'].get('execution_error', ''),
                    'exact_match': eval_result['predicted_output'].get('exact_match', False),
                    'semantic_equivalent': eval_result['predicted_output'].get('semantic_equivalent', None),
                    'semantic_explanation': eval_result['predicted_output'].get('semantic_explanation', ''),
                    'raw_response': raw_response
                }
            }

            existing_results.append(result_entry)
            instance.inference_results = existing_results

            # Save the updated instance
            self._save_updated_instance(instance, file_path, output_dir)

            self.logger.info(f"Saved results for instance {instance_id}")

        self.logger.info(f"SAVE task completed. Saved {len(self.evaluation_results)} instance files")

    def evaluate_instance(self, instance: DatasetInstance, generated_sql: str, instance_path: str) -> Dict:
        """
        Evaluate the generated SQL query against the ground truth.
        
        Args:
            instance: The DatasetInstance object containing question, schema, and ground truth SQL
            generated_sql: The SQL query generated by the model
            instance_path: Path to the original instance file for database connection
            
        Returns:
            Evaluation results as a dictionary with the full instance data and prediction information
        """
        
        # Get database connection
        db_connection, db_type = get_db_connection(instance, instance_path, self.creds)
        
        # Check execution accuracy
        exec_correct, exec_error = check_execution_accuracy_2(
            generated_sql, instance.sql, db_connection
        )
        
        # Check exact match
        exact_match = check_exact_match(generated_sql, instance.sql)
        
        # If not exact match but execution is correct, or execution failed,
        # check semantic equivalence using the model
        semantic_equivalent = None
        semantic_explanation = None
        
        # Determine semantic equivalence based on exact match, execution correctness, and errors
        if exact_match:
            # If exact match, queries are semantically equivalent
            semantic_equivalent = True
            semantic_explanation = "Exact match found"
        elif exec_correct:
            # If execution is correct but not exact match, consider it semantically equivalent
            semantic_equivalent = True
            semantic_explanation = "Execution correct but not exact match"
        elif exec_error and exec_error.strip():
            # If there's a non-empty execution error, queries are not semantically equivalent
            semantic_equivalent = False
            semantic_explanation = f"Execution failed: {exec_error}"
        else:
            # Otherwise, use the model to check semantic equivalence
            # This catches cases with no exact match, incorrect execution, but no specific error
            semantic_equivalent, semantic_explanation = check_sql_semantic_equivalence(
                self.model_provider, generated_sql, instance.sql, instance.question, self.judge_api_key
            )
        
        # Close database connection
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
        tasks: Optional[List[str]] = None,
        save_updated_files: bool = True,
        output_dir: str = None
    ) -> Dict:
        """
        Run the pipeline with specified tasks.

        Args:
            instances: List of tuples containing (DatasetInstance, file_path)
            tasks: List of tasks to execute in order. Available tasks:
                   - 'generate': Generate raw SQL responses from the model
                   - 'extract': Extract SQL queries from raw responses
                   - 'evaluate': Evaluate extracted SQL queries
                   - 'save': Save results to instance files
                   If None, runs all tasks in order: ['generate', 'extract', 'evaluate', 'save']
            save_updated_files: Whether to save updated JSON files (used if 'save' not in tasks)
            output_dir: Directory to save intermediate and final results

        Returns:
            Evaluation results with comprehensive information
        """
        # Default to running all tasks
        if tasks is None:
            tasks = [PipelineTask.GENERATE, PipelineTask.EXTRACT, PipelineTask.EVALUATE]
            if save_updated_files:
                tasks.append(PipelineTask.SAVE)

        # Convert string tasks to enum if needed
        task_enums = []
        for task in tasks:
            if isinstance(task, str):
                task_enums.append(PipelineTask(task.lower()))
            else:
                task_enums.append(task)

        self.logger.info(f"Running pipeline with tasks: {[t.value for t in task_enums]}")

        # Initialize WandB
        self._initialize_wandb(len(instances))

        # Execute tasks in order
        for task in task_enums:
            if task == PipelineTask.GENERATE:
                self.task_generate(instances, output_dir)

                # Log generation metrics to WandB
                if self.wandb_enabled and self.wandb_run:
                    successful_generations = sum(1 for r in self.raw_responses.values() if not r.get('has_error', False))
                    wandb.log({
                        'task/generate_total': len(self.raw_responses),
                        'task/generate_successful': successful_generations,
                        'task/generate_failed': len(self.raw_responses) - successful_generations
                    })

            elif task == PipelineTask.EXTRACT:
                self.task_extract(instances, output_dir)

                # Log extraction metrics to WandB
                if self.wandb_enabled and self.wandb_run:
                    successful_extractions = sum(1 for sql in self.extracted_sqls.values() if sql is not None)
                    wandb.log({
                        'task/extract_total': len(self.extracted_sqls),
                        'task/extract_successful': successful_extractions,
                        'task/extract_failed': len(self.extracted_sqls) - successful_extractions
                    })

            elif task == PipelineTask.EVALUATE:
                self.task_evaluate(instances, output_dir)

                # Log evaluation metrics to WandB
                if self.wandb_enabled and self.wandb_run:
                    successful_evals = sum(1 for r in self.evaluation_results.values() if r.get('has_prediction', False))
                    wandb.log({
                        'task/evaluate_total': len(self.evaluation_results),
                        'task/evaluate_successful': successful_evals,
                        'task/evaluate_failed': len(self.evaluation_results) - successful_evals
                    })

            elif task == PipelineTask.SAVE:
                self.task_save(instances, output_dir)

        # Calculate overall metrics if evaluation was run
        metrics = {}
        if PipelineTask.EVALUATE in task_enums and self.evaluation_results:
            # Convert evaluation results to the format expected for metrics calculation
            results_for_metrics = []
            for instance_id, eval_result in self.evaluation_results.items():
                results_for_metrics.append(eval_result)

            num_eval = len(results_for_metrics)
            num_with_prediction = sum(1 for r in results_for_metrics if r.get('has_prediction', False))

            # Only consider instances with valid predictions for accuracy metrics
            exec_correct = sum(1 for r in results_for_metrics if r.get('has_prediction', False) and
                               r['predicted_output'].get('execution_correct', False))
            exact_match = sum(1 for r in results_for_metrics if r.get('has_prediction', False) and
                              r['predicted_output'].get('exact_match', False))
            semantic_equivalent = sum(1 for r in results_for_metrics if r.get('has_prediction', False) and
                                     r['predicted_output'].get('semantic_equivalent', False))

            metrics = {
                'num_evaluated': num_eval,
                'num_with_prediction': num_with_prediction,
                'prediction_rate': num_with_prediction / num_eval if num_eval > 0 else 0,
                'execution_accuracy': exec_correct / num_with_prediction if num_with_prediction > 0 else 0,
                'exact_match_accuracy': exact_match / num_with_prediction if num_with_prediction > 0 else 0,
                'semantic_equivalent_accuracy': semantic_equivalent / num_with_prediction if num_with_prediction > 0 else 0,
                'model': self.model_info
            }

            self.logger.info(f"Prediction rate: {metrics['prediction_rate']:.2f}")
            self.logger.info(f"Execution accuracy: {metrics['execution_accuracy']:.2f}")
            self.logger.info(f"Exact match accuracy: {metrics['exact_match_accuracy']:.2f}")
            self.logger.info(f"Semantic equivalence accuracy: {metrics['semantic_equivalent_accuracy']:.2f}")

            # Log final metrics and table to WandB
            if self.wandb_enabled and self.wandb_run:
                # Log summary metrics
                wandb.log({
                    'summary/num_evaluated': num_eval,
                    'summary/num_with_prediction': num_with_prediction,
                    'summary/prediction_rate': metrics['prediction_rate'],
                    'summary/execution_accuracy': metrics['execution_accuracy'],
                    'summary/exact_match_accuracy': metrics['exact_match_accuracy'],
                    'summary/semantic_equivalent_accuracy': metrics['semantic_equivalent_accuracy']
                })

                # Create and log results table
                # Convert evaluation results to format expected by _create_results_table
                instance_map = {str(inst.id): inst for inst, _ in instances}
                results_for_table = []
                for instance_id, eval_result in self.evaluation_results.items():
                    if instance_id in instance_map:
                        result_with_instance = eval_result.copy()
                        result_with_instance['instance'] = instance_map[instance_id]
                        results_for_table.append(result_with_instance)

                results_table = self._create_results_table(results_for_table)
                if results_table:
                    wandb.log({'results_table': results_table})

                # Log the local log file as an artifact
                artifact = wandb.Artifact('pipeline_logs', type='logs')
                artifact.add_file(log_filename)
                self.wandb_run.log_artifact(artifact)

        # Finish WandB run
        if self.wandb_enabled and self.wandb_run:
            self.wandb_run.finish()
            self.logger.info("WandB run finished")

        return metrics
        
    def _save_updated_instance(self, instance: DatasetInstance, original_file_path: str, output_dir: str = None):
        """
        Save the updated instance data back to a JSON file.
        
        Args:
            instance: The updated DatasetInstance object
            original_file_path: Path to the original JSON file
            output_dir: Directory to save the updated file (if None, will update file in place)
        """
        if output_dir:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine the new file path
            file_name = os.path.basename(original_file_path)
            new_file_path = os.path.join(output_dir, file_name)
        else:
            # Update the file in place
            new_file_path = original_file_path
        
        # Convert instance to dictionary and save to file
        instance_dict = instance.to_dict()
        with open(new_file_path, 'w') as f:
            json.dump(instance_dict, f, indent=2)