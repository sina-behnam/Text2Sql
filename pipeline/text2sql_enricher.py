import os
import re
import glob
import sqlite3
import pandas as pd
from typing import Dict, Tuple, List, Optional
import sqlparse
import json
from datetime import datetime
import logging
from tqdm import tqdm
import time

from src.dataloader import DatasetInstance

from src.utils.utils import (
                        check_exact_match,
                        check_execution_accuracy_2,
                        get_db_connection,
                        check_sql_semantic_equivalence
                        )

from src.utils.templates.arctic import ArcticText2SQLTemplate
from src.utils.templates.base import BasePromptTemplate
from src.utils.templates.default import DefaultPromptTemplate

from src.models.models import ModelProvider

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
    
    def run_pipeline(self, instances: List[Tuple[DatasetInstance, str]], save_updated_files: bool = True, 
                    output_dir: str = None) -> Dict:
        """
        Run the complete pipeline: load data, generate SQL, evaluate, and update JSON files.
        
        Args:
            instances: List of tuples containing (DatasetInstance, file_path)
            save_updated_files: Whether to save updated JSON files
            output_dir: Directory to save updated files (if None, will update files in place)
            
        Returns:
            Evaluation results with comprehensive information
        """
        
        # Initialize WandB
        self._initialize_wandb(len(instances))
        
        results = []
        
        # Process each instance
        for instance, file_path in tqdm(instances, desc="Processing instances", unit="instance"):
            start_time = time.time()
            self.logger.info(f"Processing instance {instance.id}...")
            
            # Set up the data that require to generate SQL.
            question = instance.question
            schema = instance.schemas
            dialect = instance.database['type']
            evidence = instance.evidence

            # Generate SQL query
            # Get the prompt messages using the prompt template
            system_message, user_message, assis_message = self.prompt_template.create_prompt(
                question=question,
                schema=schema,
                dialect=dialect,
                evidence=evidence
            )

            # Giving the model provider, we can generate the SQL query.
            try:
                # Generate response using the configured model provider
                raw_response = self.model_provider.generate(system_message, user_message, assis_message)
            except Exception as e:
                # Handle errors
                error_message = f"Model error: {str(e)}"
                self.logger.warning(error_message)
                raw_response = f"Error generating SQL: {error_message} from the model {self.model_info['model_name']}"
                
                # Update instance with error information
                instance.inference_results = {
                    'has_prediction': False,
                    'model': self.model_info,
                    'predicted_output': {
                        'raw_response': raw_response
                    }
                }
                
                result = {
                    'instance': instance,
                    'has_prediction': False,
                    'predicted_output': {
                        'sql': None,
                        'raw_response': raw_response
                    },
                    'model': self.model_info
                }
                results.append(result)
                
                # Log to WandB
                inference_time = time.time() - start_time
                self._log_instance_result(instance, result, inference_time)
                
                self.logger.info("Failed to generate SQL from model response")
                continue

            # Extract SQL query from the raw response using the prompt template
            generated_sql = self.prompt_template.extract_sql(raw_response)
            
            if generated_sql:
                # Evaluate the generated SQL
                evaluation = self.evaluate_instance(instance, generated_sql, file_path)
                # Add model information
                evaluation['model'] = self.model_info
                results.append(evaluation)
                
                # 
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
                        'semantic_equivalent': evaluation['predicted_output'].get('semantic_equivalent', None),
                        'semantic_explanation': evaluation['predicted_output'].get('semantic_explanation', ''),
                        'raw_response': raw_response
                    }
                })

                instance.inference_results = existing_results
                
                # Log to WandB
                inference_time = time.time() - start_time
                self._log_instance_result(instance, evaluation, inference_time)
                
                self.logger.info(f"Execution correct: {evaluation['predicted_output']['execution_correct']}")
                self.logger.info(f"Exact match: {evaluation['predicted_output']['exact_match']}")
                self.logger.info(f"Semantic equivalent: {evaluation['predicted_output'].get('semantic_equivalent', False)}")
            else:
                # Failed to extract SQL
                failed_result = {
                    'instance': instance,
                    'has_prediction': False,
                    'predicted_output': {
                        'sql': None,
                        'raw_response': raw_response
                    },
                    'model': self.model_info
                }
                results.append(failed_result)
                
                # Update the original instance data with failure information
                instance.inference_results = {
                    'has_prediction': False,
                    'model': self.model_info,
                    'predicted_output': {
                        'sql': None,
                        'raw_response': raw_response
                    }
                }
                
                # Log to WandB
                inference_time = time.time() - start_time
                self._log_instance_result(instance, failed_result, inference_time)
                
                self.logger.info("Failed to extract SQL from model response")
                
            # Save the updated instance data to file
            if save_updated_files:
                self._save_updated_instance(instance, file_path, output_dir)
                
            self.logger.info("-" * 50)
        
        # Calculate overall metrics
        num_eval = len(results)
        num_with_prediction = sum(1 for r in results if r.get('has_prediction', False))
        
        # Only consider instances with valid predictions for accuracy metrics
        exec_correct = sum(1 for r in results if r.get('has_prediction', False) and 
                           r['predicted_output'].get('execution_correct', False))
        exact_match = sum(1 for r in results if r.get('has_prediction', False) and 
                          r['predicted_output'].get('exact_match', False))
        semantic_equivalent = sum(1 for r in results if r.get('has_prediction', False) and 
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
            results_table = self._create_results_table(results)
            if results_table:
                wandb.log({'results_table': results_table})
            
            # Log the local log file as an artifact
            artifact = wandb.Artifact('pipeline_logs', type='logs')
            artifact.add_file(log_filename)
            self.wandb_run.log_artifact(artifact)
            
            # Finish the run
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