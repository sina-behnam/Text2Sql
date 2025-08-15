import os
import re
import glob
import sqlite3
import pandas as pd
from typing import Dict, Tuple, List
import sqlparse
import json
from datetime import datetime
import logging
from tqdm import tqdm

# Add this import at the top
try:
    import snowflake.connector
    HAS_SNOWFLAKE = True
except ImportError:
    HAS_SNOWFLAKE = False

from src.dataloader import DatasetInstance
from src.models import (
                    TogetherAIProvider,
                    OpenAIProvider,
                    LocalHuggingFaceProvider,
                    AnthropicProvider)

from src.utils import (extract_sql_query_from_text,
                        check_exact_match,
                        create_sql_prompt,
                        check_execution_accuracy
                        )


class Text2SQLInferencePipeline:
    """
    Pipeline for Text2SQL Inferencing task: loading data, generating SQL queries, and evaluating results.
    Supports both API-based and local models.
    """
    
    def __init__(self, model_config: Dict, snowflake_config: Dict = None):
        """
        Initialize the pipeline with dataset paths and model configuration.
        
        Args:
            snowflake_config: Configuration for Snowflake connection, if applicable.
            model_config: Configuration for the model to use, with keys:
                - "type": "together_ai", "openai", "local", or "anthropic"
                - "name": Model name (for API models) or path (for local models)
                - "api_key": API key (for API models)
                - "device": Device to use for local models ("cpu", "cuda", "auto")
                - "max_new_tokens": Maximum tokens to generate (for local models)
                - "max_tokens": Maximum tokens for Anthropic models
                - "extended_thinking": Whether to use extended thinking for Anthropic
        """
        # setup logging and create the logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Text2SQLInferencePipeline...")
        
        # Use provided config or default
        self.model_config = model_config
        
        # Initialize Snowflake credentials if provided
        self.creds = snowflake_config if snowflake_config else None 
        
        # Initialize the model provider based on config
        self._init_model_provider()

    def _init_model_provider(self):
        """Initialize the model provider based on the configuration"""
        model_type = self.model_config.get("type", "together_ai").lower()
        model_name = self.model_config.get("name")
        model_path = self.model_config.get("path", None)
        extended_thinking = self.model_config.get("extended_thinking", False)
        api_key = self.model_config.get("api_key")
        
        if model_type == "together_ai":
            self.model_provider = TogetherAIProvider(model_name, api_key)
        elif model_type == "openai":
            self.model_provider = OpenAIProvider(model_name, api_key)
        elif model_type == "local":
            max_new_tokens = self.model_config.get("max_new_tokens", 512)
            self.model_provider = LocalHuggingFaceProvider(model_path, "auto", max_new_tokens, extended_thinking=extended_thinking)
        elif model_type == "anthropic":
            max_tokens = self.model_config.get("max_tokens", 1024)
            extended_thinking = self.model_config.get("extended_thinking", False)
            self.model_provider = AnthropicProvider(model_name, api_key, max_tokens, extended_thinking)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Store model info for later use
        self.model_info = {
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def check_sql_semantic_equivalence(self, predicted_sql: str, ground_truth_sql: str, 
                                  question: str) -> Tuple[bool, str]:
        """
        Use the configured model to determine if two SQL queries are semantically equivalent,
        even if they have syntactic differences.
        
        Args:
            predicted_sql: Predicted SQL query
            ground_truth_sql: Ground truth SQL query
            question: The original natural language question
            
        Returns:
            Tuple of (is_equivalent, explanation)
        """
        # Create system message for the judge
        system_message = (
            "You are a SQL expert tasked with determining if two SQL queries are semantically equivalent. "
            "This means they may have syntactic differences but would return the same results when executed "
            "on the same database. Common acceptable differences include: "
            "- Different column ordering in SELECT statements "
            "- Presence or absence of column aliases (AS) "
            "- Different formatting, spacing, or capitalization "
            "- Use of quotes around identifiers "
            "- Simple reordering of conditions that doesn't change the logic "
            "\n\nYour response must be in JSON format with two fields: "
            "'equivalent' (true/false) and 'explanation' (a brief explanation of your judgment)."
        )
        
        # Create user message
        user_message = (
            f"Question: {question}\n\n"
            f"Gold SQL Query: {ground_truth_sql}\n\n"
            f"Generated SQL Query: {predicted_sql}\n\n"
            "Are these two SQL queries semantically equivalent? Provide your judgment."
        )
        
        try:
            # Generate response using the configured model provider
            raw_response = self.model_provider.generate(system_message, user_message)
            
            # Try to extract JSON
            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
            if json_match:
                try:
                    json_obj = json.loads(json_match.group(1))
                    if "equivalent" in json_obj:
                        return json_obj["equivalent"], json_obj.get("explanation", "")
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction fails, look for yes/no in the response
            if re.search(r'\b(yes|equivalent|same|equal)\b', raw_response, re.IGNORECASE):
                return True, "Model indicated equivalence but didn't provide structured output"
            elif re.search(r'\b(no|not equivalent|different|not the same)\b', raw_response, re.IGNORECASE):
                return False, "Model indicated non-equivalence but didn't provide structured output"
            
            # Default to relying on execution results
            return False, "Could not determine semantic equivalence from model response"
        
        except Exception as e:
            # If the model call fails, default to relying on execution results
            return False, f"Error in semantic check: {str(e)}"
                    
    def get_db_connection(self, instance: DatasetInstance, instance_path: str = None):
        """Get database connection based on type"""
        database_info = instance.database
        db_type = database_info.get('type', 'sqlite').lower()

        if db_type == 'sqlite' and instance.dataset != 'spider2-lite':
            db_name = database_info['name']
            db_file = database_info['path'][0].split('/')[-1]  # Get the last part of the path
            database_path = os.path.join(os.path.dirname(instance_path),'databases', db_name,  db_file)
            return sqlite3.connect(database_path), 'sqlite'
        
        elif db_type == 'sqlite' and instance.dataset == 'spider2-lite':
            db_file = database_info['path'][0]
            database_path = os.path.join(os.path.dirname(instance_path), db_file)
            return sqlite3.connect(database_path), 'sqlite'

        elif db_type == 'snowflake':
            if not HAS_SNOWFLAKE:
                raise ImportError("Install snowflake-connector-python")

            # Load credentials
            conn = snowflake.connector.connect(
                database=database_info['name'],
                **self.creds
            )
            return conn, 'snowflake'

        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    

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
        db_connection, db_type = self.get_db_connection(instance, instance_path)
        
        # Check execution accuracy
        exec_correct, exec_error = check_execution_accuracy(
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
            semantic_equivalent, semantic_explanation = self.check_sql_semantic_equivalence(
                generated_sql, instance.sql, instance.question
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
        
        results = []
        
        # Process each instance
        for instance, file_path in tqdm(instances, desc="Processing instances", unit="instance"):
            self.logger.info(f"Processing instance {instance.id}...")
            
            # Set up the data that require to generate SQL.
            question = instance.question
            schema = instance.schemas
            evidence = instance.evidence

            # Generate SQL query
            # Get the prompt messages
            system_message, user_message = create_sql_prompt(question, schema, evidence)

            # Giving the model provider, we can generate the SQL query.
            try:
                # Generate response using the configured model provider
                raw_response = self.model_provider.generate(system_message, user_message)
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
                
                results.append({
                    'instance': instance,
                    'has_prediction': False,
                    'predicted_output': {
                        'sql': None,
                        'raw_response': raw_response
                    },
                    'model': self.model_info
                })
                self.logger.info("Failed to generate SQL from model response")
                continue

            
            # Extract SQL query from the raw response using the enhanced function
            generated_sql = extract_sql_query_from_text(raw_response)
            
            if generated_sql:
                # Evaluate the generated SQL
                evaluation = self.evaluate_instance(instance, generated_sql, file_path)
                # Add model information
                evaluation['model'] = self.model_info
                results.append(evaluation)
                
                # Update the original instance data with inference results
                instance.inference_results = {
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
                }
                
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