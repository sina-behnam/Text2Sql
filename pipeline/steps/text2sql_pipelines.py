"""
Concrete pipeline steps for Text2SQL tasks.

This module provides ready-to-use pipeline steps for common Text2SQL operations.
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from pipeline.steps.base import PipelineStep, PipelineContext
from src.dataloader import DatasetInstance
from src.models.models import ModelProvider
from src.utils.templates.base import BasePromptTemplate
from src.utils.utils import (
    check_exact_match,
    check_execution_accuracy_general,
    get_db_path,
    check_sql_semantic_equivalence
)


class GenerateStep(PipelineStep):
    """
    Generate SQL queries using a model provider.

    This step takes DatasetInstances and generates raw SQL responses
    using the configured model and prompt template.

    Attributes:
        model_provider: Model provider instance for SQL generation
        prompt_template: Prompt template for formatting inputs
        save_intermediate: Whether to save raw responses to disk
    """

    def __init__(self,
                 model_provider: ModelProvider,
                 prompt_template: BasePromptTemplate,
                 save_intermediate: bool = True,
                 name: Optional[str] = None):
        """
        Initialize the GenerateStep.

        Args:
            model_provider: Instance of ModelProvider for SQL generation
            prompt_template: Instance of BasePromptTemplate for prompt creation
            save_intermediate: Whether to save raw responses to disk
            name: Optional custom name for the step
        """
        super().__init__(name or "generate")
        self.model_provider = model_provider
        self.prompt_template = prompt_template
        self.save_intermediate = save_intermediate

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Generate SQL queries for all instances"""

        self.logger.info(f"Generating SQL for {len(instances)} instances...")

        for instance, file_path in tqdm(instances, desc="Generating SQL", unit="instance"):
            start_time = time.time()
            instance_id = str(instance.id)

            # Prepare input data
            question = instance.question
            schema = instance.schemas
            dialect = instance.database['type']
            evidence = instance.evidence

            # Create prompt using template
            system_message, user_message, assis_message = self.prompt_template.create_prompt(
                question=question,
                schema=schema,
                dialect=dialect,
                evidence=evidence
            )

            # Generate response
            try:
                raw_response = self.model_provider.generate(
                    system_message, user_message, assis_message
                )
                inference_time = time.time() - start_time

                # Store raw response with metadata
                context.raw_responses[instance_id] = {
                    'dataset': instance.dataset,
                    'database': instance.database.get('name', ''),
                    'gold_sql': instance.sql,
                    'raw_response': raw_response,
                    'inference_time': inference_time,
                    'question': question,
                    'evidence': evidence,
                    'file_path': file_path,
                    'has_error': False
                }

                self.logger.debug(f"Generated response for {instance_id} in {inference_time:.2f}s")

            except Exception as e:
                error_message = f"Model error: {str(e)}"
                self.logger.warning(f"Failed to generate for {instance_id}: {error_message}")

                inference_time = time.time() - start_time
                context.raw_responses[instance_id] = {
                    'dataset': instance.dataset,
                    'database': instance.database.get('name', ''),
                    'gold_sql': instance.sql,
                    'raw_response': f"Error generating SQL: {error_message}",
                    'inference_time': inference_time,
                    'question': question,
                    'evidence': evidence,
                    'file_path': file_path,
                    'has_error': True,
                    'error_message': error_message
                }

        # Save intermediate results if requested
        if self.save_intermediate and context.output_dir:
            self._save_raw_responses(context)

        self.logger.info(f"Generated {len(context.raw_responses)} responses")
        return context

    def _save_raw_responses(self, context: PipelineContext) -> None:
        """Save raw responses to JSON file"""
        if not context.output_dir:
            return

        os.makedirs(context.output_dir, exist_ok=True)
        output_path = os.path.join(context.output_dir, 'raw_responses.json')

        with open(output_path, 'w') as f:
            json.dump(context.raw_responses, f, indent=2)

        self.logger.info(f"Saved raw responses to {output_path}")


class ExtractStep(PipelineStep):
    """
    Extract SQL queries from raw model responses.

    This step takes raw responses and extracts clean SQL queries
    using the configured prompt template's extraction logic.

    Attributes:
        prompt_template: Prompt template with SQL extraction logic
        save_intermediate: Whether to save extracted SQLs to disk
    """

    def __init__(self,
                 prompt_template: BasePromptTemplate,
                 save_intermediate: bool = True,
                 name: Optional[str] = None):
        """
        Initialize the ExtractStep.

        Args:
            prompt_template: Instance of BasePromptTemplate for SQL extraction
            save_intermediate: Whether to save extracted SQLs to disk
            name: Optional custom name for the step
        """
        super().__init__(name or "extract")
        self.prompt_template = prompt_template
        self.save_intermediate = save_intermediate

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Extract SQL queries from raw responses"""

        # Check if we have raw responses
        if not context.has_data('raw_responses'):
            raise ValueError("No raw responses found. Run GenerateStep first or load from disk.")

        self.logger.info(f"Extracting SQL from {len(context.raw_responses)} responses...")

        for instance_id, response_data in tqdm(
            context.raw_responses.items(),
            desc="Extracting SQL",
            unit="instance"
        ):
            # Skip if generation failed
            if response_data.get('has_error', False):
                context.extracted_sqls[instance_id] = None
                self.logger.debug(f"Skipping {instance_id} (generation error)")
                continue

            raw_response = response_data['raw_response']

            # Extract SQL using template
            extracted_sql = self.prompt_template.extract_sql(raw_response)
            context.extracted_sqls[instance_id] = extracted_sql

            if extracted_sql:
                self.logger.debug(f"Successfully extracted SQL for {instance_id}")
            else:
                self.logger.warning(f"Failed to extract SQL for {instance_id}")

        # Save intermediate results if requested
        if self.save_intermediate and context.output_dir:
            self._save_extracted_sqls(context)

        successful = sum(1 for sql in context.extracted_sqls.values() if sql is not None)
        self.logger.info(f"Extracted {successful}/{len(context.extracted_sqls)} SQLs")

        return context

    def _save_extracted_sqls(self, context: PipelineContext) -> None:
        """Save extracted SQLs to JSON file"""
        if not context.output_dir:
            return

        os.makedirs(context.output_dir, exist_ok=True)
        output_path = os.path.join(context.output_dir, 'extracted_sqls.json')

        with open(output_path, 'w') as f:
            json.dump(context.extracted_sqls, f, indent=2)

        self.logger.info(f"Saved extracted SQLs to {output_path}")


class EvaluateStep(PipelineStep):
    """
    Evaluate extracted SQL queries against ground truth.

    This step evaluates SQL queries using multiple metrics:
    - Execution correctness
    - Exact match
    - Semantic equivalence (optional, using LLM judge)

    Attributes:
        model_provider: Model provider for judge model
        judge_api_key: API key for judge model
        do_judge: Whether to use LLM judge for semantic equivalence
        save_intermediate: Whether to save evaluation results to disk
    """

    def __init__(self,
                 model_provider: Optional[ModelProvider] = None,
                 judge_api_key: Optional[str] = None,
                 do_judge: bool = True,
                 save_intermediate: bool = True,
                 name: Optional[str] = None):
        """
        Initialize the EvaluateStep.

        Args:
            model_provider: Model provider for judge (required if do_judge=True)
            judge_api_key: API key for judge model (required if do_judge=True)
            do_judge: Whether to use LLM judge for semantic equivalence
            save_intermediate: Whether to save evaluation results to disk
            name: Optional custom name for the step
        """
        super().__init__(name or "evaluate")
        self.model_provider = model_provider
        self.judge_api_key = judge_api_key
        self.do_judge = do_judge
        self.save_intermediate = save_intermediate

        if self.do_judge and (not model_provider or not judge_api_key):
            self.logger.warning(
                "do_judge=True but model_provider or judge_api_key not provided. "
                "Semantic equivalence checking will be limited."
            )

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Evaluate extracted SQL queries"""

        # Check if we have extracted SQLs
        if not context.has_data('extracted_sqls'):
            raise ValueError("No extracted SQLs found. Run ExtractStep first or load from disk.")

        self.logger.info(f"Evaluating {len(context.extracted_sqls)} SQL queries...")

        # Create instance mapping
        instance_map = {str(inst.id): (inst, path) for inst, path in instances}

        for instance_id, extracted_sql in tqdm(
            context.extracted_sqls.items(),
            desc="Evaluating SQL",
            unit="instance"
        ):
            if instance_id not in instance_map:
                self.logger.warning(f"Instance {instance_id} not found in provided instances")
                continue

            instance, file_path = instance_map[instance_id]

            if extracted_sql is None:
                # No SQL to evaluate
                context.evaluation_results[instance_id] = {
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
                self.logger.debug(f"Skipping {instance_id} (no SQL extracted)")
                continue

            # Evaluate the SQL
            evaluation = self._evaluate_instance(instance, extracted_sql, file_path)
            context.evaluation_results[instance_id] = evaluation

            self.logger.debug(
                f"Evaluated {instance_id}: "
                f"exec={evaluation['predicted_output']['execution_correct']}, "
                f"exact={evaluation['predicted_output']['exact_match']}, "
                f"semantic={evaluation['predicted_output'].get('semantic_equivalent', False)}"
            )

        # Save intermediate results if requested
        if self.save_intermediate and context.output_dir:
            self._save_evaluation_results(context)

        successful = sum(1 for r in context.evaluation_results.values() if r.get('has_prediction', False))
        self.logger.info(f"Evaluated {successful}/{len(context.evaluation_results)} instances")

        return context

    def _evaluate_instance(self, instance: DatasetInstance, generated_sql: str,
                          instance_path: str) -> Dict:
        """Evaluate a single SQL query"""

        # Get database path
        db_path = get_db_path(instance, instance_path)
        if db_path == 'snowflake':
            self.logger.error("Snowflake evaluation not supported")
            raise NotImplementedError("Snowflake database evaluation is not supported")

        # Check execution accuracy
        exec_correct, exec_error = check_execution_accuracy_general(
            predicted_sql=generated_sql,
            ground_truth_sql=instance.sql,
            db_type=instance.database['type'],
            db_path=db_path,
            skip_unsafe=True
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
            semantic_explanation = "Execution correct but not exact match"
        elif exec_error and exec_error.strip():
            semantic_equivalent = False
            semantic_explanation = f"Execution failed: {exec_error}"
        elif self.do_judge and self.model_provider and self.judge_api_key:
            semantic_equivalent, semantic_explanation = check_sql_semantic_equivalence(
                self.model_provider, generated_sql, instance.sql,
                instance.question, self.judge_api_key
            )

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

    def _save_evaluation_results(self, context: PipelineContext) -> None:
        """Save evaluation results to JSON file"""
        if not context.output_dir:
            return

        os.makedirs(context.output_dir, exist_ok=True)
        output_path = os.path.join(context.output_dir, 'evaluation_results.json')

        # Convert to serializable format (remove instance objects)
        serializable_results = {}
        for instance_id, eval_result in context.evaluation_results.items():
            serializable_result = eval_result.copy()
            if 'instance' in serializable_result:
                del serializable_result['instance']
            serializable_results[instance_id] = serializable_result

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Saved evaluation results to {output_path}")


class SaveStep(PipelineStep):
    """
    Save evaluation results back to instance files.

    This step updates the original DatasetInstance files with
    inference results, maintaining the full history of model outputs.

    Attributes:
        model_info: Model information to include in results
    """

    def __init__(self,
                 model_info: Optional[Dict] = None,
                 name: Optional[str] = None):
        """
        Initialize the SaveStep.

        Args:
            model_info: Model information to include in saved results
            name: Optional custom name for the step
        """
        super().__init__(name or "save")
        self.model_info = model_info or {}

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Save evaluation results to instance files"""

        # Check if we have evaluation results
        if not context.has_data('evaluation_results'):
            raise ValueError("No evaluation results found. Run EvaluateStep first.")

        self.logger.info(f"Saving results for {len(context.evaluation_results)} instances...")

        # Create instance mapping
        instance_map = {str(inst.id): (inst, path) for inst, path in instances}

        for instance_id, eval_result in tqdm(
            context.evaluation_results.items(),
            desc="Saving results",
            unit="instance"
        ):
            if instance_id not in instance_map:
                self.logger.warning(f"Instance {instance_id} not found in provided instances")
                continue

            instance, file_path = instance_map[instance_id]

            # Get raw response if available
            raw_response = ''
            inference_time = None
            if instance_id in context.raw_responses:
                # inference_time if available
                if context.raw_responses[instance_id].get('inference_time') is not None:
                    inference_time = context.raw_responses[instance_id]['inference_time']

                raw_response = context.raw_responses[instance_id].get('raw_response', '')

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
                    'inference_time': inference_time,
                    'raw_response': raw_response
                }
            }

            existing_results.append(result_entry)
            instance.inference_results = existing_results

            # Save the updated instance
            self._save_updated_instance(instance, file_path, context.output_dir)

            self.logger.debug(f"Saved results for {instance_id}")

        self.logger.info(f"Saved {len(context.evaluation_results)} instance files")
        return context

    def _save_updated_instance(self, instance: DatasetInstance,
                              original_file_path: str,
                              output_dir: Optional[str] = None) -> None:
        """Save updated instance to file"""

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_name = os.path.basename(original_file_path)
            new_file_path = os.path.join(output_dir, file_name)
        else:
            new_file_path = original_file_path

        instance_dict = instance.to_dict()
        with open(new_file_path, 'w') as f:
            json.dump(instance_dict, f, indent=2)


class LoadStep(PipelineStep):
    """
    Load intermediate results from disk.

    This step loads previously saved results (raw_responses, extracted_sqls,
    evaluation_results) from disk, enabling pipeline resumption.

    Attributes:
        data_type: Type of data to load ('raw_responses', 'extracted_sqls', 'evaluation_results')
    """

    def __init__(self, data_type: str, name: Optional[str] = None):
        """
        Initialize the LoadStep.

        Args:
            data_type: Type of data to load ('raw_responses', 'extracted_sqls', 'evaluation_results')
            name: Optional custom name for the step
        """
        super().__init__(name or f"load_{data_type}")
        self.data_type = data_type

        valid_types = ['raw_responses', 'extracted_sqls', 'evaluation_results']
        if data_type not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}, got '{data_type}'")

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Load data from disk"""

        if not context.output_dir:
            raise ValueError("output_dir not set in context. Cannot load data.")

        file_name = f"{self.data_type}.json"
        file_path = os.path.join(context.output_dir, file_name)

        if not os.path.exists(file_path):
            self.logger.warning(f"File not found: {file_path}")
            return context

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            setattr(context, self.data_type, data)
            self.logger.info(f"Loaded {len(data)} items from {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise

        return context


class MetricsStep(PipelineStep):
    """
    Calculate and log aggregate metrics.

    This step computes overall performance metrics from evaluation results
    and stores them in the context.

    Metrics include:
    - Prediction rate
    - Execution accuracy
    - Exact match accuracy
    - Semantic equivalence accuracy
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the MetricsStep"""
        super().__init__(name or "metrics")

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Calculate aggregate metrics"""

        if not context.has_data('evaluation_results'):
            self.logger.warning("No evaluation results found. Skipping metrics calculation.")
            return context

        results = list(context.evaluation_results.values())
        num_eval = len(results)
        num_with_prediction = sum(1 for r in results if r.get('has_prediction', False))

        # Calculate accuracies
        exec_correct = sum(
            1 for r in results
            if r.get('has_prediction', False) and
               r['predicted_output'].get('execution_correct', False)
        )
        exact_match = sum(
            1 for r in results
            if r.get('has_prediction', False) and
               r['predicted_output'].get('exact_match', False)
        )
        semantic_equivalent = sum(
            1 for r in results
            if r.get('has_prediction', False) and
               r['predicted_output'].get('semantic_equivalent', False)
        )

        metrics = {
            'num_evaluated': num_eval,
            'num_with_prediction': num_with_prediction,
            'prediction_rate': num_with_prediction / num_eval if num_eval > 0 else 0,
            'execution_accuracy': exec_correct / num_with_prediction if num_with_prediction > 0 else 0,
            'exact_match_accuracy': exact_match / num_with_prediction if num_with_prediction > 0 else 0,
            'semantic_equivalent_accuracy': semantic_equivalent / num_with_prediction if num_with_prediction > 0 else 0,
        }

        context.metadata['metrics'] = metrics

        # Log metrics
        self.logger.info("=" * 50)
        self.logger.info("METRICS SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Evaluated: {metrics['num_evaluated']}")
        self.logger.info(f"With Prediction: {metrics['num_with_prediction']}")
        self.logger.info(f"Prediction Rate: {metrics['prediction_rate']:.2%}")
        self.logger.info(f"Execution Accuracy: {metrics['execution_accuracy']:.2%}")
        self.logger.info(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
        self.logger.info(f"Semantic Equiv Accuracy: {metrics['semantic_equivalent_accuracy']:.2%}")
        self.logger.info("=" * 50)

        return context


class FilterStep(PipelineStep):
    """
    Filter instances based on a condition.

    This step allows you to filter instances mid-pipeline based on
    custom conditions (e.g., only process instances from certain databases).

    Example:
        FilterStep(lambda inst, path: inst.dataset == 'spider')
    """

    def __init__(self, filter_fn, name: Optional[str] = None):
        """
        Initialize the FilterStep.

        Args:
            filter_fn: Function that takes (instance, file_path) and returns bool
            name: Optional custom name for the step
        """
        super().__init__(name or "filter")
        self.filter_fn = filter_fn

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Filter instances - this doesn't modify instances but can log info"""

        original_count = len(instances)
        # Note: We can't actually modify instances here as they're passed in
        # This step is more for logging/validation
        filtered_count = sum(1 for inst, path in instances if self.filter_fn(inst, path))

        self.logger.info(f"Filter would keep {filtered_count}/{original_count} instances")

        # Store filter info in metadata
        context.metadata['filter_info'] = {
            'original_count': original_count,
            'filtered_count': filtered_count
        }

        return context


class LogStep(PipelineStep):
    """
    Log custom information during pipeline execution.

    This step allows you to log custom information or perform
    custom logic at any point in the pipeline.

    Example:
        LogStep(lambda ctx, inst: print(f"Context has {len(ctx.raw_responses)} responses"))
    """

    def __init__(self, log_fn, name: Optional[str] = None):
        """
        Initialize the LogStep.

        Args:
            log_fn: Function that takes (context, instances) and performs logging
            name: Optional custom name for the step
        """
        super().__init__(name or "log")
        self.log_fn = log_fn

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Execute custom logging function"""

        self.log_fn(context, instances)
        return context
