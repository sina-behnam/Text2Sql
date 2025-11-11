"""
Modular, composable pipeline for Text2SQL inference.

This module provides a scikit-learn-style pipeline that allows you to
compose and chain Text2SQL operations flexibly.

Example:
    >>> from pipeline.modular_pipeline import Text2SQLPipeline
    >>> from pipeline.steps import GenerateStep, ExtractStep, EvaluateStep
    >>>
    >>> pipeline = Text2SQLPipeline([
    ...     GenerateStep(model_provider, prompt_template),
    ...     ExtractStep(prompt_template),
    ...     EvaluateStep(model_provider, judge_api_key),
    ... ])
    >>>
    >>> context = pipeline.fit_transform(instances)
"""

from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime

from pipeline.steps.base import PipelineStep, PipelineContext
from src.dataloader import DatasetInstance


class Text2SQLPipeline:
    """
    Modular pipeline for Text2SQL inference.

    Similar to scikit-learn's Pipeline, this class allows you to compose
    multiple processing steps and execute them in sequence.

    Features:
    - Composable: Chain any sequence of steps
    - Flexible: Add custom steps easily
    - Resumable: Load intermediate results from disk
    - Extensible: Skip steps conditionally
    - Clean: Each step has a single responsibility

    Attributes:
        steps: List of pipeline steps to execute
        context: Shared context between steps
        output_dir: Directory for intermediate and final results

    Example:
        >>> pipeline = Text2SQLPipeline([
        ...     GenerateStep(model, template),
        ...     ExtractStep(template),
        ...     EvaluateStep(model, api_key),
        ...     MetricsStep(),
        ... ], output_dir='./results')
        >>>
        >>> context = pipeline.fit_transform(instances)
        >>> print(context.metadata['metrics'])
    """

    def __init__(self,
                 steps: List[PipelineStep],
                 output_dir: Optional[str] = None,
                 logging_config: Optional[Dict] = None):
        """
        Initialize the pipeline.

        Args:
            steps: List of PipelineStep instances to execute in order
            output_dir: Directory for saving intermediate and final results
            logging_config: Optional logging configuration
        """
        self.steps = steps
        self.output_dir = output_dir
        self.context = PipelineContext(output_dir=output_dir)
        self.logger = logging.getLogger(__name__)

        # Setup logging if config provided
        if logging_config:
            self._setup_logging(logging_config)

        self.logger.info(f"Initialized pipeline with {len(steps)} steps")

    def _setup_logging(self, config: Dict) -> None:
        """Setup logging configuration"""
        # This allows customization of logging per pipeline
        # Configure logging to file only
        if 'filename' in config:
            # Remove all existing handlers first
            logging.getLogger().handlers.clear()

            # Then configure file logging
            logging.basicConfig(
                filename=config['filename'],
                level=config.get('level', logging.INFO),
                format=config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                filemode=config.get('filemode', 'a')
            )
            


    def fit(self, instances: List[Tuple[DatasetInstance, str]]) -> 'Text2SQLPipeline':
        """
        Fit the pipeline (execute all steps).

        This follows the scikit-learn convention of fit().

        Args:
            instances: List of (DatasetInstance, file_path) tuples

        Returns:
            self for method chaining
        """
        self.logger.info(f"Fitting pipeline on {len(instances)} instances")
        start_time = datetime.now()

        for i, step in enumerate(self.steps):
            self.logger.info(f"[{i+1}/{len(self.steps)}] Executing: {step.name}")
            try:
                self.context = step(self.context, instances)
            except Exception as e:
                self.logger.error(f"Step '{step.name}' failed: {e}")
                raise

        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Pipeline completed in {elapsed:.2f}s")

        return self

    def transform(self, instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """
        Execute the pipeline and return the context.

        This follows the scikit-learn convention of transform().

        Args:
            instances: List of (DatasetInstance, file_path) tuples

        Returns:
            PipelineContext with all intermediate and final results
        """
        self.fit(instances)
        return self.context

    def fit_transform(self, instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """
        Fit the pipeline and return the context.

        This is equivalent to calling fit() then accessing the context,
        but follows scikit-learn's fit_transform() convention.

        Args:
            instances: List of (DatasetInstance, file_path) tuples

        Returns:
            PipelineContext with all intermediate and final results
        """
        return self.transform(instances)

    def add_step(self, step: PipelineStep) -> 'Text2SQLPipeline':
        """
        Add a step to the end of the pipeline.

        Args:
            step: PipelineStep to add

        Returns:
            self for method chaining
        """
        self.steps.append(step)
        self.logger.info(f"Added step: {step.name}")
        return self

    def insert_step(self, index: int, step: PipelineStep) -> 'Text2SQLPipeline':
        """
        Insert a step at a specific position.

        Args:
            index: Position to insert the step
            step: PipelineStep to insert

        Returns:
            self for method chaining
        """
        self.steps.insert(index, step)
        self.logger.info(f"Inserted step '{step.name}' at position {index}")
        return self

    def remove_step(self, name: str) -> 'Text2SQLPipeline':
        """
        Remove a step by name.

        Args:
            name: Name of the step to remove

        Returns:
            self for method chaining
        """
        original_count = len(self.steps)
        self.steps = [s for s in self.steps if s.name != name]
        removed_count = original_count - len(self.steps)

        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} step(s) named '{name}'")
        else:
            self.logger.warning(f"No step found with name '{name}'")

        return self

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """
        Get a step by name.

        Args:
            name: Name of the step to get

        Returns:
            PipelineStep if found, None otherwise
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_context(self) -> PipelineContext:
        """
        Get the current pipeline context.

        Returns:
            Current PipelineContext
        """
        return self.context

    def reset_context(self) -> 'Text2SQLPipeline':
        """
        Reset the pipeline context to a fresh state.

        Returns:
            self for method chaining
        """
        self.context = PipelineContext(output_dir=self.output_dir)
        self.logger.info("Context reset")
        return self

    def __len__(self) -> int:
        """Return the number of steps in the pipeline"""
        return len(self.steps)

    def __getitem__(self, index: int) -> PipelineStep:
        """Get a step by index"""
        return self.steps[index]

    def __repr__(self) -> str:
        steps_repr = "\n  ".join(f"{i}. {step}" for i, step in enumerate(self.steps))
        return f"Text2SQLPipeline(\n  {steps_repr}\n)"


class PipelineBuilder:
    """
    Builder class for constructing pipelines fluently.

    This provides a more readable way to construct complex pipelines:

    Example:
        >>> pipeline = (PipelineBuilder()
        ...     .with_generation(model, template)
        ...     .with_extraction(template)
        ...     .with_evaluation(model, api_key)
        ...     .with_metrics()
        ...     .build(output_dir='./results'))
    """

    def __init__(self):
        self.steps = []
        self._output_dir = None

    def with_step(self, step: PipelineStep) -> 'PipelineBuilder':
        """Add a custom step"""
        self.steps.append(step)
        return self

    def with_generation(self,
                       model_provider,
                       prompt_template,
                       save_intermediate: bool = True) -> 'PipelineBuilder':
        """Add a GenerateStep"""
        from pipeline.steps.text2sql_pipelines import GenerateStep
        self.steps.append(GenerateStep(model_provider, prompt_template, save_intermediate))
        return self

    def with_extraction(self,
                       prompt_template,
                       save_intermediate: bool = True) -> 'PipelineBuilder':
        """Add an ExtractStep"""
        from pipeline.steps.text2sql_pipelines import ExtractStep
        self.steps.append(ExtractStep(prompt_template, save_intermediate))
        return self

    def with_evaluation(self,
                       model_provider=None,
                       judge_api_key=None,
                       do_judge: bool = True,
                       save_intermediate: bool = True) -> 'PipelineBuilder':
        """Add an EvaluateStep"""
        from pipeline.steps.text2sql_pipelines import EvaluateStep
        self.steps.append(EvaluateStep(model_provider, judge_api_key, do_judge, save_intermediate))
        return self

    def with_save(self, model_info: Optional[Dict] = None) -> 'PipelineBuilder':
        """Add a SaveStep"""
        from pipeline.steps.text2sql_pipelines import SaveStep
        self.steps.append(SaveStep(model_info))
        return self

    def with_metrics(self) -> 'PipelineBuilder':
        """Add a MetricsStep"""
        from pipeline.steps.text2sql_pipelines import MetricsStep
        self.steps.append(MetricsStep())
        return self

    def with_load(self, data_type: str) -> 'PipelineBuilder':
        """Add a LoadStep"""
        from pipeline.steps.text2sql_pipelines import LoadStep
        self.steps.append(LoadStep(data_type))
        return self

    def with_filter(self, filter_fn) -> 'PipelineBuilder':
        """Add a FilterStep"""
        from pipeline.steps.text2sql_pipelines import FilterStep
        self.steps.append(FilterStep(filter_fn))
        return self

    def with_log(self, log_fn) -> 'PipelineBuilder':
        """Add a LogStep"""
        from pipeline.steps.text2sql_pipelines import LogStep
        self.steps.append(LogStep(log_fn))
        return self

    def output_dir(self, path: str) -> 'PipelineBuilder':
        """Set the output directory"""
        self._output_dir = path
        return self

    def build(self, output_dir: Optional[str] = None) -> Text2SQLPipeline:
        """Build the pipeline"""
        final_output_dir = output_dir or self._output_dir
        return Text2SQLPipeline(self.steps, output_dir=final_output_dir)


def create_default_pipeline(
    model_provider,
    prompt_template,
    judge_api_key: Optional[str] = None,
    output_dir: str = './results',
    save_to_files: bool = True,
    do_judge: bool = True
) -> Text2SQLPipeline:
    """
    Create a standard Text2SQL pipeline with common steps.

    This is a convenience function for creating a typical pipeline:
    Generate → Extract → Evaluate → Metrics → Save

    Args:
        model_provider: Model provider for SQL generation
        prompt_template: Prompt template for formatting
        judge_api_key: API key for judge model (optional)
        output_dir: Directory for results
        save_to_files: Whether to save results to instance files
        do_judge: Whether to use LLM judge for semantic equivalence

    Returns:
        Configured Text2SQLPipeline

    Example:
        >>> pipeline = create_default_pipeline(
        ...     model_provider=my_model,
        ...     prompt_template=my_template,
        ...     judge_api_key='...',
        ...     output_dir='./results'
        ... )
        >>> context = pipeline.fit_transform(instances)
    """
    from pipeline.steps.text2sql_pipelines import (
        GenerateStep, ExtractStep, EvaluateStep, MetricsStep, SaveStep
    )

    steps = [
        GenerateStep(model_provider, prompt_template),
        ExtractStep(prompt_template),
        EvaluateStep(model_provider, judge_api_key, do_judge=do_judge),
        MetricsStep(),
    ]

    if save_to_files:
        model_info = {
            'model_name': model_provider.model_name,
            'model_config': model_provider.config.model_dump()
        }
        steps.append(SaveStep(model_info))

    return Text2SQLPipeline(steps, output_dir=output_dir)


def create_resume_pipeline(
    prompt_template,
    model_provider=None,
    judge_api_key: Optional[str] = None,
    output_dir: str = './results',
    resume_from: str = 'extracted_sqls',
    do_judge: bool = True
) -> Text2SQLPipeline:
    """
    Create a pipeline that resumes from intermediate results.

    Useful when you want to re-run evaluation with different settings
    without regenerating SQL queries.

    Args:
        prompt_template: Prompt template for formatting
        model_provider: Model provider (needed for evaluation)
        judge_api_key: API key for judge model
        output_dir: Directory with intermediate results
        resume_from: What to resume from ('raw_responses', 'extracted_sqls')
        do_judge: Whether to use LLM judge

    Returns:
        Configured Text2SQLPipeline

    Example:
        >>> # Re-run evaluation without regenerating
        >>> pipeline = create_resume_pipeline(
        ...     prompt_template=my_template,
        ...     model_provider=my_model,
        ...     judge_api_key='...',
        ...     output_dir='./results',
        ...     resume_from='extracted_sqls'
        ... )
        >>> context = pipeline.fit_transform(instances)
    """
    from pipeline.steps.text2sql_pipelines import (
        LoadStep, ExtractStep, EvaluateStep, MetricsStep
    )

    steps = []

    if resume_from == 'raw_responses':
        steps.extend([
            LoadStep('raw_responses'),
            ExtractStep(prompt_template),
        ])
    elif resume_from == 'extracted_sqls':
        steps.append(LoadStep('extracted_sqls'))
    else:
        raise ValueError(f"Invalid resume_from: {resume_from}")

    steps.extend([
        EvaluateStep(model_provider, judge_api_key, do_judge=do_judge),
        MetricsStep(),
    ])

    return Text2SQLPipeline(steps, output_dir=output_dir)
