"""
Base classes for modular Text2SQL pipeline.

This module provides the foundation for building composable, scikit-learn-style
pipelines for Text2SQL tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging
from enum import Enum

from src.dataloader import DatasetInstance


class StepStatus(str, Enum):
    """Status of a pipeline step execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineContext:
    """
    Shared context/state between pipeline steps.

    Similar to scikit-learn's fit/transform pattern, this context
    carries data and metadata through the pipeline.

    Attributes:
        raw_responses: Raw model outputs for each instance
        extracted_sqls: Extracted SQL queries for each instance
        evaluation_results: Evaluation metrics for each instance
        metadata: Additional metadata (model info, config, etc.)
        step_history: History of executed steps
    """
    # Core data
    raw_responses: Dict[str, Dict] = field(default_factory=dict)
    extracted_sqls: Dict[str, str] = field(default_factory=dict)
    evaluation_results: Dict[str, Dict] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_history: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration
    output_dir: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context by key"""
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context"""
        setattr(self, key, value)

    def has_data(self, key: str) -> bool:
        """Check if context has non-empty data for a key"""
        data = self.get(key)
        return data is not None and len(data) > 0

    def add_step_record(self, step_name: str, status: StepStatus,
                       info: Optional[Dict] = None) -> None:
        """Record a step execution in history"""
        record = {
            'step': step_name,
            'status': status.value,
            'info': info or {}
        }
        self.step_history.append(record)


class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.

    Similar to scikit-learn's BaseEstimator and TransformerMixin,
    this provides a consistent interface for pipeline components.

    Each step can:
    - Access previous results from context
    - Execute its logic
    - Update context with new results
    - Be composed with other steps

    Example:
        class CustomStep(PipelineStep):
            def __init__(self, param1=None):
                super().__init__("custom_step")
                self.param1 = param1

            def execute(self, context, instances):
                # Your logic here
                return context
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the pipeline step.

        Args:
            name: Name of the step (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self._status = StepStatus.PENDING

    @abstractmethod
    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """
        Execute the step logic.

        Args:
            context: Pipeline context with intermediate results
            instances: List of (DatasetInstance, file_path) tuples

        Returns:
            Updated pipeline context

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")

    def __call__(self, context: PipelineContext,
                 instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """
        Make the step callable.

        This allows steps to be used as functions:
            context = step(context, instances)
        """
        try:
            self._status = StepStatus.RUNNING
            self.logger.info(f"Executing step: {self.name}")

            # Execute the step
            updated_context = self.execute(context, instances)

            self._status = StepStatus.COMPLETED
            updated_context.add_step_record(self.name, StepStatus.COMPLETED)
            self.logger.info(f"Step completed: {self.name}")

            return updated_context

        except Exception as e:
            self._status = StepStatus.FAILED
            context.add_step_record(self.name, StepStatus.FAILED, {'error': str(e)})
            self.logger.error(f"Step failed: {self.name} - {str(e)}")
            raise

    @property
    def status(self) -> StepStatus:
        """Get the current status of the step"""
        return self._status

    def skip_if(self, condition_fn) -> 'ConditionalStep':
        """
        Create a conditional step that skips based on a condition.

        Args:
            condition_fn: Function that takes (context, instances) and returns bool

        Returns:
            ConditionalStep wrapper

        Example:
            step.skip_if(lambda ctx, inst: ctx.has_data('extracted_sqls'))
        """
        return ConditionalStep(self, condition_fn)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ConditionalStep(PipelineStep):
    """
    Wrapper that conditionally executes a step.

    Useful for skipping steps based on context state:
        ExtractStep().skip_if(lambda ctx, inst: ctx.has_data('extracted_sqls'))
    """

    def __init__(self, wrapped_step: PipelineStep, condition_fn):
        super().__init__(f"Conditional[{wrapped_step.name}]")
        self.wrapped_step = wrapped_step
        self.condition_fn = condition_fn

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Execute wrapped step only if condition is False"""

        should_skip = self.condition_fn(context, instances)

        if should_skip:
            self.logger.info(f"Skipping step: {self.wrapped_step.name} (condition met)")
            context.add_step_record(self.wrapped_step.name, StepStatus.SKIPPED)
            return context

        return self.wrapped_step(context, instances)


class CompositeStep(PipelineStep):
    """
    A step that composes multiple sub-steps.

    Useful for grouping related steps:
        preprocessing = CompositeStep([
            LoadStep('raw_responses'),
            LoadStep('extracted_sqls')
        ], name='preprocessing')
    """

    def __init__(self, steps: List[PipelineStep], name: Optional[str] = None):
        super().__init__(name or "CompositeStep")
        self.steps = steps

    def execute(self, context: PipelineContext,
                instances: List[Tuple[DatasetInstance, str]]) -> PipelineContext:
        """Execute all sub-steps in order"""

        for step in self.steps:
            context = step(context, instances)

        return context

    def __repr__(self) -> str:
        steps_repr = ", ".join(str(s) for s in self.steps)
        return f"CompositeStep([{steps_repr}])"
