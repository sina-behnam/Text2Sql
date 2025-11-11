"""
Pipeline steps for Text2SQL tasks.

This package provides modular, composable steps for building
Text2SQL inference pipelines similar to scikit-learn pipelines.
"""

from pipeline.steps.base import (
    PipelineStep,
    PipelineContext,
    StepStatus,
    ConditionalStep,
    CompositeStep
)

from pipeline.steps.text2sql_pipelines import (
    GenerateStep,
    ExtractStep,
    EvaluateStep,
    SaveStep,
    LoadStep,
    MetricsStep,
    FilterStep,
    LogStep
)

__all__ = [
    # Base classes
    'PipelineStep',
    'PipelineContext',
    'StepStatus',
    'ConditionalStep',
    'CompositeStep',

    # Concrete steps
    'GenerateStep',
    'ExtractStep',
    'EvaluateStep',
    'SaveStep',
    'LoadStep',
    'MetricsStep',
    'FilterStep',
    'LogStep',
]
