from typing import List, Tuple
from src.evaluation.metrics.execution.base import ExecutionBasedMetric
from src.typing.result import ExecutionResult
from src.typing.metrics import ExecutionLevelMetricType

class ExecAccuracy(ExecutionBasedMetric):
    name = ExecutionLevelMetricType.EXECUTION_ACCURACY
    description = "Execution Accuracy"

    def __init__(self, **kwargs):
        kwargs['runs_per_query'] = 1
        super().__init__(**kwargs)

    @staticmethod
    def compute_accuracy(target: ExecutionResult, prediction: ExecutionResult) -> float:
        """
        Compute execution accuracy between target and prediction results.
        
        This is the SINGLE SOURCE OF TRUTH for accuracy computation.
        Both ExecAccuracy and VES must use this method.
        """
        # Both must succeed
        if not target.success or not prediction.success:
            return 0.0

        target_results = target.results
        prediction_results = prediction.results

        # Both empty = correct
        if len(target_results) == 0 and len(prediction_results) == 0:
            return 1.0

        # Different cardinality = incorrect
        if len(target_results) != len(prediction_results):
            return 0.0
        
        # Set comparison (order-independent)
        target_row_set = set(target_results)
        prediction_row_set = set(prediction_results)

        return float(target_row_set == prediction_row_set)

    def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
        """Compute accuracy score using the shared logic."""
        return self.compute_accuracy(target, prediction)