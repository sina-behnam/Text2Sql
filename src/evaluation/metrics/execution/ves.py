from typing import List, Tuple
from src.evaluation.metrics.execution.base import ExecutionBasedMetric
from src.typing.result import ExecutionResult
from src.typing.metrics import ExecutionLevelMetricType
from src.evaluation.metrics.execution.exec_accuracy import ExecAccuracy

class VES(ExecutionBasedMetric):
    name = ExecutionLevelMetricType.VALID_EFFICIENCY_SCORE
    description = "Valid Efficiency Score"

    def __init__(self, runs_per_query: int = 100, **kwargs):
        super().__init__(runs_per_query=runs_per_query, **kwargs)

    def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
        """
        VES = sqrt(target_time / pred_time) * accuracy
        
        CRITICAL: Uses ExecAccuracy.compute_accuracy() to ensure consistency!
        """
        import numpy as np
        
        # Use THE SAME accuracy computation as ExecAccuracy
        accuracy = ExecAccuracy.compute_accuracy(target, prediction)
        
        # If incorrect, VES is 0 (no need to compute time ratio)
        if accuracy == 0.0:
            return 0.0
        
        # Compute efficiency ratio (with epsilon for safety)
        time_ratio = target.exec_time_ms / max(prediction.exec_time_ms, 1e-6)
        
        # VES formula
        return np.sqrt(time_ratio) * accuracy