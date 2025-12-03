from itertools import chain
from typing import Dict, List, Tuple
from src.typing.metrics import CellLevelMetricType
from src.typing.query import DBQuery
from src.typing.result import ExecutionResult
from src.evaluation.metrics.metric import Metric
from src.evaluation.metrics.execution.base import ExecutionBasedMetric
import numpy as np

class CellLevelMetrics(ExecutionBasedMetric):

    def __init__(self, **kwargs):
        kwargs['runs_per_query'] = 1
        super().__init__(**kwargs)

    def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
        pass


class CellPrecision(CellLevelMetrics):
    name = CellLevelMetricType.CELL_PRECISION
    description = "Cell Precision Metric"

    @staticmethod
    def cell_precision(target, prediction) -> float | int:
        CellLevelMetricType.name = CellLevelMetricType.CELL_PRECISION
        target_len = len(target)
        prediction_len = len(prediction)
        if target_len == prediction_len == 0:
            return 1.0
        if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
            return 0.0
        target = set(chain.from_iterable(target))
        prediction = set(chain.from_iterable(prediction))
        intersected_cells = target.intersection(prediction)
        sum_cell_match = len(intersected_cells)
        return round(sum_cell_match / len(prediction), 3)

    def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
        return self.cell_precision(target.results, prediction.results)

class CellRecall(CellLevelMetrics):
    name = CellLevelMetricType.CELL_RECALL
    description = "Cell Recall Metric"

    @staticmethod
    def cell_recall(target, prediction) -> float | int:
        CellLevelMetricType.name = CellLevelMetricType.CELL_RECALL
        target_len = len(target)
        prediction_len = len(prediction)
        if target_len == prediction_len == 0:
            return 1.0
        if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
            return 0.0
        target = set(chain.from_iterable(target))
        prediction = set(chain.from_iterable(prediction))
        intersected_cells = target.intersection(prediction)
        sum_cell_match = len(intersected_cells)
        return round(sum_cell_match / len(target), 3)
                
    def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
        return self.cell_recall(target.results, prediction.results)


        
