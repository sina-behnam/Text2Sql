from itertools import chain
from typing import Dict, List, Tuple
from src.typing.metrics import CellLevelMetricType
from src.typing.query import DBQuery
from src.typing.result import CellResult
from src.evaluation.metrics.metric import Metric
import numpy as np

class CellLevelMetrics(Metric):
    name: CellLevelMetricType = CellLevelMetricType
    description: str = "Cell Level Metrics for evaluating individual data cell correctness."

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
    
    def find_by_id(self, target: List[DBQuery], id: str):
        for instance in target:
            if instance.query_id == int(id):
                return instance
        return None
        
    
    def compute(self, target: DBQuery, prediction: DBQuery) -> CellResult:
        precision = self.cell_precision(target.query, prediction.query)
        recall = self.cell_recall(target.query, prediction.query)
        return CellResult(
            query_id=str(target.query_id),
            cell_precision=precision,
            cell_recall=recall
        )
    
    def compute_many(self, target: List[DBQuery], prediction: List[DBQuery]) -> List[CellResult]:
        results = []
        for t in target:
            p = self.find_by_id(prediction, str(t.query_id))
            if p is not None:
                result = self.compute(t, p)
                results.append(result)
        return results
    
    
    def compute_metric(self, metric_name: str, target: DBQuery, prediction: DBQuery) -> float | int:
        """Compute a specific metric given target and prediction."""
        if metric_name == CellLevelMetricType.CELL_PRECISION:
            return self.cell_precision(target.query, prediction.query)
        elif metric_name == CellLevelMetricType.CELL_RECALL:
            return self.cell_recall(target.query, prediction.query)
        else:
            raise ValueError(f"Unsupported metric name: {metric_name}. Valid options are 'cell_precision' and 'cell_recall'.")
        