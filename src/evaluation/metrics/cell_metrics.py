from itertools import chain
from typing import Dict

class CellLevelMetrics:

    @staticmethod
    def cell_precision(target, prediction) -> float | int:
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
    
    def compute(self, target: any, prediction: any) -> Dict[str, float | int]:
        precision = self.cell_precision(target, prediction)
        recall = self.cell_recall(target, prediction)
        return {
            "cell_precision": precision,
            "cell_recall": recall
        }
    
    def compute_metric(self, metric_name: str, target: any, prediction: any) -> float | int:
        if metric_name == "cell_precision":
            return self.cell_precision(target, prediction)
        elif metric_name == "cell_recall":
            return self.cell_recall(target, prediction)
        else:
            raise ValueError(f"Unsupported metric name: {metric_name}. Valid options are 'cell_precision' and 'cell_recall'.")
        