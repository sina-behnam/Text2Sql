from collections import Counter
from itertools import chain
from turtle import st
import numpy as np
from regex import F
from src.typing.result import ExecutionResult
from src.typing.query import DBQuery

def _sort_key(x):
    """Transforms the input value into a tuple for consistent comparison.
    This method is primarily used as a key function for Python's built-in sorting.
    It transforms the raw input into a tuple that can be used for comparison across various types.
    None values are treated as smallest, followed by numerical types, and then all other types are converted to strings.
    Args:
        x : Variable of any data type.
            The data that needs to be transformed for sorting.
    Returns:
        tuple: A two-element tuple that consists of a priority indicator (int) and a transformed value (float or str).
    Note:
        - None is treated as smallest and assigned a priority of 0.
        - Numerical types (int and float) are assigned a priority of 1 and are uniformly represented as float.
        - All other types are converted to string and assigned a priority of 2.
        - This makes it possible to sort a list containing diverse types of elements.
    """
    if x is None:
        return 0, ''
    elif isinstance(x, (int, float)):
        return 1, float(x)
    else:
        return 2, str(x)
    
def _handle_empty_results(target_rows, prediction_rows):
    """Return a default score for empty result combinations or None when both have data."""
    target_len = len(target_rows)
    prediction_len = len(prediction_rows)
    if target_len == prediction_len == 0:
        return 1.0
    if (target_len == 0) != (prediction_len == 0):
        return 0.0
    return None


class ResultBasedEvaluator:
    
    def evaluate(self, target: ExecutionResult, prediction: ExecutionResult) -> dict:
        return {
            "executability": self.executability(target, prediction),
            "emptiness": self.emptiness(target, prediction),
            "length": len(prediction.results) if prediction and prediction.success else 0,
            "accuracy": self.compute_accuracy(target, prediction),
            "latency": self.latency(target, prediction),
            "ves": self.ves(target, prediction),
            "cell_precision": self.cell_precision(target, prediction),
            "cell_recall": self.cell_recall(target, prediction),
            "tuple_cardinality": self.tuple_cardinality(target, prediction),
            "tuple_order": self.tuple_order(target, prediction),
            "tuple_constraint": self.tuple_constraint(target, prediction),
        }
    
    @staticmethod
    def executability(target: ExecutionResult = None, prediction: ExecutionResult = None) -> float:
        if target and target.success is False:
            return 0.0
        if prediction is None:
            raise ValueError("Prediction result must be provided for executability computation.")
        return float(prediction.success)
    
    @staticmethod
    def compute_accuracy(target: ExecutionResult, prediction: ExecutionResult) -> float:
        """
        Compute execution accuracy between target and prediction results.

        This is the SINGLE SOURCE OF TRUTH for accuracy computation.
        Both ExecAccuracy and VES must use this method.
        """
        if not target.success or not prediction.success:
            return 0.0

        target_results = target.results
        prediction_results = prediction.results

        if len(target_results) == len(prediction_results) == 0:
            return 1.0

        if len(target_results) != len(prediction_results):
            return 0.0

        target_row_set = set(target_results)
        prediction_row_set = set(prediction_results)

        return float(target_row_set == prediction_row_set)
    
    @staticmethod
    def emptiness(target: ExecutionResult, prediction: ExecutionResult) -> float:
        '''
        Compute emptiness score between target and prediction results.
        '''
        target_rows = target.results
        prediction_rows = prediction.results

        if len(target_rows) == 0 and len(prediction_rows) == 0:
            return 1.0
        if len(prediction_rows) == 0:
            return 0.0
        elif len(target_rows) != 0:
            return 1.0
        
        return 0.0

        
    @staticmethod
    def latency(target: ExecutionResult, prediction: ExecutionResult) -> float:
        # Avoid division by zero
        if prediction.exec_time_ms == 0:
            return 0.0
        latency_score = target.exec_time_ms / prediction.exec_time_ms
        return float(round(latency_score, 3))

    @staticmethod
    def ves(target: ExecutionResult, prediction: ExecutionResult) -> float:
        # Use THE SAME accuracy computation as ExecAccuracy
        accuracy = ResultBasedEvaluator.compute_accuracy(target, prediction)

        # If incorrect, VES is 0 (no need to compute time ratio)
        if accuracy == 0.0:
            return 0.0

        # Compute efficiency ratio (with epsilon for safety)
        time_ratio = target.exec_time_ms / max(prediction.exec_time_ms, 1e-6)

        # VES formula
        return float(np.sqrt(time_ratio) * accuracy)
    
    @staticmethod
    def cell_precision(target: ExecutionResult, prediction: ExecutionResult) -> float:
        target_rows = target.results
        prediction_rows = prediction.results

        empty_score = _handle_empty_results(target_rows, prediction_rows)
        if empty_score is not None:
            return empty_score

        target_cells = set(chain.from_iterable(target_rows))
        prediction_cells = set(chain.from_iterable(prediction_rows))
        if len(prediction_cells) == 0:
            return 0.0

        sum_cell_match = len(target_cells.intersection(prediction_cells))
        return float(round(sum_cell_match / len(prediction_cells), 3))
    
    @staticmethod
    def cell_recall(target: ExecutionResult, prediction: ExecutionResult) -> float:
        target_rows = target.results
        prediction_rows = prediction.results

        empty_score = _handle_empty_results(target_rows, prediction_rows)
        if empty_score is not None:
            return empty_score

        target_cells = set(chain.from_iterable(target_rows))
        prediction_cells = set(chain.from_iterable(prediction_rows))
        if len(target_cells) == 0:
            return 0.0

        sum_cell_match = len(target_cells.intersection(prediction_cells))
        return float(round(sum_cell_match / len(target_cells), 3))
    
    @staticmethod
    def tuple_cardinality(target: ExecutionResult, prediction: ExecutionResult) -> float:
        target_rows = target.results
        prediction_rows = prediction.results

        empty_score = _handle_empty_results(target_rows, prediction_rows)
        if empty_score is not None:
            return empty_score

        if len(prediction_rows) >= len(target_rows):
            ratio = len(target_rows) / len(prediction_rows)
        else:
            ratio = len(prediction_rows) / len(target_rows)

        return float(round(ratio, 3))

    @staticmethod
    def tuple_order(target: ExecutionResult, prediction: ExecutionResult) -> float:
        target_rows = target.results
        prediction_rows = prediction.results

        empty_score = _handle_empty_results(target_rows, prediction_rows)
        if empty_score is not None:
            return empty_score

        def _normalize(data: float) -> float:
            data_range = np.array([-1.0, data, 1.0], dtype=float)
            data_range = (data_range - np.min(data_range)) / (np.max(data_range) - np.min(data_range))
            return float(data_range[1])

        # Convert to tuples for hashability (lists aren't hashable)
        target_tuples = [tuple(row) for row in target_rows]
        pred_tuples = [tuple(row) for row in prediction_rows]

        target_set = set(target_tuples)
        pred_set = set(pred_tuples)

        new_pred = []
        seen_pred = set()
        for pred in pred_tuples:
            if pred in target_set and pred not in seen_pred:
                new_pred.append(pred)
                seen_pred.add(pred)

        new_target = []
        seen_target = set()
        for tar in target_tuples:
            if tar in pred_set and tar not in seen_target:
                new_target.append(tar)
                seen_target.add(tar)

        if len(new_target) == 0:
            rho = 0.0
        else:
            target_index_map = {item: idx for idx, item in enumerate(new_target)}

            target_ranks = list(range(len(new_target)))
            pred_ranks = [target_index_map[row] for row in new_pred]
            diff_rank_squared = [(tar - pred) ** 2 for tar, pred in zip(target_ranks, pred_ranks)]
            sum_diff_rank_squared = sum(diff_rank_squared)
            n = len(new_target) if len(new_target) > 1 else 2
            rho = 1 - 6 * sum_diff_rank_squared / (n * (n ** 2 - 1))
        return _normalize(round(rho, 3))
    
    @staticmethod
    def tuple_constraint(target: ExecutionResult, prediction: ExecutionResult) -> float:
        target_rows = target.results
        prediction_rows = prediction.results

        empty_score = _handle_empty_results(target_rows, prediction_rows)
        if empty_score is not None:
            return empty_score

        def _sort_with_different_types(arr):
            return sorted(arr, key=_sort_key)

        target_sorted = [tuple(_sort_with_different_types(row)) for row in target_rows]
        prediction_sorted = [tuple(_sort_with_different_types(row)) for row in prediction_rows]
        count_targ_dict = Counter(target_sorted)
        count_pred_dict = Counter(prediction_sorted)
        cardinality_matches = [count_pred_dict[key] == count for key, count in count_targ_dict.items()]
        return float(round(sum(cardinality_matches) / len(cardinality_matches), 3))




    
        
