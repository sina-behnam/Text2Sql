import numpy as np
from collections import Counter
from typing import Dict

def _normalize(data: float):
    data = [-1, data, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data[1]

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


def _sort_with_different_types(arr):
    sorted_arr = sorted(arr, key=_sort_key)
    return sorted_arr


class TupleLevelMetrics:

    @staticmethod
    def tuple_cardinality(target: list[list], prediction: list[list]) -> float | int:
    
        if len(target) == len(prediction) == 0:
                return 1.0

        if len(prediction) >= len(target):
            # in case we have more elements in the prediction than in the target
            return round(len(target) / len(prediction), 3)

        # in case we have more elements in the target than in the prediction
        return round(len(prediction) / len(target), 3)

    @staticmethod
    def tuple_constraint(target: list[list], prediction: list[list]) -> float | int:
        target_len = len(target)
        prediction_len = len(prediction)
        if target_len == prediction_len == 0:
            return 1.0
        if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
            return 0.0
        # When comparing tuples, the projection orders do not matter (Name, Surname) = (Surname, Name)
        target = [tuple(_sort_with_different_types(row)) for row in target]
        prediction = [tuple(_sort_with_different_types(row)) for row in prediction]
        count_targ_dict = Counter(target)
        count_pred_dict = Counter(prediction)
        cardinality = [count_pred_dict[key] == count for key, count in count_targ_dict.items()]
        return round(sum(cardinality) / len(cardinality), 3)
    
    @staticmethod
    def tuple_order(target: list[list], prediction: list[list]) -> float | int:
        target_len = len(target)
        prediction_len = len(prediction)
        if target_len == prediction_len == 0:
            return 1.0
        if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
            return 0.0
        # take only prediction that are in target without duplicates
        # MAINTAINING the order
        new_pred = []
        [new_pred.append(pred) for pred in prediction
         if pred in target and pred not in new_pred]
        # same for target
        new_target = []
        [new_target.append(tar) for tar in target
         if tar in prediction and tar not in new_target]
        if len(new_target) == 0:
            # case when prediction does not have any element in target
            rho = 0.0
        else:
            target_ranks = [i for i in range(len(new_target))]
            pred_ranks = [new_target.index(row) for row in new_pred]
            diff_rank_squared = [(tar - pred) ** 2
                                 for tar, pred in zip(target_ranks, pred_ranks)]
            sum_diff_rank_squared = sum(diff_rank_squared)
            n = len(new_target) if len(new_target) > 1 else 2
            rho = 1 - 6 * sum_diff_rank_squared / (n * (n ** 2 - 1))
        return _normalize(round(rho, 3))
    
    def compute(self, target: list[list], prediction: list[list]) -> Dict[str, float | int]:
        """Compute all tuple-level metrics given target and prediction.

        Args:
            target (list[list]): The ground truth data as a list of tuples.
            prediction (list[list]): The predicted data as a list of tuples.
        Returns:
            Dict[str, float | int]: A dictionary containing the computed metric values for cardinality,
                                    order, and constraints.                             
        """
        return {
            "tuple_cardinality": self.tuple_cardinality(target, prediction),
            "tuple_order": self.tuple_order(target, prediction),
            "tuple_constraints": self.tuple_constraint(target, prediction)
        }

    def compute_metric(self, metric_name: str, target: list[list], prediction: list[list]) -> float | int:
        """Compute a specific tuple-level metric given target and prediction.

        Args:
            metric_name (str): The name of the metric to compute. Must be one
                                 of "tuple_cardinality", "tuple_order", or "tuple_constraints".
            target (list[list]): The ground truth data as a list of tuples.
            prediction (list[list]): The predicted data as a list of tuples.
        """
        if metric_name == "tuple_cardinality":
            return self.tuple_cardinality(target, prediction)
        elif metric_name == "tuple_order":
            return self.tuple_order(target, prediction)
        elif metric_name == "tuple_constraints":
            return self.tuple_constraint(target, prediction)
        else:
            raise ValueError(f"Unknown metric name: {metric_name}. Valid options are 'tuple_cardinality', 'tuple_order', 'tuple_constraints'.")