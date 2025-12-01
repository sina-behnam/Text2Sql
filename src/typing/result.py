from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class ExecutionResult:
    query_id: str
    results: List[Tuple]
    exec_time_ms: float
    success: bool
    error: str = ""

@dataclass
class CellResult:
    query_id: str
    cell_precision: float
    cell_recall: float


@dataclass
class TupleResult:
    query_id: str
    tuple_cardinality: float
    tuple_order: float
    tuple_constraint: float


def to_numpy(results : List[CellResult | TupleResult], metric_name: str) -> np.ndarray:
    """Convert a specific metric from a list of results to a NumPy array.

    Args:
        results (List[CellResult | TupleResult]): List of result objects.
        metric_name (str): The name of the metric to extract.

    Returns:
        np.ndarray: NumPy array of the specified metric values.
    """
    metric_values = []
    for res in results:
        value = getattr(res, metric_name, None)
        if value is not None:
            metric_values.append(value)
    return np.array(metric_values)

def count_successful_executions(results: List[ExecutionResult]) -> int:
    """Count the number of successful executions in the results.

    Args:
        results (List[ExecutionResult]): List of execution result objects.

    Returns:
        int: The count of successful executions.
    """
    return sum(1 for res in results if res.success)

