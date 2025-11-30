"""
Module defining the base Metric class and MetricType enumeration.

The categories of metrics are as follows:
- Execution Level Metrics: Metrics that evaluate the performance of code execution, such as accuracy and execution time.
- Cell Level Metrics: Metrics that assess the correctness of individual data cells, including precision and recall.
- Tuple Level Metrics: Metrics that analyze the structure and order of data tuples, including cardinality, order, and constraints.
- Exact Match Metric: A metric that checks for an exact match between target and prediction.
"""
from typing import Any, Dict
from abc import ABC, abstractmethod
from enum import Enum
from src.typing.metrics import MetricType
    
class Metric(ABC):
    name: MetricType = None
    description: str = None

    @classmethod
    def get_name(cls) -> MetricType:
        return cls.name

    @abstractmethod
    def compute(self, target, prediction) -> Dict[str, float | int]:
        """Compute the metric value given target and prediction.

        Args:
            target (Any): The ground truth data.
            prediction (Any): The predicted data.
        Returns:
            Dict[str, float | int]: The computed metric values.
        """
        pass

    def compute_metric(self, metric_name: str, target, prediction) -> float | int:
        """Compute a specific metric given target and prediction.

        Args:
            metric_name (str): The name of the metric to compute.
            target (Any): The ground truth data.
            prediction (Any): The predicted data.
        Returns:
            float | int: The computed metric value.
        """
        pass


