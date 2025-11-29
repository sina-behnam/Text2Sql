"""
There are two possible input can happen when initializing ExecAccuracy metric.

1. Providing the list of queries to be executed for both target and prediction. In this case, the metric will execute the queries using the provided SQLWorker.
2. Providing the executed results to check the execution accuracy directly.
"""
from src.workers.sql_worker import SQLWorker, ExecutionResult
from src.evaluation.metrics.metric import Metric, MetricType
from typing import List, Tuple

class ExecAccuracy(Metric):
    name: MetricType = MetricType.EXECUTION_ACCURACY
    description: str = "Execution Accuracy metric for evaluating SQL query execution correctness."

    def __init__(
        self,
        sql_worker: SQLWorker = None,
        *args,
        **kwargs
    ):
        """Initialize the Execution Accuracy metric."""
        if sql_worker is not None and isinstance(sql_worker, SQLWorker):
            self.sql_worker = sql_worker
        else:
            self.sql_worker = SQLWorker(*args, **kwargs)

        if self.sql_worker.runs_per_query != 1:
            self.sql_worker.runs_per_query = 1
            print("Warning: For only execution accuracy computation, runs_per_query is only should be 1. Therefore it has been set to 1.")

        super().__init__()

    def query2results(
        self,
        queries: List[Tuple[str, str, str]]
    ) -> List[ExecutionResult]:
        """Execute the provided queries and return their execution results."""
        return self.sql_worker.execute_parallel(queries)
    
    @staticmethod
    def _find_by_id_(
        lst: List[ExecutionResult],
        id: str
    ) -> ExecutionResult | None:
        """Find an ExecutionResult by its ID in a list."""
        for res in lst:
            if res.query_id == id:
                return res
        return None
    
    @staticmethod
    def _evalute_(target: ExecutionResult, prediction: ExecutionResult) -> float:
        """Evaluate execution accuracy between target and prediction results."""
        if not target.success or not prediction.success:
            return 0.0

        target_results = target.results
        prediction_results = prediction.results

        if len(target_results) == 0 and len(prediction_results) == 0:
            return 1.0

        if len(target_results) != len(prediction_results):
            return 0.0
        
        target_row_set = set(target_results)
        prediction_row_set = set(prediction_results)

        return float(target_row_set == prediction_row_set)
    
    def compute_from_results(
        self,
        target: List[ExecutionResult],
        prediction: List[ExecutionResult]
    ) -> float:
        """Compute execution accuracy given executed target and prediction results."""
        accuracies = []
        for t in target:
            
            if t.success is False:
                continue

            p = self._find_by_id_(prediction, t.query_id)

            if p is None:
                continue

            accuracy = self._evalute_(t, p)
            accuracies.append(accuracy)

        if len(accuracies) == 0:
            return 0.0
        
        return sum(accuracies) / len(accuracies)
    
    def compute(
        self,
        target: List[Tuple[str, str, str]],
        prediction: List[Tuple[str, str, str]]
    ) -> float:
        """Compute the execution accuracy between target and predicted results."""
        print("Executing Target Queries...")
        target_results = self.query2results(target)
        print("Executing Predicted Queries...")
        prediction_results = self.query2results(prediction)
        accuracy = self.compute_from_results(target_results, prediction_results)
        return accuracy    
    
    def compute_metric(
        self,
        name: str,
        target: List[Tuple[str, str, str]],
        prediction: List[Tuple[str, str, str]]
    ) -> float:
        """Compute Execution Accuracy given target and prediction queries."""
        if name != self.name.value:
            raise ValueError(f"Metric name {name} does not match ExecAccuracy metric name {self.name.value}.")
        return self.compute(target, prediction)
