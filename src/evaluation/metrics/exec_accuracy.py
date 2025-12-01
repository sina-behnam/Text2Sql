"""
There are two possible input can happen when initializing ExecAccuracy metric.

1. Providing the list of queries to be executed for both target and prediction. In this case, the metric will execute the queries using the provided SQLWorker.
2. Providing the executed results to check the execution accuracy directly.
"""
from src.workers.sql_worker import SQLWorker
from src.evaluation.metrics.metric import Metric
from src.typing.metrics import ExecutionLevelMetricType
from src.typing.query import DBQuery, TargetPredictedDBQuery
from src.typing.result import ExecutionResult
from typing import List, Tuple

class ExecAccuracy(Metric):
    name: ExecutionLevelMetricType = ExecutionLevelMetricType.EXECUTION_ACCURACY
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

    def query2result(
        self,
        query: DBQuery
    ) -> ExecutionResult:
        """Execute a single query and return its execution result."""
        query = (query.db_path, query.query_id, query.query)
        results = self.sql_worker.execute_single(*query)
        return results

    def queries2results(
        self,
        queries: List[DBQuery]
    ) -> List[ExecutionResult]:
        """Execute the provided queries and return their execution results."""
        return self.sql_worker.execute_parallel(queries)
    
    def find_by_id(
        self,
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
    ) -> List[float]:
        """Compute execution accuracy given executed target and prediction results."""
        accuracies = []
        for t in target:
            
            if t.success is False:
                print(f"Warning: Target query with ID {t.query_id} failed during execution. Skipping accuracy computation for this query.")
                accuracies.append(0.0)
                continue

            p = self.find_by_id(prediction, t.query_id)

            if p is None:
                print(f"Warning: No prediction result found for query ID {t.query_id}. Skipping accuracy computation for this query.")
                accuracies.append(0.0)
                continue

            accuracy = self._evalute_(t, p)
            accuracies.append(accuracy)

        return accuracies
    
    def compute(
        self,
        target: DBQuery,
        prediction: DBQuery
    ) -> float:
        """Compute the execution accuracy between a single target and predicted query."""
        target_result = self.query2result(target)
        prediction_result = self.query2result(prediction)
        accuracy = self._evalute_(target_result, prediction_result)
        return accuracy
    
    def compute_many(
        self,
        target: List[DBQuery],
        prediction: List[DBQuery]
    ) ->  List[float]:
        """Compute the execution accuracy between target and predicted results."""
        print("Executing Target Queries...")
        target_results = self.queries2results(target)
        print("Executing Predicted Queries...")
        prediction_results = self.queries2results(prediction)
        return self.compute_from_results(target_results, prediction_results)
    
    def compute_metric(
        self,
        name: str,
        target: List[DBQuery],
        prediction: List[DBQuery]
    ) -> float:
        raise NotImplemented(f"Unsupported metric name: {name}")
