"""
There are two type of execution can happen 

1. Batch execution, which is a set of queries that all belong to one database. And this happen in Parallel ... 

2. Single execution. (just simple executing a query)

But since we need to calculate VES which basically executing each query multiple times 
    to record what is the average execution time.

    
"""
from typing import List, Tuple
from src.workers.sql_worker import SQLWorker
from src.typing.result import ExecutionResult
from src.typing.query import DBQuery
from src.typing.metrics import ExecutionLevelMetricType
from src.evaluation.metrics.metric import Metric
from src.evaluation.metrics.exec_accuracy import ExecAccuracy
import json
import numpy as np
from pathlib import Path
    
class VES(Metric):
    name: ExecutionLevelMetricType = ExecutionLevelMetricType.VALID_EFFICIENCY_SCORE
    description: str = "Valid Efficiency Score (VES) metric for evaluating SQL query execution efficiency."

    def __init__(
        self,
        cache_target_path: str = None,
        sql_worker: SQLWorker = None,
        *args,
        **kwargs
    ):
        """Initialize the VES metric with optional target query execution caching.
        
        Args:
            cache_target_path (str, optional): Path to cache file for target query results.
            sql_worker (SQLWorker, optional): An instance of SQLWorker for executing queries.
            *args: Additional arguments for SQLWorker initialization.
            **kwargs: Additional keyword arguments for SQLWorker initialization.
        """
        
        if sql_worker is not None:
            self.sql_worker = sql_worker
        else:
            self.sql_worker = SQLWorker(*args, **kwargs)

        self.target_results = self._load_from_cache_file(cache_target_path) if cache_target_path is not None else None
        
        super().__init__()

    @staticmethod
    def _save_to_cache_file(cache_file_path: str, results: dict[int, ExecutionResult]) -> None:
        cache_dict = {}
        for res in results.values():
            cache_dict[int(res.query_id)] = {
                'results': res.results,
                'exec_time_ms': res.exec_time_ms,
                'success': res.success,
                'error': res.error
            }
        with open(cache_file_path, 'w') as f:
            json.dump(cache_dict, f, indent=4, default=list)

    @staticmethod
    def _load_from_cache_file(cache_file_path: str) -> dict[str, ExecutionResult] | None:

        if Path(cache_file_path).is_file():
            results = {}

            try:
                with open(cache_file_path, 'r') as f:
                    cached_data = json.load(f)
                for _id, v in cached_data.items():
                    query_id = str(_id)
                    obj = ExecutionResult(
                        query_id=query_id,
                        results=[tuple(row) for row in v['results']],
                        exec_time_ms=v['exec_time_ms'],
                        success=v['success'],
                        error=v['error']
                    )
                    results[query_id] = obj
            except Exception as e:
                print(f"Error loading cached results: {e}")
                return None

            return results
        else:
            return None

    @staticmethod
    def _list_to_dict(results: List[ExecutionResult]) -> dict[str, ExecutionResult]:
        result_dict = {}
        for res in results:
            result_dict[str(res.query_id)] = res
        return result_dict
    
    def find_by_id(self, target, id):
        return super().find_by_id(target, id)

    def _target_execution(
        self,
        target_queries: List[DBQuery],
        cache_target_path: str = None,
        cache_target: bool = True
    ) -> dict[str, ExecutionResult]:
        """Initialize the execution of target queries, possibly using cached results."""
        if self.target_results is not None:
            return self.target_results
        
        print("Executing target queries for VES computation...")
        target_results = self._list_to_dict(self.sql_worker.execute_parallel(
            target_queries
        ))
        print(f"Executed {len(target_results)} target queries.")
        
        if cache_target and cache_target_path is not None:
            self._save_to_cache_file(cache_target_path, target_results)
        
        return target_results

    def compute_ves(
        self,
        prediction: List[DBQuery],
    ) -> List[ExecutionResult]:
        """Compute the VES for a batch of SQL queries."""

        assert hasattr(self, 'target_results'), "Target results not initialized. Call _target_execution first."
        
        print("Executing predicted queries for VES computation...")
        prediction_results = self.sql_worker.execute_parallel(
            prediction
        )
        
        ves_scores = []
        for pred_res in prediction_results:
            target_res = self.target_results.get(str(pred_res.query_id), None)
            if target_res is None or not target_res.success:
                print(f"Target result for query ID {pred_res.query_id} not found or unsuccessful. Assigning VES score of 0.")
                ves_score = 0.0
            else:
                if pred_res.success:
                    ves_score = np.sqrt(target_res.exec_time_ms / pred_res.exec_time_ms) * ExecAccuracy._evalute_(target_res, pred_res)
                else:
                    ves_score = 0.0
            ves_scores.append((pred_res.query_id, ves_score))
        
        return ves_scores
    
    def compute(
        self,
        target: DBQuery,
        prediction: DBQuery
    ) -> float:
        """Compute VES score for a single target and prediction query pair."""
        target_res = None
        if self.target_results is not None:
            print("Retrieving target result from cached results...")
            target_res = self.target_results.get(prediction.query_id, None)

        target = (target.db_path, target.query_id, target.query)
        prediction = (prediction.db_path, prediction.query_id, prediction.query)        

        if target_res is None:
            target_res = self.sql_worker.execute_single(*target)

        pred_res = self.sql_worker.execute_single(*prediction)
        
        if target_res is None or not target_res.success:
            return 0.0
        
        if pred_res.success:
            ves_score = np.sqrt(target_res.exec_time_ms / pred_res.exec_time_ms) * ExecAccuracy._evalute_(target_res, pred_res)
        else:
            ves_score = 0.0
        
        return ves_score
    
    def compute_many(self, target: List[DBQuery], prediction: List[DBQuery]) -> List[float]:
        """Compute VES scores given target and prediction queries."""
        if self.target_results is None:
            self.target_results = self._target_execution(target_queries=target)
        
        ves_results = self.compute_ves(prediction=prediction)
        ves_scores = [score for _, score in ves_results]
        return ves_scores
    
    def compute_metric(self, metric_name: str, target: DBQuery, prediction: DBQuery) -> float:
        """Compute a specific metric (VES) given target and prediction queries."""
        if metric_name == ExecutionLevelMetricType.VALID_EFFICIENCY_SCORE:
            return self.compute(target, prediction)
        else:
            raise ValueError(f"Unsupported metric name: {metric_name}")
    



        
            
        