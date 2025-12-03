from abc import abstractmethod
from typing import List, Tuple, Dict, Optional
from src.workers.sql_worker import SQLWorker
from src.evaluation.metrics.metric import Metric
from src.typing.query import DBQuery
from src.typing.result import ExecutionResult
import json
from pathlib import Path
from utils.loggers.metrics_logger import MetricsLogger, log_with_emoji
import logging

# Get logger instance
metrics_logger = MetricsLogger.get_instance().get_metrics_logger(
    log_file='logs/metrics_evaluation.log',
    level=logging.INFO
)

class ExecutionBasedMetric(Metric):
    """Base class for metrics that require query execution."""

    def __init__(
        self,
        executor = None,
        sql_worker: SQLWorker = None,
        runs_per_query: int = 1,
        cache_target_path: str = None,
        **kwargs
    ):
        """
        Args:
            sql_worker: Optional pre-configured SQLWorker
            runs_per_query: How many times to run each query (for timing)
            cache_target_path: Path to cache target execution results
        """
        # dependency injection
        if executor is not None:
            self._target_results = executor._target_results
        else:
            self._target_results = None

        if sql_worker is not None:
            self.sql_worker = sql_worker
        else:
            self.sql_worker = SQLWorker(runs_per_query=runs_per_query, **kwargs)
        
        # Load cached target results if available
        self._target_results = self._load_cache(cache_target_path) if cache_target_path else None
        self.cache_path = cache_target_path
        
        super().__init__()

    @property
    def target_results(self) -> Optional[Dict[str, ExecutionResult]]:
        """Get cached target execution results, if any."""
        return self._target_results
    
    def set_target_results(self, target_queries: List[DBQuery]) -> None:
        """Set cached target execution results."""
        self._target_results = self._execute_targets(target_queries)

    @staticmethod
    def _load_cache(cache_path: str) -> Optional[Dict[str, ExecutionResult]]:
        """Load cached execution results."""
        if not Path(cache_path).is_file():
            metrics_logger.warning(f"Cache file not found: {cache_path}")
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            results = {}
            for _id, v in cached_data.items():
                results[str(_id)] = ExecutionResult(
                    query_id=str(_id),
                    results=[tuple(row) for row in v['results']],
                    exec_time_ms=v['exec_time_ms'],
                    success=v['success'],
                    error=v.get('error', '')
                )

            log_with_emoji(
                metrics_logger, 
                logging.INFO, 
                f"Loaded {len(results)} cached target results from {cache_path}",
                "check_mark"
            )
            return results
        
        except Exception as e:
            metrics_logger.error(f"Error loading cache from {cache_path}: {e}")
            return None

    @staticmethod
    def _save_cache(cache_path: str, results: List[ExecutionResult]) -> None:
        """Save execution results to cache."""
        try:
            cache_dict = {}
            for res in results:
                cache_dict[int(res.query_id)] = {
                    'results': [list(row) for row in res.results],
                    'exec_time_ms': res.exec_time_ms,
                    'success': res.success,
                    'error': res.error
                }
            
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache_dict, f, indent=4)
            
            log_with_emoji(
                metrics_logger,
                logging.INFO,
                f"Saved {len(results)} target results to cache: {cache_path}",
                "floppy_disk"
            )
            
        except Exception as e:
            metrics_logger.error(f"Error saving cache to {cache_path}: {e}")

    def _execute_targets(self, target_queries: List[DBQuery]) -> Dict[str, ExecutionResult]:
        """Execute target queries, using cache if available."""
        if self._target_results is not None:
            log_with_emoji(
                metrics_logger,
                logging.INFO,
                "Using cached target results",
                "lightning"
            )
            return self._target_results
        
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Executing {len(target_queries)} target queries...",
            "hourglass"
        )
        results_list = self.sql_worker.execute_parallel(target_queries)

        successful_count = sum(1 for r in results_list if r.success)
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Target execution complete: {successful_count}/{len(results_list)} successful",
            "bar_chart"
        )
        
        # Convert to dict
        results_dict = {str(r.query_id): r for r in results_list}
        
        # Cache if path provided
        if self.cache_path:
            self._save_cache(self.cache_path, results_list)
        
        self._target_results = results_dict
        return results_dict

    def _execute_predictions(self, prediction_queries: List[DBQuery]) -> List[ExecutionResult]:
        """Execute prediction queries."""
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Executing {len(prediction_queries)} prediction queries...",
            "rocket"
        )
        
        results = self.sql_worker.execute_parallel(prediction_queries)
        
        successful = sum(1 for r in results if r.success)
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Prediction execution complete: {successful}/{len(results)} successful",
            "bar_chart"
        )
        
        return results

    def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
        """Compute metric score for a single target-prediction pair.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("_compute_score must be implemented by subclasses.")

    def compute_many(self,target: List[DBQuery],prediction: List[DBQuery]) -> Tuple[List[float], List[str]]:
        """
        Compute metric for multiple queries.
        
        Returns:
            Tuple of (scores, skipped_ids)
            - scores: List of metric values (only for valid targets)
            - skipped_ids: List of query IDs that were skipped due to invalid targets
        """
        # Execute all queries
        _target_results_ = self._execute_targets(target)
        prediction_results = self._execute_predictions(prediction)
        
        scores = []
        skipped = []
        
        for pred_res in prediction_results:
            query_id = str(pred_res.query_id)
            target_res = _target_results_.get(query_id)
            
            # Skip if target doesn't exist or failed
            if target_res is None:
                log_with_emoji(
                    metrics_logger,
                    logging.WARNING,
                    f"Query {query_id}: Target result not found",
                    "warning"
                )
                skipped.append(query_id)
                continue
            
            if not target_res.success:
                log_with_emoji(
                    metrics_logger,
                    logging.WARNING,
                    f"Query {query_id}: Invalid target result (error: {target_res.error})",
                    "x"
                )
                skipped.append(query_id)
                continue

            # Compute score using subclass logic
            score = self._compute_score(target_res, pred_res)
            scores.append(score)

        # Summary
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Evaluation complete: {len(scores)} valid, {len(skipped)} skipped",
            "trophy"
        )
        
        if skipped:
            metrics_logger.info(f"Skipped query IDs: {', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''}")
        
        return scores, skipped

    def compute(self, target: DBQuery, prediction: DBQuery) -> float:
        """Compute metric for a single query pair."""
        query_id = str(target.query_id)
        
        # Check cache first
        target_res = None
        if self.target_results:
            target_res = self.target_results.get(query_id)
        
        # Execute if not cached
        if target_res is None:
            target_res = self.sql_worker.execute_single(
                target.db_path, target.query_id, target.query
            )
        
        pred_res = self.sql_worker.execute_single(
            prediction.db_path, prediction.query_id, prediction.query
        )
        
        # Skip if target invalid
        if not target_res.success:
            log_with_emoji(
                metrics_logger,
                logging.WARNING,
                f"Query {query_id}: Invalid target, skipping",
                "warning"
            )
            return None
        
        score = self._compute_score(target_res, pred_res)
        
        log_with_emoji(
            metrics_logger,
            logging.DEBUG,
            f"Query {query_id}: Score = {score:.4f}",
            "dart"
        )
        
        return score
    
    def compute_metric(self, metric_name: str, target, prediction) -> float:
        """Compute a specific metric given target and prediction.

        Args:
            metric_name (str): The name of the metric to compute.
            target (Any): The ground truth data.
            prediction (Any): The predicted data.
        Returns:
            float | int: The computed metric value.
        """
        raise NotImplementedError("Use compute or compute_many methods for ExecutionBasedMetric.")