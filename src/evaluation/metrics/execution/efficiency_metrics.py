"""
Efficiency-based metrics for evaluating SQL query performance.

These metrics evaluate query efficiency from different aspects:
- QueryPlanScore: Analyzes EXPLAIN QUERY PLAN output
- QueryComplexity: Static analysis of query structure
- ResultSetEfficiency: Compares result set sizes and memory footprint
- MemoryEfficiency: Memory usage during query execution
"""

import re
import sys
from typing import List, Tuple, Dict
from tqdm import tqdm
from src.evaluation.metrics.execution.base import ExecutionBasedMetric
from src.typing.result import ExecutionResult
from src.typing.query import DBQuery
from src.typing.metrics import EfficiencyMetricType
from src.workers.efficiency_sql_worker import (
    EfficiencySQLWorker,
    QueryPlanResult
)
from src.utils.loggers.metrics_logger import MetricsLogger, log_with_emoji
import logging

metrics_logger = MetricsLogger.get_instance().get_metrics_logger(
    log_file='metrics_evaluation.log',
    level=logging.INFO
)


def _analyze_query_complexity(query_id: str, sql: str) -> Dict:
    """
    Perform static analysis of SQL query complexity.

    Returns a dict with complexity metrics.
    """
    sql_upper = sql.upper()

    # Count JOINs
    join_count = len(re.findall(r'\bJOIN\b', sql_upper))

    # Count subqueries (SELECT within parentheses, excluding main SELECT)
    subquery_count = max(0, sql_upper.count('SELECT') - 1)

    # Count aggregates
    aggregate_patterns = [r'\bCOUNT\s*\(', r'\bSUM\s*\(', r'\bAVG\s*\(',
                         r'\bMAX\s*\(', r'\bMIN\s*\(', r'\bGROUP_CONCAT\s*\(']
    aggregate_count = sum(len(re.findall(p, sql_upper)) for p in aggregate_patterns)

    # Count DISTINCT
    distinct_count = len(re.findall(r'\bDISTINCT\b', sql_upper))

    # Count UNION/INTERSECT/EXCEPT
    union_count = len(re.findall(r'\b(UNION|INTERSECT|EXCEPT)\b', sql_upper))

    # Count CASE statements
    case_count = len(re.findall(r'\bCASE\b', sql_upper))

    # Calculate nesting depth
    nested_depth = 0
    current_depth = 0
    for i, char in enumerate(sql_upper):
        if char == '(':
            current_depth += 1
            remaining = sql_upper[i+1:i+20]
            if 'SELECT' in remaining[:10]:
                nested_depth = max(nested_depth, current_depth)
        elif char == ')':
            current_depth = max(0, current_depth - 1)

    # Calculate total complexity score (weighted sum)
    total_score = (
        join_count * 2.0 +
        subquery_count * 3.0 +
        aggregate_count * 1.0 +
        distinct_count * 1.5 +
        union_count * 2.0 +
        case_count * 0.5 +
        nested_depth * 1.5
    )

    return {
        'query_id': query_id,
        'join_count': join_count,
        'subquery_count': subquery_count,
        'aggregate_count': aggregate_count,
        'distinct_count': distinct_count,
        'union_count': union_count,
        'case_count': case_count,
        'nested_depth': nested_depth,
        'total_complexity_score': total_score
    }


class QueryPlanScore(ExecutionBasedMetric):
    """
    Evaluates query efficiency based on EXPLAIN QUERY PLAN analysis.

    Scoring:
    - Full table scans are penalized
    - Index usage is rewarded
    - Covering indexes get bonus points
    - Fewer operations is better

    Score range: [0, 2] where 1 is equal efficiency, >1 is better than target
    """
    name = EfficiencyMetricType.QUERY_PLAN_SCORE
    description = "Query Plan Efficiency Score"

    def __init__(self, efficiency_worker: EfficiencySQLWorker = None, **kwargs):
        kwargs['runs_per_query'] = 1
        super().__init__(**kwargs)

        if efficiency_worker is not None:
            self.efficiency_worker = efficiency_worker
        else:
            self.efficiency_worker = EfficiencySQLWorker(
                timeout=kwargs.get('timeout', 6),
                max_try_timeout=kwargs.get('max_try_timeout', 5)
            )

        self._plan_cache: Dict[str, QueryPlanResult] = {}

    def _get_plan_info(self, query: DBQuery) -> QueryPlanResult:
        """Get query plan info with caching."""
        cache_key = f"{query.db_path}:{query.query_id}:{hash(query.query)}"
        if cache_key not in self._plan_cache:
            self._plan_cache[cache_key] = self.efficiency_worker.get_query_plan(
                query.db_path, str(query.query_id), query.query
            )
        return self._plan_cache[cache_key]

    @staticmethod
    def _plan_quality(plan: QueryPlanResult) -> float:
        """Calculate quality score for a single plan."""
        score = 1.0

        # Penalize full scans heavily
        if plan.has_full_scan:
            score *= 0.3

        # Reward index usage
        if plan.has_index_scan:
            score *= 1.5

        # Bonus for covering index
        if plan.has_covering_index:
            score *= 1.3

        # Penalize too many operations
        if plan.operation_count > 0:
            op_penalty = 1.0 / (1.0 + 0.1 * max(0, plan.operation_count - 1))
            score *= op_penalty

        return max(0.01, score)

    @staticmethod
    def compute_plan_score(target_plan: QueryPlanResult, pred_plan: QueryPlanResult) -> float:
        """
        Compute plan efficiency score comparing prediction to target.

        Returns ratio of prediction quality to target quality.
        """
        if not target_plan.success or not pred_plan.success:
            return 0.0

        target_quality = QueryPlanScore._plan_quality(target_plan)
        pred_quality = QueryPlanScore._plan_quality(pred_plan)

        ratio = pred_quality / target_quality
        return min(2.0, max(0.0, ratio))

    # def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
    #     """Not directly usable - requires query text for plan analysis."""
    #     return 1.0

    def compute(self, target: DBQuery, prediction: DBQuery) -> float:
        """Compute plan score for a single query pair."""
        target_plan = self._get_plan_info(target)
        pred_plan = self._get_plan_info(prediction)

        if not target_plan.success:
            log_with_emoji(
                metrics_logger,
                logging.WARNING,
                f"Query {target.query_id}: Target plan analysis failed: {target_plan.error}",
                "warning"
            )
            return 0.0

        return self.compute_plan_score(target_plan, pred_plan)

    def compute_many(
        self,
        target_queries: List[DBQuery],
        predicted_queries: List[DBQuery]
    ) -> Tuple[List[float], List[str]]:
        """Compute plan scores for multiple query pairs."""
        scores = []
        skipped = []

        # Build lookup dict
        target_dict = {str(q.query_id): q for q in target_queries}

        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Analyzing query plans for {len(predicted_queries)} query pairs...",
            "magnifying_glass"
        )

        for pred_query in tqdm(predicted_queries, desc="Query Plan Analysis"):
            query_id = str(pred_query.query_id)
            target_query = target_dict.get(query_id)

            if target_query is None:
                skipped.append(query_id)
                continue

            score = self.compute(target_query, pred_query)
            if score is not None:
                scores.append(score)
            else:
                skipped.append(query_id)

        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Plan analysis complete: {len(scores)} scored, {len(skipped)} skipped",
            "check_mark"
        )

        return scores, skipped

class QueryComplexity(ExecutionBasedMetric):
    """
    Evaluates query complexity through static analysis.

    Analyzes SQL structure for:
    - Number of JOINs
    - Subquery count and depth
    - Aggregate functions
    - Set operations (UNION, etc.)

    Score interpretation:
    - Score > 1: Prediction is simpler than target (good)
    - Score = 1: Equal complexity
    - Score < 1: Prediction is more complex than target
    """
    name = EfficiencyMetricType.QUERY_COMPLEXITY
    description = "Query Complexity Score"

    def __init__(self, **kwargs):
        kwargs['runs_per_query'] = 1
        super().__init__(**kwargs)

    @staticmethod
    def compute_complexity_score(target_complexity: Dict, pred_complexity: Dict) -> float:
        """
        Compute relative complexity score.

        Higher score = prediction is simpler = better.
        """
        target_score = max(1.0, target_complexity['total_complexity_score'])
        pred_score = max(1.0, pred_complexity['total_complexity_score'])

        ratio = target_score / pred_score
        return min(3.0, max(0.1, ratio))

    # def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
    #     """Not used for this metric - requires query text."""
    #     return 1.0

    def compute(self, target: DBQuery, prediction: DBQuery) -> float:
        """Compute complexity score for a single query pair."""
        target_complexity = _analyze_query_complexity(str(target.query_id), target.query)
        pred_complexity = _analyze_query_complexity(str(prediction.query_id), prediction.query)
        return self.compute_complexity_score(target_complexity, pred_complexity)

    def compute_many(
        self,
        target_queries: List[DBQuery],
        predicted_queries: List[DBQuery]
    ) -> Tuple[List[float], List[str]]:
        """Compute complexity scores for multiple query pairs."""
        scores = []
        skipped = []

        target_dict = {str(q.query_id): q for q in target_queries}

        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Analyzing query complexity for {len(predicted_queries)} query pairs...",
            "magnifying_glass"
        )

        for pred_query in tqdm(predicted_queries, desc="Query Complexity Analysis"):
            query_id = str(pred_query.query_id)
            target_query = target_dict.get(query_id)

            if target_query is None:
                skipped.append(query_id)
                continue

            score = self.compute(target_query, pred_query)
            scores.append(score)

        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Complexity analysis complete: {len(scores)} scored, {len(skipped)} skipped",
            "check_mark"
        )

        return scores, skipped

class ResultSetEfficiency(ExecutionBasedMetric):
    """
    Evaluates efficiency based on result set characteristics.

    Compares:
    - Number of rows returned
    - Estimated memory footprint of results

    A prediction that returns correct data with smaller memory footprint
    is considered more efficient.
    """
    name = EfficiencyMetricType.RESULT_SET_EFFICIENCY
    description = "Result Set Efficiency Score"

    def __init__(self, **kwargs):
        kwargs['runs_per_query'] = 1
        super().__init__(**kwargs)

    @staticmethod
    def _estimate_memory_footprint(results: List[Tuple]) -> int:
        """Estimate memory footprint of result set in bytes."""
        if not results:
            return 0

        total_size = 0
        for row in results:
            for cell in row:
                if cell is None:
                    total_size += 8
                elif isinstance(cell, bool):
                    total_size += 28
                elif isinstance(cell, int):
                    total_size += 28
                elif isinstance(cell, float):
                    total_size += 24
                elif isinstance(cell, str):
                    total_size += 49 + len(cell)
                elif isinstance(cell, bytes):
                    total_size += 33 + len(cell)
                else:
                    total_size += sys.getsizeof(cell)

        return total_size

    @staticmethod
    def compute_result_efficiency(target: ExecutionResult, prediction: ExecutionResult) -> float:
        """
        Compute result set efficiency score.

        Returns value in [0, 2] where:
        - 0: Incorrect results
        - 1: Same efficiency
        - >1: Prediction is more memory efficient
        """
        if not target.success or not prediction.success:
            return 0.0

        target_results = target.results
        pred_results = prediction.results

        if len(target_results) == 0 and len(pred_results) == 0:
            return 1.0

        if len(target_results) != len(pred_results):
            return 0.0

        target_set = set(target_results)
        pred_set = set(pred_results)

        if target_set != pred_set:
            return 0.0

        target_memory = ResultSetEfficiency._estimate_memory_footprint(target_results)
        pred_memory = ResultSetEfficiency._estimate_memory_footprint(pred_results)

        if pred_memory == 0:
            return 1.0

        efficiency_ratio = target_memory / max(pred_memory, 1)
        return min(2.0, max(0.0, efficiency_ratio))

    def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
        """Compute result set efficiency for a single pair."""
        return self.compute_result_efficiency(target, prediction)

class MemoryEfficiency(ExecutionBasedMetric):
    """
    Evaluates memory efficiency during query execution.

    Uses tracemalloc to track memory allocations during query execution.
    Compares peak memory usage between target and predicted queries.

    Score interpretation:
    - Score > 1: Prediction uses less memory (good)
    - Score = 1: Equal memory usage
    - Score < 1: Prediction uses more memory
    """
    name = EfficiencyMetricType.MEMORY_EFFICIENCY
    description = "Memory Efficiency Score"

    def __init__(self, efficiency_worker: EfficiencySQLWorker = None, **kwargs):
        kwargs['runs_per_query'] = 1
        super().__init__(**kwargs)

        if efficiency_worker is not None:
            self.efficiency_worker = efficiency_worker
        else:
            self.efficiency_worker = EfficiencySQLWorker(
                timeout=kwargs.get('timeout', 6),
                max_try_timeout=kwargs.get('max_try_timeout', 5)
            )

    @staticmethod
    def compute_memory_score(target_memory: int, pred_memory: int, pred_success: bool) -> float:
        """Compute memory efficiency score."""
        if not pred_success:
            return 0.0

        if target_memory == 0 and pred_memory == 0:
            return 1.0

        if pred_memory == 0:
            return 2.0

        ratio = target_memory / pred_memory
        return min(2.0, max(0.0, ratio))

    # def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
    #     """Not directly usable - memory tracking requires specialized execution."""
    #     return 1.0

    def compute(self, target: DBQuery, prediction: DBQuery) -> float:
        """Compute memory efficiency for a single query pair."""
        target_result = self.efficiency_worker.execute_with_memory(
            target.db_path, str(target.query_id), target.query
        )

        if not target_result.success:
            log_with_emoji(
                metrics_logger,
                logging.WARNING,
                f"Query {target.query_id}: Target execution failed: {target_result.error}",
                "warning"
            )
            return None

        pred_result = self.efficiency_worker.execute_with_memory(
            prediction.db_path, str(prediction.query_id), prediction.query
        )

        return self.compute_memory_score(
            target_result.peak_memory_bytes,
            pred_result.peak_memory_bytes,
            pred_result.success
        )

    def compute_many(
        self,
        target_queries: List[DBQuery],
        predicted_queries: List[DBQuery]
    ) -> Tuple[List[float], List[str]]:
        """Compute memory efficiency scores."""
        scores = []
        skipped = []

        target_dict = {str(q.query_id): q for q in target_queries}

        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Measuring memory usage for {len(predicted_queries)} query pairs...",
            "brain"
        )

        for pred_query in tqdm(predicted_queries, desc="Memory Usage Analysis"):
            query_id = str(pred_query.query_id)
            target_query = target_dict.get(query_id)

            if target_query is None:
                skipped.append(query_id)
                continue

            score = self.compute(target_query, pred_query)
            if score is not None:
                scores.append(score)
            else:
                skipped.append(query_id)

        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Memory analysis complete: {len(scores)} scored, {len(skipped)} skipped",
            "check_mark"
        )

        return scores, skipped
