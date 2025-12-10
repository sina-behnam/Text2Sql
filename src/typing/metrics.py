from enum import Enum

class TupleLevelMetricType(str, Enum):
    TUPLE_CARDINALITY = "tuple_cardinality"
    TUPLE_ORDER = "tuple_order"
    TUPLE_CONSTRAINTS = "tuple_constraints"

class ExecutionLevelMetricType(str, Enum):
    EXECUTION_ACCURACY = "execution_accuracy"
    EXECUTION_TIME = "execution_time"
    VALID_EFFICIENCY_SCORE = "valid_efficiency_score"
    EXECUTION_LATENCY = "execution_latency"


class EfficiencyMetricType(str, Enum):
    """Metrics for evaluating query efficiency beyond simple latency."""
    QUERY_PLAN_SCORE = "query_plan_score"
    QUERY_COMPLEXITY = "query_complexity"
    RESULT_SET_EFFICIENCY = "result_set_efficiency"
    MEMORY_EFFICIENCY = "memory_efficiency"

class CellLevelMetricType(str, Enum):
    CELL_PRECISION = "cell_precision"
    CELL_RECALL = "cell_recall"

class MatchingMetricType(str, Enum):
    EXACT_MATCH = "exact_match"
    PROPORTIONAL_EXACT_MATCH = "proportional_exact_match"
    COMPONENT_MATCHING = "component_matching"

class MetricType(str, Enum):
    # Execution Level Metrics
    EXECUTION_ACCURACY = ExecutionLevelMetricType.EXECUTION_ACCURACY
    EXECUTION_TIME = ExecutionLevelMetricType.EXECUTION_TIME
    VALID_EFFICIENCY_SCORE = ExecutionLevelMetricType.VALID_EFFICIENCY_SCORE

    # Cell Level Metrics
    CELL_PRECISION = CellLevelMetricType.CELL_PRECISION
    CELL_RECALL = CellLevelMetricType.CELL_RECALL

    # Tuple Level Metrics
    TUPLE_CARDINALITY = TupleLevelMetricType.TUPLE_CARDINALITY
    TUPLE_ORDER = TupleLevelMetricType.TUPLE_ORDER
    TUPLE_CONSTRAINTS = TupleLevelMetricType.TUPLE_CONSTRAINTS

    # Exact Match Metric
    EXACT_MATCH = MatchingMetricType.EXACT_MATCH
    PROPORTIONAL_EXACT_MATCH = MatchingMetricType.PROPORTIONAL_EXACT_MATCH
    COMPONENT_MATCHING = MatchingMetricType.COMPONENT_MATCHING

    # Efficiency Metrics
    QUERY_PLAN_SCORE = EfficiencyMetricType.QUERY_PLAN_SCORE
    QUERY_COMPLEXITY = EfficiencyMetricType.QUERY_COMPLEXITY
    RESULT_SET_EFFICIENCY = EfficiencyMetricType.RESULT_SET_EFFICIENCY
    MEMORY_EFFICIENCY = EfficiencyMetricType.MEMORY_EFFICIENCY