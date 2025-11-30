from enum import Enum

class TupleLevelMetricType(str, Enum):
    TUPLE_CARDINALITY = "tuple_cardinality"
    TUPLE_ORDER = "tuple_order"
    TUPLE_CONSTRAINTS = "tuple_constraints"

class ExecutionLevelMetricType(str, Enum):
    EXECUTION_ACCURACY = "execution_accuracy"
    EXECUTION_TIME = "execution_time"
    VALID_EFFICIENCY_SCORE = "valid_efficiency_score"

class CellLevelMetricType(str, Enum):
    CELL_PRECISION = "cell_precision"
    CELL_RECALL = "cell_recall"

class ExactMatchMetricType(str, Enum):
    EXACT_MATCH = "exact_match"

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
    EXACT_MATCH = ExactMatchMetricType.EXACT_MATCH