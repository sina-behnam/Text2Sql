from src.evaluation.metrics.metric import Metric
from src.typing.metrics import MetricType
from src.typing.query import DBQuery
import re
import sqlparse
from abc import abstractmethod

def normalize_sql(sql: str) -> str:
    """
    Normalize SQL query by removing extra spaces and formatting.
    
    Args:
        sql: SQL query string
        
    Returns:
        Normalized SQL query string
    """
    # Use sqlparse to format the SQL query
    try:
        parsed = sqlparse.parse(sql)

        # Convert parsed SQL back to string
        normalized_sql = sqlparse.format(str(parsed[0]), reindent=True, keyword_case='upper')

        # Remove extra spaces
        normalized_sql = re.sub(r'\s+', ' ', normalized_sql).strip()
    except Exception:
        # Fallback: simple whitespace normalization
        normalized_sql = re.sub(r'\s+', ' ', sql).strip()
    
    return normalized_sql

class MatchingMetric(Metric):
    """
    Base class for exact matching metrics.
    """
    @abstractmethod
    def compute(self, target: DBQuery, prediction: DBQuery) -> float:
        pass

    def compute_many(self, target: list[DBQuery], prediction: list[DBQuery]) -> list[float]:
        """
        Compute the metric values for multiple target-prediction pairs.
        Args:
            target (List[DBQuery]): List of ground truth queries.
            prediction (List[DBQuery]): List of predicted queries.
        Returns:
            List[float]: List of computed metric values.
        """

        scores = []
        for prediction_query in prediction:
            
            target_query = self.find_by_id(target, prediction_query.query_id)

            if target_query is None:
                raise ValueError(f"Target query with ID {prediction_query.query_id} not found.")
            
            scores.append(self.compute(target_query, prediction_query))

        return scores

class ExactMatching(MatchingMetric):
    name = MetricType.EXACT_MATCH
    description = "Exact Match Metric"

    def compute(self, target: DBQuery, prediction: DBQuery) -> float:
        """
        Compute exact match between target and prediction SQL queries.

        Args:
            target (DBQuery): The ground truth query.
            prediction (DBQuery): The predicted query.
        Returns:
            float: 1.0 if exact match, 0.0 otherwise.
        # Normalize both queries
        """
        normalized_target = normalize_sql(target.query)
        normalized_prediction = normalize_sql(prediction.query)

        return 1.0 if normalized_target == normalized_prediction else 0.0
    
class ProportionalExactMatching(MatchingMetric):
    name = MetricType.PROPORTIONAL_EXACT_MATCH
    description = "Proportional Exact Match Metric"
    
    def compute(self, target: DBQuery, prediction: DBQuery) -> float:
        """
        Compute proportional exact match between target and prediction SQL queries.

        Args:
            target (DBQuery): The ground truth query.
            prediction (DBQuery): The predicted query.
        Returns:
            float: Proportional exact match score between 0.0 and 1.0
        # Normalize both queries
        """
        normalized_target = normalize_sql(target.query)
        normalized_prediction = normalize_sql(prediction.query)

        target_tokens = normalized_target.split()
        prediction_tokens = normalized_prediction.split()

        # Calculate the number of matching tokens
        matching_tokens = sum(1 for t, p in zip(target_tokens, prediction_tokens) if t == p)

        # Proportional score based on the length of the target query
        if len(target_tokens) == 0:
            return 0.0
        
        proportional_score = matching_tokens / len(target_tokens)
        return round(proportional_score, 3)

class ComponentMatching(MatchingMetric):
    name = MetricType.COMPONENT_MATCHING
    description = "Component-wise Exact Match Metric"

    @staticmethod
    def extract_sql_components(sql: str) -> dict:
            """
            Extract basic SQL components from a SQL query string.
            This is a simplified extractor and may not cover all SQL syntax.
            """
            components = {
                'SELECT': [],
                'FROM': [],
                'WHERE': [],
                'GROUP BY': [],
                'ORDER BY': []
            }
            
            # Normalize and split the SQL query
            sql = normalize_sql(sql).upper()
            tokens = sql.split()
            
            current_component = None
            for token in tokens:
                if token in components:
                    current_component = token
                elif current_component:
                    components[current_component].append(token)
            
            return components

    def compute(self, target: DBQuery, prediction: DBQuery) -> float:
        """
        Compute SQL component-wise exact match between target and prediction SQL queries.

        Args:
            target (DBQuery): The ground truth query.
            prediction (DBQuery): The predicted query.
        Returns:
            float: Component-wise exact match score between 0.0 and 1.0
        """
        target_components = self.extract_sql_components(target.query)
        prediction_components = self.extract_sql_components(prediction.query)

        total_components = 0
        matching_components = 0

        for component in target_components:
            target_part = set(target_components[component])
            prediction_part = set(prediction_components[component])

            if target_part:
                total_components += 1
                if target_part == prediction_part:
                    matching_components += 1

        if total_components == 0:
            return 0.0
        
        component_score = matching_components / total_components
        return round(component_score, 3)
