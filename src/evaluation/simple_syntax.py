from turtle import st
import sqlparse
import re
from src.typing.query import DBQuery
from src.evaluation.helpers.dict_schema import extract_sqlite_schema
from src.evaluation.helpers.normalizers import normalize_sql, canonical_form
from collections import Counter

class QuerySyntaxBasedEvaluator:
    """
    Evaluator for SQL query syntax based on normalization.
    """

    def __call__(self, target: DBQuery, prediction: DBQuery) -> dict:
        return self.evaluate(target.query, prediction.query)

    def evaluate(self, target: str, prediction: str) -> dict:
        return {
            "exact_match": self.exact_match(target, prediction),
            "components_match": self.components_match(target, prediction),
            "token_precision": self.token_precision(target, prediction),
            "token_recall": self.token_recall(target, prediction),
            "token_f1": self.token_f1(target, prediction)
        }

    @staticmethod
    def exact_match(target_sql: str, prediction_sql: str) -> float:
        """
        Check if the target and prediction SQL queries match exactly after normalization.
        
        Args:
            target_sql: Target SQL query string
            prediction_sql: Predicted SQL query string
        Returns:
            1.0 if they match exactly, else 0.0
        """
        norm_target = normalize_sql(target_sql)
        norm_prediction = normalize_sql(prediction_sql)
        return 1.0 if norm_target == norm_prediction else 0.0
    
    @staticmethod
    def components_match(target_sql: str, prediction_sql: str) -> float:
        """
        Check if the high-level SQL clauses (SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, OFFSET)
        match after basic normalization.

        Returns:
            1.0 if all extracted clause texts match exactly, else 0.0
        """
        CLAUSES = [
            "SELECT",
            "FROM",
            "WHERE",
            "GROUP BY",
            "HAVING",
            "ORDER BY",
            "LIMIT",
            "OFFSET",
        ]

        # Build a regex that finds clause starts, preferring multi-word matches first.
        clause_pattern = r"\b(" + "|".join(sorted(map(re.escape, CLAUSES), key=len, reverse=True)) + r")\b"

        def extract_components(sql: str) -> dict:
            # Normalize formatting a bit to make splitting more stable
            formatted = sqlparse.format(sql, keyword_case="upper", strip_comments=True, reindent=False)
            formatted = re.sub(r"\s+", " ", formatted).strip()

            parts = re.split(clause_pattern, formatted, flags=re.IGNORECASE)
            # re.split returns: [prefix, CLAUSE, body, CLAUSE, body, ...]
            components = {}
            i = 1
            while i < len(parts) - 1:
                clause = parts[i].upper().strip()
                body = parts[i + 1].strip()

                # Stop clause body at the next clause occurrence (already handled by split)
                # Store normalized body for stable comparison
                components[clause] = re.sub(r"\s+", " ", body).strip()
                i += 2

            return components

        target_components = extract_components(target_sql)
        prediction_components = extract_components(prediction_sql)

        return 1.0 if target_components == prediction_components else 0.0
    
    @staticmethod
    def token_precision(target_sql: str, prediction_sql: str) -> float:
        """
        Token-level precision between canonicalized target and prediction.

        Uses token multiplicities (Counter) so repeated tokens matter.

        Returns:
            Precision score in [0.0, 1.0]
        """
        norm_target = canonical_form(target_sql)
        norm_prediction = canonical_form(prediction_sql)

        target_tokens = norm_target.split()
        prediction_tokens = norm_prediction.split()

        if not prediction_tokens:
            return 1.0 if not target_tokens else 0.0

        target_counts = Counter(target_tokens)
        pred_counts = Counter(prediction_tokens)

        overlap = sum((target_counts & pred_counts).values())

        precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0

        return precision
    
    @staticmethod
    def token_recall(target_sql: str, prediction_sql: str) -> float:
        """
        Token-level recall between canonicalized target and prediction.

        Uses token multiplicities (Counter) so repeated tokens matter.

        Returns:
            Recall score in [0.0, 1.0]
        """
        norm_target = canonical_form(target_sql)
        norm_prediction = canonical_form(prediction_sql)

        target_tokens = norm_target.split()
        prediction_tokens = norm_prediction.split()

        if not target_tokens:
            return 1.0 if not prediction_tokens else 0.0

        target_counts = Counter(target_tokens)
        pred_counts = Counter(prediction_tokens)

        overlap = sum((target_counts & pred_counts).values())

        recall = overlap / len(target_tokens) if target_tokens else 0.0

        return recall
    
    @staticmethod
    def token_f1(target_sql: str, prediction_sql: str) -> float:
        """
        Token-level F1 between canonicalized target and prediction.

        Uses token multiplicities (Counter) so repeated tokens matter.

        Returns:
            F1 score in [0.0, 1.0]
        """
        precision = QuerySyntaxBasedEvaluator.token_precision(target_sql, prediction_sql)
        recall = QuerySyntaxBasedEvaluator.token_recall(target_sql, prediction_sql)

        if precision + recall == 0.0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    

    