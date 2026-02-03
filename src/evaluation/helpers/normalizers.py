import sqlparse
import re
import sqlglot

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

def canonical_form(sql: str) -> str:
    try:
        expression = sqlglot.parse_one(sql)
        return expression.sql(dialect="sqlite", pretty=False, sort=True)
    except Exception:
        return normalize_sql(sql)