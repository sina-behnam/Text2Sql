import numpy as np
import pandas as pd
from typing import Dict, List
import spacy
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where
from sqlparse.tokens import Keyword, DML
import networkx as nx
from collections import Counter
import re


try:
    NLP = spacy.load("en_core_web_trf")
except Exception as e:
    raise RuntimeError("Error loading spaCy model: {}".format(e))

class BaseFeatureExtractor:
    """Base class for feature extractors"""
    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names"""
        raise NotImplementedError("Subclasses must implement get_feature_names()")
    
    def vectorize_features(self, analysis: Dict) -> np.ndarray:
        """Convert analysis dictionary to feature vector"""
        raise NotImplementedError("Subclasses must implement vectorize_features()")
    
    def to_dataframe(self, analysis: List[Dict]) -> pd.DataFrame:
        """Convert list of analysis dictionaries to pandas DataFrame"""
        feature_vectors = [self.vectorize_features(a) for a in analysis]
        feature_names = self.get_feature_names()
        return pd.DataFrame(feature_vectors, columns=feature_names)
    
    def fit(self, analyses: List[Dict]):
        """Fit any internal models if needed (optional)"""
        pass

    
class QuestionFeatures(BaseFeatureExtractor):
    """Vectorizer for question analysis features"""
    
    # Common spaCy entity types
    ENTITY_TYPES = [
        'PERSON', 'ORG', 'GPE', 'DATE', 'CARDINAL', 'ORDINAL',
        'MONEY', 'QUANTITY', 'TIME', 'PERCENT', 'PRODUCT', 'EVENT'
    ]
      
    def analyze_question(self,question: str, schemas: List[Dict]) -> Dict:
        """
        Analyze question text for linguistic features and schema overlap

        Args:
            question: The natural language question
            schemas: List of database schemas

        Returns:
            Dictionary with question analysis features
        """
        analysis = {
            'char_length': len(question),
            'word_length': len(question.split()),
            'entities': [],
            'entity_types': [],
            'has_entities': False,
            'numbers': [],
            'has_numbers': False,
            'has_negation': False,
            'negation_words': [],
            'comparatives': [],
            'has_comparatives': False,
            'superlatives': [],
            'has_superlatives': False,
            'table_overlap_count': 0,
            'table_overlap_lemma_count': 0,
            'column_overlap_count': 0,
            'column_overlap_lemma_count': 0
        }

        
        doc = NLP(question)

        # Extract entities
        for ent in doc.ents:
            analysis['entities'].append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        analysis['entity_types'] = list(set([ent['label'] for ent in analysis['entities']]))
        analysis['has_entities'] = len(analysis['entities']) > 0
        # Use spaCy for negation detection
        negations = [token.text for token in doc if token.dep_ == "neg"]
        analysis['negation_words'] = negations
        analysis['has_negation'] = len(negations) > 0

        # Use spaCy POS tagging for comparatives and superlatives
        comparatives = [token.text for token in doc if token.tag_ in ['JJR', 'RBR']]  # Comparative adjectives/adverbs
        superlatives = [token.text for token in doc if token.tag_ in ['JJS', 'RBS']]  # Superlative adjectives/adverbs

        analysis['comparatives'] = comparatives
        analysis['has_comparatives'] = len(comparatives) > 0
        analysis['superlatives'] = superlatives
        analysis['has_superlatives'] = len(superlatives) > 0

        # Detect numbers using spaCy
        numbers = [token.text for token in doc if token.like_num or token.pos_ == 'NUM']
        analysis['numbers'] = numbers
        analysis['has_numbers'] = len(numbers) > 0

        # Get lemmatized words for schema matching
        question_lemmas = set([token.lemma_.lower() for token in doc if not token.is_punct])
        question_words_set = set([token.text.lower() for token in doc if not token.is_punct])
        
        # Extract tables and columns from schemas
        all_tables = set()
        all_columns = set()

        for schema in schemas:
            # Get table name from provided attributes
            table_name = schema.get('table_name') or schema.get('schema_name', '')
            if table_name:
                all_tables.add(table_name.lower())

            # Parse DDL ONLY to extract column names
            ddl = schema.get('DDL', '')
            if ddl:
                parsed = sqlparse.parse(ddl)
                for stmt in parsed:
                    # Extract column names from token stream
                    for token in list(stmt.flatten()):
                        if token.ttype is sqlparse.tokens.Name and not token.is_keyword:
                            col_name = token.value.strip('`"\' ').lower()
                            # Filter out SQL keywords and constraint names
                            if col_name and col_name.upper() not in ['PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'CONSTRAINT', 'UNIQUE', 'NOT', 'NULL', 'DEFAULT', 'CHECK']:
                                all_columns.add(col_name)

        # Count overlaps
        for table in all_tables:
            if table in question_words_set:
                analysis['table_overlap_count'] += 1
            if table in question_lemmas:
                analysis['table_overlap_lemma_count'] += 1

        for column in all_columns:
            column_parts = column.split('_')
            if any(part in question_words_set for part in column_parts):
                analysis['column_overlap_count'] += 1
            if any(part in question_lemmas for part in column_parts):
                analysis['column_overlap_lemma_count'] += 1

        return analysis
    
    def vectorize_features(self, question_analysis: Dict) -> np.ndarray:
        """
        Convert question analysis to feature vector
        
        Args:
            question_analysis: Output from analyze_question()
            
        Returns:
            Feature vector of shape (n_features,)
        """
        features = []
        
        # Numerical features (11)
        features.extend([
            question_analysis.get('char_length', 0),
            question_analysis.get('word_length', 0),
            question_analysis.get('table_overlap_count', 0),
            question_analysis.get('table_overlap_lemma_count', 0),
            question_analysis.get('column_overlap_count', 0),
            question_analysis.get('column_overlap_lemma_count', 0),
            len(question_analysis.get('entities', [])),
            len(question_analysis.get('numbers', [])),
            len(question_analysis.get('negation_words', [])),
            len(question_analysis.get('comparatives', [])),
            len(question_analysis.get('superlatives', []))
        ])
        
        # Binary features (5)
        features.extend([
            int(question_analysis.get('has_entities', False)),
            int(question_analysis.get('has_numbers', False)),
            int(question_analysis.get('has_negation', False)),
            int(question_analysis.get('has_comparatives', False)),
            int(question_analysis.get('has_superlatives', False))
        ])
        
        # Multi-hot encoding for entity types (12)
        entity_types = set(question_analysis.get('entity_types', []))
        entity_vector = [int(et in entity_types) for et in self.ENTITY_TYPES]
        features.extend(entity_vector)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        names = [
            'q_char_length', 'q_word_length',
            'q_table_overlap_count', 'q_table_overlap_lemma_count',
            'q_column_overlap_count', 'q_column_overlap_lemma_count',
            'q_entity_count', 'q_number_count',
            'q_negation_count', 'q_comparative_count', 'q_superlative_count',
            'q_has_entities', 'q_has_numbers', 'q_has_negation',
            'q_has_comparatives', 'q_has_superlatives'
        ]
        names.extend([f'q_entity_{et.lower()}' for et in self.ENTITY_TYPES])
        return names
    
class SQLFeatures(BaseFeatureExtractor):
    """Vectorizer for SQL analysis features"""
    
    # Aggregation functions from analyze_sql
    AGGREGATION_FUNCTIONS = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    
    # SQL clause types from analyze_sql
    CLAUSE_TYPES = [
        'GROUP BY', 'ORDER BY', 'HAVING', 
        'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'
    ]

    def analyze_sql(self,sql: str) -> Dict:
        """
        Analyze SQL query for structural features using sqlparse

        Args:
            sql: The SQL query string

        Returns:
            Dictionary with SQL analysis features
        """
        analysis = {
            'char_length': len(sql),
            'tables_count': 0,
            'tables': [],
            'join_count': 0,
            'where_conditions': 0,
            'subquery_count': 0,
            'clauses_count': 0,
            'clause_types': [],
            'aggregation_function_count': 0,
            'aggregation_functions': [],
            'select_columns': 0
        }

        # Parse SQL
        parsed = sqlparse.parse(sql)
        if not parsed:
            return analysis

        stmt = parsed[0]

        # Count subqueries
        analysis['subquery_count'] = sql.upper().count('SELECT') - 1

        # Extract tables and analyze tokens
        tables = set()
        agg_functions = set()
        in_from = False
        in_join = False

        tokens = list(stmt.flatten())

        for token in tokens:
            # Track FROM and JOIN clauses
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                in_from = True
                in_join = False
            elif token.ttype is Keyword and 'JOIN' in token.value.upper():
                in_join = True
                in_from = False
                analysis['join_count'] += 1
            elif token.ttype is Keyword:
                in_from = False
                in_join = False

            # Extract table names
            if (in_from or in_join) and not token.is_whitespace and token.ttype not in (Keyword, sqlparse.tokens.Punctuation):
                table_name = token.value.strip('`"\' ')
                if table_name and not table_name.upper() in ['AS', 'ON']:
                    tables.add(table_name)

        analysis['tables'] = sorted(list(tables))
        analysis['tables_count'] = len(tables)

        # Extract aggregation functions
        agg_keywords = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
        for token in tokens:
            if token.ttype is sqlparse.tokens.Name.Builtin or str(token).upper() in agg_keywords:
                func_name = str(token).upper()
                if func_name in agg_keywords:
                    agg_functions.add(func_name)
                    analysis['aggregation_function_count'] += 1

        analysis['aggregation_functions'] = sorted(list(agg_functions))

        # Analyze WHERE conditions
        where_clauses = [token for token in stmt.tokens if isinstance(token, Where)]
        if where_clauses:
            where_str = str(where_clauses[0])
            analysis['where_conditions'] = 1 + where_str.upper().count(' AND ') + where_str.upper().count(' OR ')

        # Detect clauses
        sql_upper = sql.upper()
        clause_keywords = ['GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT']
        for clause in clause_keywords:
            if clause in sql_upper:
                analysis['clause_types'].append(clause)
                analysis['clauses_count'] += 1

        # Count SELECT columns
        select_seen = False
        column_count = 0
        for token in stmt.tokens:
            if token.ttype is DML and token.value.upper() == 'SELECT':
                select_seen = True
            elif select_seen and isinstance(token, IdentifierList):
                column_count = len([t for t in token.get_identifiers()])
                break
            elif select_seen and isinstance(token, Identifier):
                column_count = 1
                break
            elif select_seen and token.ttype is sqlparse.tokens.Wildcard:
                column_count = 1
                break
            
        analysis['select_columns'] = column_count

        return analysis
    
    def vectorize_features(self, sql_analysis: Dict) -> np.ndarray:
        """
        Convert SQL analysis to feature vector
        
        Args:
            sql_analysis: Output from analyze_sql()
            
        Returns:
            Feature vector of shape (n_features,)
        """
        features = []
        
        # Numerical features (8)
        features.extend([
            sql_analysis.get('char_length', 0),
            sql_analysis.get('tables_count', 0),
            sql_analysis.get('join_count', 0),
            sql_analysis.get('where_conditions', 0),
            sql_analysis.get('subquery_count', 0),
            sql_analysis.get('clauses_count', 0),
            sql_analysis.get('aggregation_function_count', 0),
            sql_analysis.get('select_columns', 0)
        ])
        
        # Multi-hot encoding for aggregation functions (5)
        agg_functions = set(sql_analysis.get('aggregation_functions', []))
        agg_vector = [int(agg in agg_functions) for agg in self.AGGREGATION_FUNCTIONS]
        features.extend(agg_vector)
        
        # Multi-hot encoding for clause types (7)
        clause_types = set(sql_analysis.get('clause_types', []))
        clause_vector = [int(clause in clause_types) for clause in self.CLAUSE_TYPES]
        features.extend(clause_vector)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        names = [
            'sql_char_length', 'sql_tables_count', 'sql_join_count',
            'sql_where_conditions', 'sql_subquery_count', 'sql_clauses_count',
            'sql_aggregation_function_count', 'sql_select_columns'
        ]
        names.extend([f'sql_agg_{agg.lower()}' for agg in self.AGGREGATION_FUNCTIONS])
        names.extend([f'sql_clause_{clause.lower().replace(" ", "_")}' for clause in self.CLAUSE_TYPES])
        return names
    
class SchemaFeatures(BaseFeatureExtractor):
    """Vectorizer for schema analysis features"""
    
    # Naming conventions from analyze_database_schema
    NAMING_CONVENTIONS = ['snake_case', 'camelCase', 'PascalCase', 'unknown']
    
    # Schema patterns from analyze_database_schema
    SCHEMA_PATTERNS = ['star', 'snowflake', 'normalized', 'simple', 'unknown']

    def analyze_database_schema(self,schemas: List[Dict]) -> Dict:
        """
        Analyze database schema for design features

        Args:
            schemas: List of schema dictionaries with DDL and table info

        Returns:
            Dictionary with schema design features
        """
        analysis = {
            # Structural
            'table_count': len(schemas),
            'total_columns': 0,
            'avg_columns_per_table': 0,

            # Normalization
            'composite_key_count': 0,
            'junction_table_count': 0,
            'redundant_column_patterns': 0,

            # Graph metrics
            'foreign_key_count': 0,
            'clustering_coefficient': 0,
            'connected_components': 0,
            'avg_betweenness_centrality': 0,
            'max_betweenness_centrality': 0,
            'central_tables': [],

            # Semantic
            'naming_convention': 'unknown',  # snake_case, camelCase, PascalCase
            'avg_name_length': 0,
            'lookup_table_count': 0,
            'transactional_table_count': 0,
            'schema_pattern': 'unknown'  # star, snowflake, normalized
        }

        # Build graph
        G = nx.Graph()

        # Track columns across tables
        column_names = []
        table_info = {}

        for schema in schemas:
            table_name = schema.get('table_name') or schema.get('schema_name', '')
            if not table_name:
                continue

            G.add_node(table_name)

            ddl = schema.get('DDL', '')
            if not ddl:
                continue
            
            # Parse DDL
            parsed = sqlparse.parse(ddl)
            if not parsed:
                continue

            stmt = parsed[0]

            # Extract columns and constraints
            columns = []
            foreign_keys = []
            has_composite_pk = False
            has_timestamps = False

            # Simple parsing for columns and constraints
            ddl_upper = ddl.upper()

            # Count columns
            for line in ddl.split('\n'):
                line_stripped = line.strip()
                # Column definition pattern
                if line_stripped and not line_stripped.startswith('--') and not line_stripped.upper().startswith(('CREATE', 'PRIMARY', 'FOREIGN', 'CONSTRAINT', 'CHECK', 'UNIQUE', ')')):
                    col_match = re.match(r'^(["\']?\w+["\']?)\s+', line_stripped)
                    if col_match:
                        col_name = col_match.group(1).strip('`"\' ').lower()
                        columns.append(col_name)
                        column_names.append(col_name)

                        # Check for timestamp columns
                        if any(ts in col_name for ts in ['created_at', 'updated_at', 'timestamp', 'date']):
                            has_timestamps = True

            # Detect composite primary key
            pk_match = re.search(r'PRIMARY KEY\s*\(([^)]+)\)', ddl_upper)
            if pk_match:
                pk_cols = pk_match.group(1).split(',')
                if len(pk_cols) > 1:
                    has_composite_pk = True
                    analysis['composite_key_count'] += 1

            # Extract foreign keys and build graph edges
            fk_pattern = r'FOREIGN KEY\s*\([^)]+\)\s*REFERENCES\s+(["\']?\w+["\']?)'
            fk_matches = re.findall(fk_pattern, ddl_upper)
            for fk_table in fk_matches:
                fk_table = fk_table.strip('`"\' ').lower()
                foreign_keys.append(fk_table)
                G.add_edge(table_name, fk_table)
                analysis['foreign_key_count'] += 1

            # Alternative FK pattern: column REFERENCES table
            ref_pattern = r'REFERENCES\s+(["\']?\w+["\']?)'
            ref_matches = re.findall(ref_pattern, ddl_upper)
            for ref_table in ref_matches:
                ref_table = ref_table.strip('`"\' ').lower()
                if ref_table not in foreign_keys:
                    foreign_keys.append(ref_table)
                    G.add_edge(table_name, ref_table)
                    analysis['foreign_key_count'] += 1

            # Detect junction tables (≤3 columns, ≥2 FKs)
            if len(columns) <= 3 and len(foreign_keys) >= 2:
                analysis['junction_table_count'] += 1

            # Categorize table type
            is_lookup = len(columns) < 5 and any(keyword in table_name for keyword in ['type', 'status', 'category', 'lookup'])
            is_transactional = has_timestamps or len(columns) > 8

            table_info[table_name] = {
                'column_count': len(columns),
                'fk_count': len(foreign_keys),
                'composite_pk': has_composite_pk,
                'is_lookup': is_lookup,
                'is_transactional': is_transactional
            }

            if is_lookup:
                analysis['lookup_table_count'] += 1
            if is_transactional:
                analysis['transactional_table_count'] += 1

        # Column statistics
        analysis['total_columns'] = len(column_names)
        if analysis['table_count'] > 0:
            analysis['avg_columns_per_table'] = len(column_names) / analysis['table_count']

        # Redundant column patterns (same column name in multiple tables)
        column_counter = Counter(column_names)
        analysis['redundant_column_patterns'] = sum(1 for count in column_counter.values() if count > 1)

        # Graph metrics
        if len(G.nodes) > 0:
            try:
                analysis['clustering_coefficient'] = nx.average_clustering(G)
            except:
                analysis['clustering_coefficient'] = 0

            analysis['connected_components'] = nx.number_connected_components(G)

            if len(G.edges) > 0:
                betweenness = nx.betweenness_centrality(G)
                analysis['avg_betweenness_centrality'] = sum(betweenness.values()) / len(betweenness)
                analysis['max_betweenness_centrality'] = max(betweenness.values())
                # Top 3 central tables
                sorted_tables = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
                analysis['central_tables'] = [t[0] for t in sorted_tables[:3]]

        # Naming convention detection
        table_names = [schema.get('table_name') or schema.get('schema_name', '') for schema in schemas]
        table_names = [t for t in table_names if t]

        if table_names:
            snake_case_count = sum(1 for t in table_names if '_' in t)
            camel_case_count = sum(1 for t in table_names if re.search(r'[a-z][A-Z]', t))

            if snake_case_count > len(table_names) * 0.5:
                analysis['naming_convention'] = 'snake_case'
            elif camel_case_count > len(table_names) * 0.5:
                analysis['naming_convention'] = 'camelCase'
            else:
                analysis['naming_convention'] = 'mixed'

            analysis['avg_name_length'] = sum(len(t) for t in table_names) / len(table_names)

        # Schema pattern detection
        if len(G.nodes) > 0:
            degrees = dict(G.degree())
            max_degree = max(degrees.values()) if degrees else 0

            # Star: one central table with high degree, others with degree 1
            hub_tables = [t for t, d in degrees.items() if d >= max_degree * 0.7]
            peripheral = [t for t, d in degrees.items() if d == 1]

            if len(hub_tables) == 1 and len(peripheral) > len(G.nodes) * 0.6:
                analysis['schema_pattern'] = 'star'
            elif len(hub_tables) >= 1 and analysis['junction_table_count'] > 0:
                analysis['schema_pattern'] = 'normalized'
            elif max_degree > 3:
                analysis['schema_pattern'] = 'snowflake'
            else:
                analysis['schema_pattern'] = 'simple'

        return analysis
    
    def vectorize_features(self, schema_analysis: Dict) -> np.ndarray:
        """
        Convert schema analysis to feature vector
        
        Args:
            schema_analysis: Output from analyze_database_schema()
            
        Returns:
            Feature vector of shape (n_features,)
        """
        features = []
        
        # Numerical features (13)
        features.extend([
            schema_analysis.get('table_count', 0),
            schema_analysis.get('total_columns', 0),
            schema_analysis.get('avg_columns_per_table', 0.0),
            schema_analysis.get('composite_key_count', 0),
            schema_analysis.get('junction_table_count', 0),
            schema_analysis.get('redundant_column_patterns', 0),
            schema_analysis.get('foreign_key_count', 0),
            schema_analysis.get('lookup_table_count', 0),
            schema_analysis.get('transactional_table_count', 0),
            schema_analysis.get('clustering_coefficient', 0.0),
            schema_analysis.get('connected_components', 0),
            schema_analysis.get('avg_betweenness_centrality', 0.0),
            schema_analysis.get('max_betweenness_centrality', 0.0)
        ])
        
        # One-hot encoding for naming convention (4)
        naming_convention = schema_analysis.get('naming_convention', 'unknown')
        naming_vector = [int(naming_convention == nc) for nc in self.NAMING_CONVENTIONS]
        features.extend(naming_vector)
        
        # One-hot encoding for schema pattern (5)
        schema_pattern = schema_analysis.get('schema_pattern', 'unknown')
        pattern_vector = [int(schema_pattern == sp) for sp in self.SCHEMA_PATTERNS]
        features.extend(pattern_vector)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        names = [
            'schema_table_count', 'schema_total_columns', 'schema_avg_columns_per_table',
            'schema_composite_key_count', 'schema_junction_table_count',
            'schema_redundant_column_patterns', 'schema_foreign_key_count',
            'schema_lookup_table_count', 'schema_transactional_table_count',
            'schema_clustering_coefficient', 'schema_connected_components',
            'schema_avg_betweenness_centrality', 'schema_max_betweenness_centrality'
        ]
        names.extend([f'schema_naming_{nc}' for nc in self.NAMING_CONVENTIONS])
        names.extend([f'schema_pattern_{sp}' for sp in self.SCHEMA_PATTERNS])
        return names