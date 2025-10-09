
from dataset_readers import (
    SpiderDatasetReader,
    BIRDDatasetReader,
    Spider2DatasetReader
)
from instance import StandardizedInstance

from typing import List, Optional, Dict
from pathlib import Path
import json
import logging
import spacy
import re
# SQL parsing
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where
from sqlparse.tokens import Keyword, DML
import networkx as nx
from collections import Counter
from tqdm import tqdm

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataProcessor:
    """Main processor class that orchestrates dataset reading and processing"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.readers = {
            'spider': SpiderDatasetReader,
            'bird': BIRDDatasetReader,
            'spider2': Spider2DatasetReader
        }

        # Load spacy model once
        try:
            nlp = spacy.load("en_core_web_trf")
            self.nlp = nlp
        except Exception as e:
            logger.error("Error loading spaCy model: {}".format(e))
            raise e

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

        
        doc = self.nlp(question)

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

    def process_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        split: str = 'dev',
        limit: Optional[int] = None,
        save_to_file: bool = False,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> List[StandardizedInstance]:
        """
        Process a dataset and return standardized instances
        
        Args:
            dataset_name: Name of the dataset ('spider', 'bird', 'spider2')
            dataset_path: Path to the dataset directory
            split: Dataset split to process
            limit: Limit number of instances to process
            save_to_file: Whether to save processed instances to files
            output_dir: Directory to save processed instances
            **kwargs: Additional arguments for specific readers
            
        Returns:
            List of standardized instances
        """
        if dataset_name not in self.readers:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(self.readers.keys())}")
        
        # Create reader instance
        if dataset_name == 'spider2':
            reader = self.readers[dataset_name](
                dataset_path,
                dataset_type=kwargs.get('dataset_type', 'lite')
            )
        else:
            reader = self.readers[dataset_name](dataset_path, split)
        
        # Load instances
        instances = reader.load_instances(limit)

        for instance in tqdm(instances, desc="Processing instances"):
            # Analyze question
            try:
                question_analysis = self.analyze_question(instance.question, instance.schemas)
            except Exception as e:
                logger.error(f"Error analyzing question for instance {instance.id}: {e}")
                question_analysis = {}
            
            # Analyze SQL
            try:
                sql_analysis = self.analyze_sql(instance.sql)
            except Exception as e:
                logger.error(f"Error analyzing SQL for instance {instance.id}: {e}")
                sql_analysis = {}
            # Analyze schema
            try:
                if instance.schema_analysis is not None and instance.schema_analysis != {}:
                    schema_analysis = instance.schema_analysis
                else:
                    schema_analysis = self.analyze_database_schema(instance.schemas)
            except Exception as e:
                logger.error(f"Error analyzing schema for instance {instance.id}: {e}")
                schema_analysis = {}
            
            instance.question_analysis = question_analysis
            instance.sql_analysis = sql_analysis
            instance.schema_analysis = schema_analysis

        # Save to files if requested
        if save_to_file:
            self._save_instances(instances, dataset_name, output_dir)
        
        return instances
    
    def _save_instances(
        self,
        instances: List[StandardizedInstance],
        dataset_name: str,
        output_dir: Optional[str] = None
    ):
        """Save processed instances to JSON files"""
        if not output_dir:
            output_dir = f"processed_{dataset_name}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each instance as a separate JSON file
        for instance in instances:
            filename = f"instance_{dataset_name}_{instance.id:04d}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(instance.to_dict(), f, indent=2)
        
        logger.info(f"Saved {len(instances)} instances to {output_path}")
        
        # Also save a summary file
        summary_file = output_path / f"{dataset_name}_summary.json"
        summary = {
            'dataset': dataset_name,
            'total_instances': len(instances),
            'difficulties': {
                'simple': sum(1 for i in instances if i.difficulty == 'simple'),
                'moderate': sum(1 for i in instances if i.difficulty == 'moderate'),
                'challenging': sum(1 for i in instances if i.difficulty == 'challenging')
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")

