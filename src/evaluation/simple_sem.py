from sqlglot import exp, parse_one
from collections import Counter
import numpy as np
# Parsing errors
import sqlglot
from src.typing.query import DBQuery

from zss import simple_distance, Node

# ============== NODE TYPE MAPPING ==============
NODE_TYPES = {
    # Clauses
    'Select': 0, 'From': 1, 'Where': 2, 'Join': 3, 'GroupBy': 4,
    'OrderBy': 5, 'Having': 6, 'Limit': 7, 'Distinct': 8,
    # Set Operations
    'Union': 9, 'Intersect': 10, 'Except': 11,
    # Logical
    'And': 12, 'Or': 13, 'Not': 14,
    # Comparisons
    'EQ': 15, 'NEQ': 16, 'GT': 17, 'GTE': 18, 'LT': 19, 'LTE': 20,
    'In': 21, 'Between': 22, 'Like': 23, 'IsNull': 24,
    # Arithmetic
    'Add': 25, 'Sub': 26, 'Mul': 27, 'Div': 28,
    # Aggregates
    'Count': 29, 'Sum': 30, 'Avg': 31, 'Min': 32, 'Max': 33,
    # References
    'Table': 34, 'Column': 35, 'Star': 36, 'Alias': 37, 'Subquery': 38,
    # Values
    'Literal': 39, 'Null': 40, 'Boolean': 41,
    # Misc
    'Case': 42, 'Cast': 43, 'Function': 44, 'Window': 45,
    'CTE': 46, 'Placeholder': 47, 'Other': 48
}

# Direct mapping for faster lookup
_NODE_CLASS_MAP = {
    exp.Select: 'Select', exp.From: 'From', exp.Where: 'Where',
    exp.Join: 'Join', exp.Group: 'GroupBy', exp.Order: 'OrderBy',
    exp.Having: 'Having', exp.Limit: 'Limit', exp.Distinct: 'Distinct',
    exp.Union: 'Union', exp.Intersect: 'Intersect', exp.Except: 'Except',
    exp.And: 'And', exp.Or: 'Or', exp.Not: 'Not',
    exp.EQ: 'EQ', exp.NEQ: 'NEQ', exp.GT: 'GT', exp.GTE: 'GTE',
    exp.LT: 'LT', exp.LTE: 'LTE', exp.In: 'In', exp.Between: 'Between',
    exp.Like: 'Like', exp.Is: 'IsNull',
    exp.Add: 'Add', exp.Sub: 'Sub', exp.Mul: 'Mul', exp.Div: 'Div',
    exp.Count: 'Count', exp.Sum: 'Sum', exp.Avg: 'Avg',
    exp.Min: 'Min', exp.Max: 'Max',
    exp.Table: 'Table', exp.Column: 'Column', exp.Star: 'Star',
    exp.Alias: 'Alias', exp.Subquery: 'Subquery',
    exp.Literal: 'Literal', exp.Null: 'Null', exp.Boolean: 'Boolean',
    exp.Case: 'Case', exp.Cast: 'Cast', exp.Window: 'Window',
    exp.CTE: 'CTE', exp.Placeholder: 'Placeholder',
}

ENTITY_KEYS = ['tables', 'columns', 'aliases', 'literals', 
               'functions', 'schemas', 'parameters', 'datatypes']

# ============== SIMILARITY ==============
def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors."""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def jaccard_similarity(v1, v2):
    """Jaccard similarity (treat as sets via non-zero positions)."""
    s1, s2 = set(np.nonzero(v1)[0]), set(np.nonzero(v2)[0])
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

class SemnticalBasedEvaluator:

    def __call__(self, target: DBQuery | str, prediction: DBQuery | str, *args, **kwds) -> dict:
        # check if target, prediction are passed as DBQuery then get the .query of them and call evaluate
        if target and isinstance(target, DBQuery):
            target = target.query
        if prediction and isinstance(prediction, DBQuery):
            prediction = prediction.query
            
        return self.evaluate(target, prediction, *args, **kwds)

    def evaluate(self, target : str, prediction : str, vocabulary=None, method='cosine'):
        vocab = vocabulary
        if vocab is None:
            vocab = self.build_vocabulary([target, prediction])

        comp = self.compare_queries(target, prediction, vocabulary=vocab, method=method)
        tree_dist = self.ZSS.tree_distance(target, prediction)
        comp['tree_distance'] = tree_dist
        return comp

    # ============ ZSS TREE DISTANCE ==============
    class ZSS:

        @staticmethod
        def _ast_to_zss(node):
            """Convert sqlglot AST to zss Node."""
            label = type(node).__name__
            children = []
            for child in node.args.values():
                if isinstance(child, exp.Expression):
                    children.append(SemnticalBasedEvaluator.ZSS._ast_to_zss(child))
                elif isinstance(child, list):
                    children.extend(SemnticalBasedEvaluator.ZSS._ast_to_zss(c) for c in child if isinstance(c, exp.Expression))
            return Node(label, children)

        @staticmethod
        def tree_distance(sql1, sql2):
            try: 
                t1 = SemnticalBasedEvaluator.ZSS._ast_to_zss(parse_one(sql1))
                t2 = SemnticalBasedEvaluator.ZSS._ast_to_zss(parse_one(sql2))
            except sqlglot.errors.ParseError:
                return None
            except sqlglot.errors.TokenError:
                return None
            return simple_distance(t1, t2)

    # ============== STRUCTURE ENCODER ==============
    @staticmethod
    def get_node_type_idx(node):
        """Map sqlglot node to type index using direct class lookup."""
        node_class = type(node)

        # Direct match
        if node_class in _NODE_CLASS_MAP:
            return NODE_TYPES[_NODE_CLASS_MAP[node_class]]

        # Check inheritance for functions
        if isinstance(node, exp.AggFunc):
            name = node_class.__name__
            if name in NODE_TYPES:
                return NODE_TYPES[name]
            return NODE_TYPES['Function']

        if isinstance(node, exp.Func):
            return NODE_TYPES['Function']

        return NODE_TYPES['Other']

    @staticmethod
    def encode_structure(sql, normalize=False):
        """
        Encode SQL structure as node-type count vector.

        Args:
            sql: SQL string or parsed AST
            normalize: If True, return L2-normalized vector

        Returns:
            np.ndarray of shape (len(NODE_TYPES),)
        """
        ast = parse_one(sql) if isinstance(sql, str) else sql
        counts = Counter()

        for node in ast.walk():
            counts[SemnticalBasedEvaluator.get_node_type_idx(node)] += 1

        vector = np.zeros(len(NODE_TYPES), dtype=np.float32)
        for idx, count in counts.items():
            vector[idx] = count

        if normalize and (norm := np.linalg.norm(vector)) > 0:
            vector /= norm

        return vector


    # ============== ENTITY ENCODER ==============
    @staticmethod
    def extract_entities(sql):
        """Extract named entities from SQL."""
        ast = parse_one(sql) if isinstance(sql, str) else sql
        entities = {k: [] for k in ENTITY_KEYS}

        for node in ast.walk():
            if isinstance(node, exp.Table) and node.name:
                entities['tables'].append(node.name.lower())
            elif isinstance(node, exp.Column) and node.name:
                entities['columns'].append(node.name.lower())
            elif isinstance(node, exp.Alias) and node.alias:
                entities['aliases'].append(node.alias.lower())
            elif isinstance(node, exp.Literal) and node.this:
                entities['literals'].append(str(node.this))
            elif isinstance(node, exp.Func):
                entities['functions'].append(type(node).__name__.upper())
            elif isinstance(node, exp.Schema) and node.name:
                entities['schemas'].append(node.name.lower())
            elif isinstance(node, exp.Placeholder):
                entities['parameters'].append(node.name or '?')
            elif isinstance(node, exp.DataType):
                entities['datatypes'].append(str(node.this.value).upper())

        return entities

    @staticmethod
    def build_vocabulary(sqls):
        """Build vocabulary from list of SQL queries."""
        vocab = {k: set() for k in ENTITY_KEYS}

        for sql in sqls:
            try:
                entities = SemnticalBasedEvaluator.extract_entities(sql)
                for key in vocab:
                    vocab[key].update(entities[key])
            except Exception:
                continue  # Skip unparseable queries
            
        return {k: sorted(v) for k, v in vocab.items()}

    @staticmethod
    def encode_entities(sql, vocabulary, normalize=False, oov_count=False):
        """
        Encode SQL entities as vocabulary-based count vector.

        Args:
            sql: SQL string or parsed AST
            vocabulary: Dict from build_vocabulary()
            normalize: If True, return L2-normalized vector
            oov_count: If True, append OOV counts per entity type. (Out-of-vocabulary)

        Returns:
            np.ndarray
        """
        entities = SemnticalBasedEvaluator.extract_entities(sql)
        vectors = []
        oov_counts = []

        for key in ENTITY_KEYS:
            vocab_set = set(vocabulary.get(key, []))
            counter = Counter(entities[key])

            vec = [counter.get(e, 0) for e in vocabulary.get(key, [])]
            vectors.extend(vec)

            if oov_count:
                oov = sum(c for e, c in counter.items() if e not in vocab_set)
                oov_counts.append(oov)

        if oov_count:
            vectors.extend(oov_counts)

        vector = np.array(vectors, dtype=np.float32)

        if normalize and (norm := np.linalg.norm(vector)) > 0:
            vector /= norm

        return vector


    # ============== COMBINED ENCODER ==============
    @staticmethod
    def encode_sql(sql, vocabulary=None, normalize=False, weights=(1.0, 1.0)):
        """
        Combined encoding: structure + entities.

        Args:
            sql: SQL string
            vocabulary: Entity vocabulary (required for entity encoding)
            normalize: Normalize each sub-vector before concatenation
            weights: (structure_weight, entity_weight)

        Returns:
            np.ndarray concatenated vector
        """
        struct_vec = SemnticalBasedEvaluator.encode_structure(sql, normalize=normalize) * weights[0]

        if vocabulary:
            entity_vec = SemnticalBasedEvaluator.encode_entities(sql, vocabulary, normalize=normalize) * weights[1]
            return np.concatenate([struct_vec, entity_vec])

        return struct_vec

    @staticmethod
    def compare_queries(sql1, sql2, vocabulary=None, method='cosine'):
        """
        Compare two SQL queries.

        Returns:
            dict with structure_sim, entity_sim (if vocab), combined_sim
        """
        try:
            struct1 = SemnticalBasedEvaluator.encode_structure(sql1, normalize=True)
            struct2 = SemnticalBasedEvaluator.encode_structure(sql2, normalize=True)
        except sqlglot.errors.ParseError as e:
            return {'error': f'Parsing error: {e}'}
        except sqlglot.errors.TokenError as e:
            return {'error': f'Tokenization error: {e}'}

        sim_fn = cosine_similarity if method == 'cosine' else jaccard_similarity

        result = {'structure_sim': sim_fn(struct1, struct2)}

        if vocabulary:
            ent1 = SemnticalBasedEvaluator.encode_entities(sql1, vocabulary, normalize=True, oov_count=True)
            ent2 = SemnticalBasedEvaluator.encode_entities(sql2, vocabulary, normalize=True, oov_count=True)
            result['entity_sim'] = sim_fn(ent1, ent2)
            result['combined_sim'] = (result['structure_sim'] + result['entity_sim']) / 2

        return result