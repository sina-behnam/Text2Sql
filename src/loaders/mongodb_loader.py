"""
MongoDB loader for normalized Text2SQL data.

#Collections
1. `instances`

Stores Text2SQL problem instances (questions/queries).

- instance_id: Unique ID (e.g., bird_dev_123)
- dataset: Dataset name (e.g., bird)
- split: Split name (e.g., dev)
- original_id: Original numeric ID
- db_name, db_path: Database info
- sql: Gold/reference SQL
- evidence: Additional context
- schemas: Table schemas
- cache_sql_execution: Cached execution of gold SQL

-------

2. `inference_results`

Stores model outputs for each instance.

- instance_id: References instances collection
- model_name: Model that generated response
- config_profile: Config ID (e.g., t0.5_f0_p0)
- raw_response: Model's raw output
- version, created_at, updated_at: Versioning
- cache: Contains:
- extracted_sql: Parsed SQL from response
- execution: Cached execution result

--------

3. `config_profiles`

Stores inference configurations.

- _id: Profile ID (e.g., t0.5_f0_p0)
- temperature, frequency_penalty, presence_penalty

--------

4. `inference_results_history`
Archives old versions when keep_history=True.

--------

## Indexes

- instances.instance_id (unique)
- inference_results.(instance_id, model_name, config_profile) (unique compound)
"""
from dataclasses import dataclass, field
import time
from typing import List, Optional, Dict, Iterator
from datetime import datetime, timezone
from emoji import config
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from loguru import logger

from src.typing.result import CachedExecutionResult
from src.typing.query import DBQuery
from src.workers.sql_worker import SQLWorker
from src.workers.sql_parser import SQLExtractor

SCHEMA_VERSION = "1.0.0"
MAX_RESULT_ROWS = 10000  # Skip caching results with more rows than this

@dataclass
class InferenceResultCache:
    """Cached extraction and execution for an inference result."""
    extracted_sql: Optional[str] = None
    execution: Optional[CachedExecutionResult] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'InferenceResultCache':
        if not data:
            return None
        exec_data = data.get('execution')
        return cls(
            extracted_sql=data.get('extracted_sql'),
            execution=CachedExecutionResult.from_dict(exec_data) if exec_data else None
        )

    def to_dict(self) -> Dict:
        return {
            'extracted_sql': self.extracted_sql,
            'execution': self.execution.to_dict() if self.execution else None
        }


@dataclass
class InferenceResult:
    """Single inference result from a model."""
    model_name: str
    config_profile: str
    raw_response: str
    version: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    cache: Optional[InferenceResultCache] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'InferenceResult':
        cache_data = data.get('cache')
        return cls(
            model_name=data['model_name'],
            config_profile=data['config_profile'],
            raw_response=data.get('raw_response', ''),
            version=data.get('version', 1),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            cache=InferenceResultCache.from_dict(cache_data) if cache_data else None
        )

    def to_dict(self) -> Dict:
        d = {
            'model_name': self.model_name,
            'config_profile': self.config_profile,
            'raw_response': self.raw_response,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        if self.cache:
            d['cache'] = self.cache.to_dict()
        return d


@dataclass
class Instance:
    """Text2SQL instance with inference results."""
    instance_id: str
    dataset: str
    split: str
    original_id: int
    db_name: Optional[str] = None
    db_path: Optional[str] = None
    sql: Optional[str] = None
    evidence: Optional[str] = None
    schemas: Optional[List[Dict]] = None
    cache_sql_execution: Optional[CachedExecutionResult] = None
    inference_results: List[InferenceResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict, inference_results: List[Dict] = None) -> 'Instance':
        cache_data = data.get('cache_sql_execution')
        return cls(
            instance_id=data['instance_id'],
            dataset=data['dataset'],
            split=data['split'],
            original_id=data['original_id'],
            db_name=data.get('db_name'),
            db_path=data.get('db_path'),
            sql=data.get('sql'),
            evidence=data.get('evidence'),
            schemas=data.get('schemas'),
            cache_sql_execution=CachedExecutionResult.from_dict(cache_data) if cache_data else None,
            inference_results=[InferenceResult.from_dict(r) for r in (inference_results or [])]
        )

    def to_dict(self) -> Dict:
        d = {
            'instance_id': self.instance_id,
            'dataset': self.dataset,
            'split': self.split,
            'original_id': self.original_id,
            'db_name': self.db_name,
            'db_path': self.db_path,
            'sql': self.sql,
            'evidence': self.evidence,
            'schemas': self.schemas
        }
        if self.cache_sql_execution:
            d['cache_sql_execution'] = self.cache_sql_execution.to_dict()
        return d


class MongoDBLoader:
    """Loader for normalized Text2SQL MongoDB collections."""

    def __init__(self, db: Database):
        """
        Args:
            db: PyMongo database instance
        """
        self.db = db
        self.instances: Collection = db['instances']
        self.configs: Collection = db['config_profiles']
        self.results: Collection = db['inference_results']

        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes if not exist."""
        self.instances.create_index('instance_id', unique=True)
        self.results.create_index(
            [('instance_id', 1), ('model_name', 1), ('config_profile', 1)],
            unique=True
        )

    # ─────────────────────────────────────────────────────────────────
    # Read operations
    # ─────────────────────────────────────────────────────────────────

    def _shape_instance_id(self, *args, **kwargs) -> str:
        """Helper to shape instance_id from components."""

        if 'instance_id' in kwargs:
            return kwargs['instance_id']

        the_id = kwargs.get('id') or kwargs.get('original_id') or kwargs.get('_id') or kwargs.get('instance_id')
        if the_id is None:
            raise ValueError("One of 'instance_id', 'id', 'original_id', or '_id' must be provided.")

        return f"{kwargs['dataset']}_{kwargs['split']}_{the_id}"

    def get(self, *args, **kwargs) -> Optional[Instance]:
        """Get single instance with its inference results."""
        instance_id = self._shape_instance_id(*args, **kwargs)
        doc = self.instances.find_one({'instance_id': instance_id})
        if not doc:
            return None

        results = list(self.results.find({'instance_id': instance_id}))
        return Instance.from_dict(doc, results)

    def get_by_filter(
        self,
        dataset: str = None,
        split: str = None,
        model_name: str = None,
        config_profile: str = None
    ) -> Iterator[Instance]:
        """Get instances matching filters."""
        query = {}
        if dataset:
            query['dataset'] = dataset
        if split:
            query['split'] = split

        for doc in self.instances.find(query):
            # Build result filter
            result_query = {'instance_id': doc['instance_id']}
            if model_name:
                result_query['model_name'] = model_name
            if config_profile:
                result_query['config_profile'] = config_profile

            results = list(self.results.find(result_query))
            yield Instance.from_dict(doc, results)

    def list_models(self) -> List[str]:
        """Get all unique model names."""
        return self.results.distinct('model_name')

    def list_config_profiles(self) -> List[Dict]:
        """Get all config profiles."""
        return list(self.configs.find({}, {'_id': 1, 'temperature': 1, 'frequency_penalty': 1, 'presence_penalty': 1}))

    # ─────────────────────────────────────────────────────────────────
    # Write operations
    # ─────────────────────────────────────────────────────────────────

    def add_or_update_instance(
        self,
        instance: Instance
    ) -> bool:
        """
        Add or update a single instance.

        Args:
            instance: Instance object to add/update
        Returns:
            True if inserted, False if updated
        """
        raise NotImplementedError("Use add_or_update_inference_result for adding results.")
        query = {'instance_id': instance.instance_id}
        existing = self.instances.find_one(query)

        if existing:
            # Update existing
            self.instances.update_one(
                query,
                {'$set': instance.to_dict()}
            )
            return False
        else:
            # Insert new
            self.instances.insert_one(instance.to_dict())
            return True

    def _config_to_profile(self, temperature: float = None, frequency_penalty: float = None, presence_penalty: float = None) -> str:
        """Helper to convert config to profile ID."""
        if temperature is None and frequency_penalty is None and presence_penalty is None:
            return "default"
        temp = f"{float(temperature):.1f}" if temperature is not None else "None"
        freq = f"{float(frequency_penalty):.1f}" if frequency_penalty is not None else "None"
        pres = f"{float(presence_penalty):.1f}" if presence_penalty is not None else "None"
        return f"t{temp}_f{freq}_p{pres}"

    def add_or_update_inference_result(
        self,
        instance_id: str,
        model_name: str,
        raw_response: str,
        temperature: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        config_profile: Optional[str] = None,
        keep_history: bool = False
    ) -> bool:
        """
        Add or update a single inference result with versioning.

        Args:
            instance_id: e.g., 'bird_dev_123'
            model_name: e.g., 'Arctic-Text2SQL-R1-7B'
            config_profile: Optional profile ID; if None, derived from config
            temperature: Sampling temperature
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty  
            raw_response: Model's raw response
            keep_history: If True, archive old version before updating

        Returns:
            True if inserted, False if updated
        """
        _config_profile = config_profile or self._config_to_profile(
            temperature, frequency_penalty, presence_penalty
        )
        query['config_profile'] = _config_profile

        query = {
            'instance_id': instance_id,
            'model_name': model_name,
            'config_profile': _config_profile
        }

        existing = self.results.find_one(query)
        now = datetime.utcnow()

        if existing:
            # Archive old version if requested
            if keep_history:
                existing['archived_at'] = now
                self.db['inference_results_history'].insert_one(existing)

            # Update with incremented version
            self.results.update_one(
                query,
                {
                    '$set': {
                        'raw_response': raw_response,
                        'updated_at': now
                    },
                    '$inc': {'version': 1}
                }
            )
            return False
        else:
            # Insert new
            self.results.insert_one({
                **query,
                'raw_response': raw_response,
                'version': 1,
                'created_at': now,
                'updated_at': now
            })
            return True

    def add_or_update_batch(
        self,
        results: List[Dict],
        dataset: str = None,
        split: str = None
    ) -> Dict[str, int]:
        """
        Batch add/update inference results.

        Args:
            results: List of dicts with keys:
                - instance_id: str or int (if int, dataset/split required to build compound id)
                - model_name: str
                - raw_response: str
                - config_profile: str (optional, derived from config if not provided)
                - temperature: float (optional)
                - frequency_penalty: float (optional)
                - presence_penalty: float (optional)
            dataset: Dataset name (e.g., 'bird') - required if instance_id is numeric
            split: Split name (e.g., 'dev') - required if instance_id is numeric

        Returns:
            Dict with 'inserted' and 'updated' counts
        """
        from pymongo import UpdateOne

        now = datetime.now(timezone.utc)
        operations = []

        for r in results:
            # Derive config_profile if not provided
            config_profile = r.get('config_profile') or self._config_to_profile(
                r.get('temperature'),
                r.get('frequency_penalty'),
                r.get('presence_penalty')
            )

            # Build compound instance_id if needed
            raw_id = r['instance_id']
            if isinstance(raw_id, int) or (isinstance(raw_id, str) and raw_id.isdigit()):
                if not dataset or not split:
                    raise ValueError(f"dataset and split required when instance_id is numeric: {raw_id}")
                instance_id = f"{dataset}_{split}_{raw_id}"
            else:
                instance_id = raw_id

            query = {
                'instance_id': instance_id,
                'model_name': r['model_name'],
                'config_profile': config_profile
            }

            doc = {
                **query,
                'raw_response': r['raw_response'],
                'updated_at': now
            }

            operations.append(
                UpdateOne(
                    query,
                    {
                        '$set': doc,
                        '$setOnInsert': {'created_at': now},
                        '$inc': {'version': 1}
                    },
                    upsert=True
                )
            )

        if not operations:
            return {'inserted': 0, 'updated': 0}

        result = self.results.bulk_write(operations)
        return {
            'inserted': result.upserted_count,
            'updated': result.modified_count
        }

    def ensure_config_profile(
        self,
        temperature: float,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> str:
        """
        Ensure config profile exists and return its ID.

        Returns:
            Profile ID, e.g., 't0.5_f0_p0'
        """
        profile_id = f"t{temperature}_f{frequency_penalty}_p{presence_penalty}"
        self.configs.replace_one(
            {'_id': profile_id},
            {
                '_id': profile_id,
                'temperature': temperature,
                'frequency_penalty': frequency_penalty,
                'presence_penalty': presence_penalty
            },
            upsert=True
        )
        return profile_id

    # ─────────────────────────────────────────────────────────────────
    # History
    # ─────────────────────────────────────────────────────────────────

    def get_history(
        self,
        instance_id: str,
        model_name: str,
        config_profile: str
    ) -> List[Dict]:
        """Get version history for an inference result."""
        query = {
            'instance_id': instance_id,
            'model_name': model_name,
            'config_profile': config_profile
        }
        history = list(
            self.db['inference_results_history']
            .find(query)
            .sort('version', -1)
        )
        # Include current version
        current = self.results.find_one(query)
        if current:
            history.insert(0, current)
        return history

    # ─────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────

    def model_stats(self) -> Dict[str, int]:
        """Get counts of inference results per model."""
        pipeline = [
            {
                '$group': {
                    '_id': '$model_name',
                    'count': {'$sum': 1}
                }
            }
        ]
        stats = self.results.aggregate(pipeline)
        return {item['_id']: item['count'] for item in stats}

    def stats(self) -> Dict:
        """Get collection statistics."""
        return {
            'instances': self.instances.count_documents({}),
            'config_profiles': self.configs.count_documents({}),
            'inference_results': self.results.count_documents({}),
            'models': len(self.list_models())
        }

    # ─────────────────────────────────────────────────────────────────
    # Migrations
    # ─────────────────────────────────────────────────────────────────

    # def migrate_numeric_instance_ids(self, dataset: str, split: str, dry_run: bool = True) -> Dict[str, int]:
    #     """
    #     Migrate inference_results with numeric instance_id to compound format.

    #     Converts instance_id like 1 or "1" to "bird_dev_1".

    #     Args:
    #         dataset: Dataset name (e.g., 'bird')
    #         split: Split name (e.g., 'dev')
    #         dry_run: If True, only count affected docs without updating

    #     Returns:
    #         Dict with 'matched' and 'modified' counts
    #     """
    #     from pymongo import UpdateOne

    #     # Find docs where instance_id is numeric
    #     numeric_query = {
    #         '$or': [
    #             {'instance_id': {'$type': 'int'}},
    #             {'instance_id': {'$type': 'long'}},
    #             {'instance_id': {'$type': 'double'}},
    #         ]
    #     }

    #     docs = list(self.results.find(numeric_query, {'_id': 1, 'instance_id': 1}))
    #     logger.info(f"Found {len(docs)} docs with numeric instance_id")

    #     if dry_run:
    #         logger.info("Dry run - no changes made")
    #         return {'matched': len(docs), 'modified': 0}

    #     operations = []
    #     for doc in docs:
    #         new_instance_id = self._shape_instance_id(
    #             original_id=doc['instance_id'],
    #             dataset=dataset,
    #             split=split
    #         )
    #         operations.append(
    #             UpdateOne(
    #                 {'_id': doc['_id']},
    #                 {'$set': {'instance_id': new_instance_id}}
    #             )
    #         )

    #     if not operations:
    #         return {'matched': 0, 'modified': 0}

    #     result = self.results.bulk_write(operations)
    #     logger.info(f"Migrated {result.modified_count} docs to compound instance_id format")
    #     return {'matched': len(docs), 'modified': result.modified_count}

    # def migrate_none_config_to_default(self) -> int:
    #     """
    #     Fix config_profile: rename 'tNone_fNone_pNone' to 'default'.

    #     Returns:
    #         Number of documents updated
    #     """
    #     # Update inference_results
    #     result = self.results.update_many(
    #         {'config_profile': 'tNone_fNone_pNone'},
    #         {'$set': {'config_profile': 'default'}}
    #     )
    #     updated = result.modified_count

    #     # Update config_profiles collection
    #     old_config = self.configs.find_one({'_id': 'tNone_fNone_pNone'})
    #     if old_config:
    #         self.configs.replace_one(
    #             {'_id': 'default'},
    #             {
    #                 '_id': 'default',
    #                 'temperature': None,
    #                 'frequency_penalty': None,
    #                 'presence_penalty': None
    #             },
    #             upsert=True
    #         )
    #         self.configs.delete_one({'_id': 'tNone_fNone_pNone'})

    #     logger.info(f"Migrated {updated} inference results to 'default' config_profile")
    #     return updated

    def migrate_cache_sql_execution(
        self,
        dataset: str = None,
        split: str = None,
        batch_size: int = 100,
        num_workers: int = 4,
        runs_per_query: int = 1,
        timeout: int = 30,
        force: bool = False,
        do_tqdm: bool = True
    ) -> Dict[str, int]:
        """
        Cache gold SQL execution results on instances.

        Finds instances with `sql` and `db_path`, executes them,
        and stores result in `cache_sql_execution`.

        Args:
            dataset: Filter by dataset (optional)
            split: Filter by split (optional)
            batch_size: Number of instances to process per batch
            num_workers: Parallel workers for SQL execution
            runs_per_query: Runs per query for timing (default 1 for caching)
            timeout: Query timeout in seconds
            force: Re-execute even if cache exists
            do_tqdm: Show progress bar

        Returns:
            Dict with 'processed', 'cached', 'failed' counts
        """
        query = {'sql': {'$exists': True, '$ne': None}, 'db_path': {'$exists': True, '$ne': None}}
        if dataset:
            query['dataset'] = dataset
        if split:
            query['split'] = split
        if not force:
            query['cache_sql_execution'] = {'$exists': False}

        total = self.instances.count_documents(query)
        logger.info(f"Found {total} instances to cache SQL execution")

        if total == 0:
            return {'processed': 0, 'cached': 0, 'failed': 0}

        worker = SQLWorker(
            num_workers=num_workers,
            runs_per_query=runs_per_query,
            timeout=timeout
        )

        stats = {'processed': 0, 'cached': 0, 'failed': 0}
        cursor = self.instances.find(query, batch_size=batch_size)

        instances_batch = []
        for doc in cursor:
            instances_batch.append(doc)

            if len(instances_batch) >= batch_size:
                self._process_cache_batch(instances_batch, worker, stats, do_tqdm)
                instances_batch = []

        # Process remaining
        if instances_batch:
            self._process_cache_batch(instances_batch, worker, stats, do_tqdm)

        logger.info(f"Cache migration complete: {stats}")
        return stats

    def _process_cache_batch(
        self,
        docs: List[Dict],
        worker: SQLWorker,
        stats: Dict[str, int],
        do_tqdm: bool
    ):
        """Process a batch of instances for caching."""
        from pymongo import UpdateOne

        # Build DBQuery objects
        queries = [
            DBQuery(
                db_name=doc.get('db_name', ''),
                db_path=doc['db_path'],
                query_id=doc['instance_id'],
                query=doc['sql']
            )
            for doc in docs
        ]

        # Execute in parallel
        results = worker.execute_parallel(queries, do_tqdm=do_tqdm)

        # Build updates
        operations = []

        for result in results:
            stats['processed'] += 1

            # Check if result is too large to cache
            if result.success and len(result.results) > MAX_RESULT_ROWS:
                stats['failed'] += 1
                cache_data = {
                    'results': [],
                    'exec_time_ms': result.exec_time_ms,
                    'success': False,
                    'error': f'Result too large ({len(result.results)} rows)'
                }
            elif result.success:
                stats['cached'] += 1
                cache_data = CachedExecutionResult.from_execution_result(result).to_dict()
            else:
                stats['failed'] += 1
                cache_data = CachedExecutionResult.from_execution_result(result).to_dict()

            operations.append(
                UpdateOne(
                    {'instance_id': result.query_id},
                    {'$set': {'cache_sql_execution': cache_data}}
                )
            )

        if operations:
            self.instances.bulk_write(operations)

    def migrate_cache_inference_execution(
        self,
        model_name: str = None,
        config_profile: str = None,
        dataset: str = None,
        split: str = None,
        batch_size: int = 100,
        num_workers: int = 4,
        runs_per_query: int = 1,
        timeout: int = 30,
        force: bool = False,
        do_tqdm: bool = True
    ) -> Dict[str, int]:
        """
        Cache inference result SQL extraction and execution.

        Extracts SQL from raw_response, executes it, and stores:
        - extracted_sql: the parsed SQL
        - execution: CachedExecutionResult dict

        Args:
            model_name: Filter by model (optional)
            config_profile: Filter by config profile (optional)
            dataset: Filter by dataset (optional)
            split: Filter by split (optional)
            batch_size: Instances per batch
            num_workers: Parallel workers for execution
            runs_per_query: Runs per query for timing
            timeout: Query timeout in seconds
            force: Re-process even if cache exists
            do_tqdm: Show progress bar

        Returns:
            Dict with 'processed', 'extracted', 'executed', 'failed_extract', 'failed_exec' counts
        """
        # Build query for inference_results
        query = {}
        if model_name:
            query['model_name'] = model_name
        if config_profile:
            query['config_profile'] = config_profile
        if not force:
            query['cache'] = {'$exists': False}

        # If filtering by dataset/split, get valid instance_ids first
        instance_filter = None
        if dataset or split:
            inst_query = {}
            if dataset:
                inst_query['dataset'] = dataset
            if split:
                inst_query['split'] = split
            instance_filter = set(
                doc['instance_id'] for doc in self.instances.find(inst_query, {'instance_id': 1})
            )

        total = self.results.count_documents(query)
        logger.info(f"Found {total} inference results to process")

        if total == 0:
            return {'processed': 0, 'extracted': 0, 'executed': 0, 'failed_extract': 0, 'failed_exec': 0}

        worker = SQLWorker(
            num_workers=num_workers,
            runs_per_query=runs_per_query,
            timeout=timeout
        )

        stats = {'processed': 0, 'extracted': 0, 'executed': 0, 'failed_extract': 0, 'failed_exec': 0}

        # Build instance_id -> db_path lookup
        db_path_map = {
            doc['instance_id']: doc.get('db_path')
            for doc in self.instances.find({}, {'instance_id': 1, 'db_path': 1})
        }

        cursor = self.results.find(query, batch_size=batch_size)
        batch = []

        for doc in cursor:
            # Filter by dataset/split if needed
            if instance_filter and doc['instance_id'] not in instance_filter:
                continue

            db_path = db_path_map.get(doc['instance_id'])
            if not db_path:
                continue

            batch.append((doc, db_path))

            if len(batch) >= batch_size:
                self._process_inference_cache_batch(batch, worker, stats, do_tqdm)
                batch = []

        if batch:
            self._process_inference_cache_batch(batch, worker, stats, do_tqdm)

        logger.info(f"Inference cache migration complete: {stats}")
        return stats

    def _process_inference_cache_batch(
        self,
        batch: List[tuple],  # List of (doc, db_path)
        worker: SQLWorker,
        stats: Dict[str, int],
        do_tqdm: bool
    ):
        """Process a batch of inference results for caching."""
        from pymongo import UpdateOne

        # Extract SQL from each result
        extracted = []  # (doc, db_path, sql)
        operations_no_sql = []
        
        extractor = SQLExtractor(dialect='sqlite')  # Assuming SQLite dialect; adjust as needed

        for doc, db_path in batch:
            stats['processed'] += 1
            sql = extractor.extract(doc.get('raw_response', ''), timeout=5)
            if sql:
                stats['extracted'] += 1
                extracted.append((doc, db_path, sql))
            else:
                stats['failed_extract'] += 1
                # Cache extraction failure
                operations_no_sql.append(
                    UpdateOne(
                        {'_id': doc['_id']},
                        {'$set': {'cache': {'extracted_sql': None, 'execution': None}}}
                    )
                )

        # Execute extracted SQLs in parallel
        if extracted:
            queries = [
                DBQuery(
                    db_name='',
                    db_path=db_path,
                    query_id=str(doc['_id']),  # Use _id as query_id for mapping
                    query=sql
                )
                for doc, db_path, sql in extracted
            ]

            results = worker.execute_parallel(queries, do_tqdm=do_tqdm)

            # Map results back by _id
            result_map = {r.query_id: r for r in results}

            operations = []
            for doc, db_path, sql in extracted:
                exec_result = result_map.get(str(doc['_id']))

                # Check if result is too large to cache
                if exec_result and exec_result.success and len(exec_result.results) > MAX_RESULT_ROWS:
                    stats['failed_exec'] += 1
                    cache_data = {
                        'extracted_sql': sql,
                        'execution': {
                            'results': [],
                            'exec_time_ms': exec_result.exec_time_ms,
                            'success': False,
                            'error': f'Result too large ({len(exec_result.results)} rows)'
                        }
                    }
                elif exec_result and exec_result.success:
                    stats['executed'] += 1
                    cache_data = {
                        'extracted_sql': sql,
                        'execution': CachedExecutionResult.from_execution_result(exec_result).to_dict()
                    }
                else:
                    stats['failed_exec'] += 1
                    cache_data = {
                        'extracted_sql': sql,
                        'execution': CachedExecutionResult.from_execution_result(exec_result).to_dict() if exec_result else None
                    }

                operations.append(
                    UpdateOne(
                        {'_id': doc['_id']},
                        {'$set': {'cache': cache_data}}
                    )
                )

            if operations:
                self.results.bulk_write(operations)

        if operations_no_sql:
            self.results.bulk_write(operations_no_sql)

    def migrate_inference_by_query(
        self,
        query: Dict,
        batch_size: int = 100,
        num_workers: int = 4,
        runs_per_query: int = 1,
        timeout: int = 30,
        do_tqdm: bool = True
    ) -> Dict[str, int]:
        """
        Re-extract and re-execute inference results matching a custom query.

        Unlike migrate_cache_inference_execution, this accepts any MongoDB query
        and always re-processes matching documents (no 'force' flag needed).

        Args:
            query: MongoDB query dict for inference_results collection
            batch_size: Instances per batch
            num_workers: Parallel workers for execution
            runs_per_query: Runs per query for timing
            timeout: Query timeout in seconds
            do_tqdm: Show progress bar

        Returns:
            Dict with 'processed', 'extracted', 'executed', 'failed_extract', 'failed_exec' counts

        Example:
            # Re-process failed extractions for claude model
            loader.migrate_inference_by_query({
                'model_name': 'claude-3-7-sonnet-20250219',
                'cache.extracted_sql': {'$regex': '```'}  # extraction included markdown
            })
        """
        total = self.results.count_documents(query)
        logger.info(f"Found {total} inference results matching query")

        if total == 0:
            return {'processed': 0, 'extracted': 0, 'executed': 0, 'failed_extract': 0, 'failed_exec': 0}

        worker = SQLWorker(
            num_workers=num_workers,
            runs_per_query=runs_per_query,
            timeout=timeout
        )

        stats = {'processed': 0, 'extracted': 0, 'executed': 0, 'failed_extract': 0, 'failed_exec': 0}

        # Build instance_id -> db_path lookup
        db_path_map = {
            doc['instance_id']: doc.get('db_path')
            for doc in self.instances.find({}, {'instance_id': 1, 'db_path': 1})
        }

        cursor = self.results.find(query, batch_size=batch_size)
        batch = []

        for doc in cursor:
            db_path = db_path_map.get(doc['instance_id'])
            if not db_path:
                continue

            batch.append((doc, db_path))

            if len(batch) >= batch_size:
                self._process_inference_cache_batch(batch, worker, stats, do_tqdm)
                batch = []

        if batch:
            self._process_inference_cache_batch(batch, worker, stats, do_tqdm)

        logger.info(f"Migration by query complete: {stats}")
        return stats
