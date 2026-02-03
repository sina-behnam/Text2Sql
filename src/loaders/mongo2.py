import sys
sys.path.append('../../')

from manage import MongoDBConfig
from pymongo import MongoClient
from pymongo import UpdateOne
from loguru import logger

def migrate_model_config_to_dict(collection):
    """
    Migrate model_config from array [temp, freq_pen, pres_pen] to dict format.
    """
    docs = collection.find({'inference_results': {'$exists': True}})

    for doc in docs:
        updated = False
        inference_results = doc.get('inference_results', [])

        for inf in inference_results:
            config = inf.get('model_config')
            # Debug: print type and value
            if isinstance(config, list) and config is not None:
                inf['model_config'] = {
                    'temperature': config[0],
                    'frequency_penalty': config[1],
                    'presence_penalty': config[2]
                }
                updated = True

        if updated:
            collection.update_one(
                {'_id': doc['_id']},
                {'$set': {'inference_results': inference_results}}
            )
            logger.info(f"Migrated doc {doc.get('id')}")

    logger.info("Migration complete")


def add_or_update_inference_result(collection, doc_id: str, new_inference: dict):
    """
    Add or update a single inference result.
    Key = (model_name, temperature, frequency_penalty, presence_penalty)
    """
    def get_config_key(inference: dict) -> tuple:
        config = inference['model_config']
        return (
            inference['model_name'],
            config['temperature'],
            config['frequency_penalty'],
            config['presence_penalty']
        )

    doc = collection.find_one({'id': doc_id}, {'inference_results': 1})
    existing = doc.get('inference_results', []) if doc else []

    # Build dict keyed by (model_name, temp, freq_pen, pres_pen)
    merged = {get_config_key(r): r for r in existing}
    merged[get_config_key(new_inference)] = new_inference

    collection.update_one(
        {'id': doc_id},
        {'$set': {'inference_results': list(merged.values())}},
        upsert=True
    )


def migrate_to_normalized(old_collection, new_db, dataset_name: str, split: str):
    """
    Migrate embedded structure to normalized collections in a new database.

    Args:
        old_collection: Source collection with embedded inference_results
        new_db: Target database for normalized collections
        dataset_name: e.g., 'bird', 'spider'
        split: e.g., 'dev', 'train', 'test'
    """
    instances_col = new_db['instances']
    configs_col = new_db['config_profiles']
    results_col = new_db['inference_results']

    # Create unique index on composite key
    results_col.create_index(
        [('instance_id', 1), ('model_name', 1), ('config_profile', 1)],
        unique=True
    )
    instances_col.create_index('instance_id', unique=True)

    seen_configs = {}

    for doc in old_collection.find():
        # Build composite instance_id
        instance_id = f"{dataset_name}_{split}_{doc['id']}"

        # Insert instance (without inference_results)
        instance = {
            'instance_id': instance_id,
            'dataset': dataset_name,
            'split': split,
            'original_id': doc['id'],
            'db_name': doc.get('db_name'),
            'db_path': doc.get('db_path'),
            'sql': doc.get('sql'),
            'evidence': doc.get('evidence'),
            'schemas': doc.get('schemas')
        }
        instances_col.replace_one({'instance_id': instance_id}, instance, upsert=True)

        # Process inference results
        for inf in doc.get('inference_results', []):
            config = inf.get('model_config', {})

            # Handle both dict and list formats
            if isinstance(config, list):
                temp, freq, pres = config[0], config[1], config[2]
            else:
                temp = config.get('temperature', 0)
                freq = config.get('frequency_penalty', 0)
                pres = config.get('presence_penalty', 0)

            config_key = (temp, freq, pres)

            # Create/get config profile
            if config_key not in seen_configs:
                profile_id = f"t{temp}_f{freq}_p{pres}"
                configs_col.replace_one(
                    {'_id': profile_id},
                    {
                        '_id': profile_id,
                        'temperature': temp,
                        'frequency_penalty': freq,
                        'presence_penalty': pres
                    },
                    upsert=True
                )
                seen_configs[config_key] = profile_id

            # Insert inference result
            results_col.replace_one(
                {
                    'instance_id': instance_id,
                    'model_name': inf['model_name'],
                    'config_profile': seen_configs[config_key]
                },
                {
                    'instance_id': instance_id,
                    'model_name': inf['model_name'],
                    'config_profile': seen_configs[config_key],
                    'raw_response': inf.get('raw_response', '')
                },
                upsert=True
            )

        logger.debug(f"Migrated instance {instance_id}")

    logger.info(f"Migration complete: {dataset_name}_{split}")


if __name__ == "__main__":

    config = MongoDBConfig()

    client = MongoClient(config.db_uri)
    db = client[config.db_name]
    collection = db[config.collection_name]

    projection = {
        "_id": 0,
        "id": 1,
        "sql" : 1,
        "schemas" : 1,
        "evidence": 1,
        "inference_results.model.model_name": 1,
        "inference_results.model.model_config.temperature": 1,
        "inference_results.model.model_config.frequency_penalty": 1,
        "inference_results.model.model_config.presence_penalty": 1,
        "inference_results.predicted_output.raw_response": 1
    }

    docs = collection.find({}, projection)

