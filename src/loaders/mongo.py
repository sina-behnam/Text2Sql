from typing import List, Tuple, Dict, Any
from pymongo import MongoClient, errors, ASCENDING
from src.loaders.base import BaseLoader

class MongoDBDataLoader(BaseLoader):
    
    def __init__(self, db_uri: str, db_name: str, collection_name: str):
        
        """Initialize the MongoDBDataLoader with connection parameters.
        
        Args:
            db_uri (str): The MongoDB connection URI.
            db_name (str): The name of the database to connect to.
            collection_name (str): The name of the collection to fetch data from.
        """
        self.db_uri = db_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self._connect()
        # self._create_index()

    def _create_index(self):
        """Create indexes for efficient querying."""
        # Unique composite index on instance_id + dataset
        self.collection.create_index(
            [("unique_id", ASCENDING)], 
            unique=True
        )
        
        # Additional indexes for common queries
        self.collection.create_index([("id", ASCENDING), ("dataset", ASCENDING)])
        self.collection.create_index([("dataset", ASCENDING)])
        self.collection.create_index([("database.name", ASCENDING)])

    def _connect(self):
        """Establish a connection to the MongoDB database."""
        try:
            self.client = MongoClient(self.db_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
        except errors.ConnectionError as e:
            print(f"Error connecting to MongoDB: {e}")
            raise e
    
    @staticmethod
    def pipeline(model: str, dataset: str , **configs) -> List[Dict]:
        match_stage = {
                '$match': {
                    'dataset': dataset,
                    'inference_results': {
                        '$elemMatch': {
                            'has_prediction': True,
                            'model.model_name': model,
                            '$or': [
                                {
                                    # Match by specific config parameters
                                    '$and': [
                                        {'model.model_config.temperature': {'$exists': True}},
                                        {'model.model_config.frequency_penalty': {'$exists': True}},
                                        {'model.model_config.presence_penalty': {'$exists': True}},
                                        {'model.model_config.temperature': configs.get('temperature')},
                                        {'model.model_config.frequency_penalty': configs.get('frequency_penalty')},
                                        {'model.model_config.presence_penalty': configs.get('presence_penalty')}
                                    ]
                                },
                                {
                                    # Match by profile
                                    'model.model_config.profile': configs.get('profile')
                                }
                            ]
                        }
                    }
                }
            }
        
        addfields_stage = {
                '$addFields': {
                    'inference_results': {
                        '$filter': {
                            'input': '$inference_results',
                            'cond': {
                                '$and': [
                                    {'$eq': ['$$this.has_prediction', True]},
                                    {'$eq': ['$$this.model.model_name', model]},
                                    {
                                        '$or': [
                                            {
                                                # Filter by config parameters
                                                '$and': [
                                                    {'$eq': [{'$ifNull': ['$$this.model.model_config.temperature', None]}, {'$ifNull': [configs.get('temperature'), None]}]},
                                                    {'$eq': [{'$ifNull': ['$$this.model.model_config.frequency_penalty', None]}, {'$ifNull': [configs.get('frequency_penalty'), None]}]},
                                                    {'$eq': [{'$ifNull': ['$$this.model.model_config.presence_penalty', None]}, {'$ifNull': [configs.get('presence_penalty'), None]}]}
                                                ]
                                            },
                                            {
                                                # Filter by profile
                                                '$eq': ['$$this.model.model_config.profile', configs.get('profile')]
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                }
            }

        project_stage = {
                '$project': {
                    'unique_id': 1,
                    'id': 1,
                    'inference_results.model.model_name': 1,
                    'inference_results.model.model_config.temperature': 1,
                    'inference_results.model.model_config.frequency_penalty': 1,
                    'inference_results.model.model_config.presence_penalty': 1,
                    'inference_results.model.model_config.profile': 1,
                    'inference_results.predicted_output.raw_response': 1
                }
            }
        return [match_stage, addfields_stage, project_stage]
        
    def load_data(self, model : str, dataset : str , **configs) -> List:
        """
        Loading the data from MongoDB
        """
        docs = self.collection.aggregate(MongoDBDataLoader.pipeline(model, dataset, profile='default' ,**configs))

        results = {}
        for d in docs:
            _id = int(d['id'])
            try:
                results[_id] = {'text' : d['inference_results'][0]['predicted_output']['raw_response']} # Assuming only one matching inference result
            except (IndexError, KeyError):
                print(f"Warning: No matching inference result for document ID {_id}. Skipping.")
                continue

        return results