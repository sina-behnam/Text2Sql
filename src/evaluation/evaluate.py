import sys

sys.path.append('../../')
sys.path.append('../../src/')

import json 
import os

from src.dataloader import DatasetInstance
from src.evaluation.metrics.metric import MetricType, Metric
from src.typing.query import DBQuery, TargetPredictedDBQuery
from src.templates.base import BasePromptTemplate
from src.utils.utils import get_db_path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Mapping, Any
from enum import Enum

from pymongo import MongoClient, errors, ASCENDING

class BaseLoader:

    def __init__(self):
        pass

    def format_data(self, *args, **kwargs) -> Any:
        pass

    def load_data(self, *args, **kwargs) -> Any:
        raise NotImplementedError("load_data method must be implemented by subclasses.")

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
        
    def load_data(self, model : str, dataset : str , **configs) -> List:
        """
        Loading the data from MongoDB
        """

        pipeline = [
            {
                '$match': {
                    'dataset': dataset,
                    'inference_results': {
                        '$elemMatch': {
                            'has_prediction': True,
                            'model.model_name': model,
                            'model.model_config.temperature': configs.get('temperature'),
                            'model.model_config.frequency_penalty': configs.get('frequency_penalty'),
                            'model.model_config.presence_penalty': configs.get('presence_penalty'),
                        }
                    }
                }
            },
            {
                '$addFields': {
                    'inference_results': {
                        '$filter': {
                            'input': '$inference_results',
                            'cond': {
                                '$and': [
                                    {'$eq': ['$$this.has_prediction', True]},
                                    {'$eq': ['$$this.model.model_name', model]},
                                    {'$eq': ['$$this.model.model_config.temperature', configs.get('temperature')]},
                                    {'$eq': ['$$this.model.model_config.frequency_penalty', configs.get('frequency_penalty')]},
                                    {'$eq': ['$$this.model.model_config.presence_penalty', configs.get('presence_penalty')]}
                                ]
                            }
                        }
                    }
                }
            },
            {
                '$project': {
                    'unique_id': 1,
                    'id': 1,
                    # 'inference_results.model.model_name': 1,
                    # 'inference_results.model.model_config.temperature': 1,
                    # 'inference_results.model.model_config.frequency_penalty': 1,
                    # 'inference_results.model.model_config.presence_penalty': 1,
                    'inference_results.predicted_output.raw_response': 1
                }
            }
        ]

        docs = self.collection.aggregate(pipeline)

        results = {}
        for d in docs:
            _id = int(d['id'])
            try:
                results[_id] = {'text' : d['inference_results'][0]['predicted_output']['raw_response']} # Assuming only one matching inference result
            except (IndexError, KeyError):
                print(f"Warning: No matching inference result for document ID {_id}. Skipping.")
                continue

        return results

class JSONFileDataLoader(BaseLoader):
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    def _conversion_format(self, data: Dict[str, Any]) -> Dict[int, Any]:
        converted_data = {}
        for key, value in data.items():
            converted_data[int(key)] = value
        return converted_data

    def load_data(self) -> Dict[str, Any]:
        """Load data from a JSON file."""
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return self._conversion_format(data)
    
class Evaluate():

    class QueryType(Enum):
        TARGET = 'target'
        PREDICTED = 'predicted'
        BOTH = 'both'

    def __init__(self, 
                metrics: Iterable[Metric],
                loader : BaseLoader = None,
                prompt_template: BasePromptTemplate = None,
                *args, **kwargs):

        self.metrics = list(metrics)
        self.loader = loader
        self.prompt_template = prompt_template
        
        # model, config params for mongo loader
        self._model = kwargs.get('model', None) 
        self._dataset = kwargs.get('dataset', None)
        self._temperature = kwargs.get('temperature', None)
        self._frequency_penalty = kwargs.get('frequency_penalty', None)
        self._presence_penalty = kwargs.get('presence_penalty', None)
        
    @property
    def data(self) -> Any:
        if self.loader is None:
            raise ValueError("Loader is not defined.")
        return self.loader.load_data(self._model, 
                                    dataset=self._dataset,
                                    temperature=self._temperature,
                                    frequency_penalty=self._frequency_penalty,
                                    presence_penalty=self._presence_penalty)

    @staticmethod
    def _sql_extraction(data : dict, prompt_template : BasePromptTemplate) -> Dict[int, str]:
        results = {}
        for key in data.keys():
            k = int(key)
            sql_query = prompt_template.extract_sql(data[key]['text'])
            results[k] = sql_query
        return results
    
    @staticmethod
    def _instance_find_by_id(instances : DatasetInstance, id : str):
        for instance,p in instances:
            if instance.id == int(id):
                return instance,p
        return None,None
    
    @staticmethod
    def _instance2_dbquery(instances: List[Tuple[DatasetInstance, str]], queries: Dict) -> Dict[int, TargetPredictedDBQuery]:

        _target_predicted_db_queries = {}
        for instance, instance_path in instances:

            _id = instance.id
            db_path = get_db_path(instance, instance_path)
            db_name = instance.database['name']
            target_sql = instance.sql

            predicted_sql = queries.get(_id, None)
            if predicted_sql is None:
                continue

            target_db_query = DBQuery(
                db_name=db_name,
                db_path=db_path,
                query_id=_id,
                query=target_sql
            )
            predicted_db_query = DBQuery(
                db_name=db_name,
                db_path=db_path,
                query_id=_id,
                query=predicted_sql
            )
            
            _target_predicted_db_queries[_id] = TargetPredictedDBQuery(
                target=target_db_query,
                predicted=predicted_db_query
            )

        return _target_predicted_db_queries
    
    @staticmethod
    def _target_predicted_dict_to_list(tp_dict: Dict[int, TargetPredictedDBQuery], dist : str | QueryType = 'target') -> List[DBQuery]:
        if dist not in Evaluate.QueryType._value2member_map_:
            raise ValueError(f"Invalid dist value: {dist}. Must be one of {[e.value for e in Evaluate.QueryType]}")
        
        tp_list = []
        for _id, tp in tp_dict.items():
            if dist == 'target':
                tp_list.append(tp.target)
            elif dist == 'predicted':
                tp_list.append(tp.predicted)
            else:
                tp_list.append(tp)
        return tp_list

    def evaluate(self,
                instances: List[Tuple[DatasetInstance, str]]) -> Mapping[str, float | int | List[float | int]]:
        """Evaluate the extracted SQL queries against the target queries using the specified metrics.
        Args:
            instances (List[Tuple[DatasetInstance, str]]): List of dataset instances along with their file paths.
        Returns:
            Mapping[str, float | int | List[float | int]]: A dictionary containing the computed metric values.
        """
        extracted_sqls = self._sql_extraction(self.data,self.prompt_template)
        
        #get target-predicted dict
        tp_dict = self._instance2_dbquery(instances, extracted_sqls)
    
        #get 
        target_queries = self._target_predicted_dict_to_list(tp_dict, dist='target')
        predicted_queries = self._target_predicted_dict_to_list(tp_dict, dist='predicted')

        results = {}
        for metric in self.metrics:
            results[metric.get_name()] = metric.compute(
                target=target_queries,
                prediction=predicted_queries
            )
        
        return results

    
        


    

    