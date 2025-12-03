import sys

sys.path.append('../../')
sys.path.append('../../src/')

from src.loaders.dataloader import DatasetInstance
from src.loaders.base import BaseLoader
from src.evaluation.metrics.metric import MetricType, Metric
from src.typing.query import DBQuery, TargetPredictedDBQuery
from src.templates.base import BasePromptTemplate
from src.utils.utils import get_db_path
from typing import List, Tuple, Dict, Iterable, Mapping, Any
from enum import Enum
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("SQL extraction timed out.")
    
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
        
    # @property
    # def data(self) -> Any:
    #     if self.loader is None:
    #         raise ValueError("Loader is not defined.")
    #     return self.loader.load_data(self._model, 
    #                                 dataset=self._dataset,
    #                                 temperature=self._temperature,
    #                                 frequency_penalty=self._frequency_penalty,
    #                                 presence_penalty=self._presence_penalty)

    @staticmethod
    def _sql_extraction(data : dict, prompt_template : BasePromptTemplate) -> Dict[int, str]:
        results = {}
        for key in data.keys():
            k = int(key)
            # adding signal timeout for extraction
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # Set the timeout duration (e.g., 5 seconds)
            try:
                sql_query = prompt_template.extract_sql(data[key]['text'])
            except TimeoutError:
                print(f"Timeout occurred while extracting SQL for key: {key}")
                continue
            finally:
                signal.alarm(0)  # Disable the alarm

            results[k] = sql_query
        return results
    
    @staticmethod
    def _instance_find_by_id(instances : DatasetInstance, id : str):
        for instance,p in instances:
            if instance.id == int(id):
                return instance,p
        return None,None
    
    @staticmethod
    def get_target_queries(instances : List[Tuple[DatasetInstance, str]]) -> List[DBQuery]:
        target_queries = []
        for instance, instance_path in instances:
            _id = instance.id
            db_path = get_db_path(instance, instance_path)
            db_name = instance.database['name']
            target_sql = instance.sql

            target_db_query = DBQuery(
                db_name=db_name,
                db_path=db_path,
                query_id=_id,
                query=target_sql
            )
            target_queries.append(target_db_query)
        return target_queries
    
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
                print(f"There is no predicted SQL for id {_id}")
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
        # load data
        data = self.loader.load_data(self._model, 
                                     dataset=self._dataset,
                                     temperature=self._temperature,
                                     frequency_penalty=self._frequency_penalty,
                                     presence_penalty=self._presence_penalty)
        
        # extract sql queries
        extracted_queries = self._sql_extraction(data, self.prompt_template)

        # convert to TargetPredictedDBQuery
        tp_db_queries = self._instance2_dbquery(instances, extracted_queries)
        results = {}
        for metric in self.metrics:
            tp_list = self._target_predicted_dict_to_list(tp_db_queries, dist=Metric.get_name(metric))
            metric_values = metric.compute_many(
                target=self._target_predicted_dict_to_list(tp_db_queries, dist='target'),
                prediction=tp_list
            )
            results[Metric.get_name(metric).value] = metric_values

        return results

        
    
        


    

    