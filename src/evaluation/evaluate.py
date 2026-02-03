import json
import sys
import signal
from typing import List, Tuple, Literal
from dataclasses import dataclass
from tqdm import tqdm
from sqlglot import parse_one

# Project imports
sys.path.append('../../')
sys.path.append('../../src/')
sys.path.append('../../src/utils')

from src.loaders.dataloader import DatasetLoader, DatasetInstance
from src.loaders.mongodb_loader import MongoDBLoader
from src.typing.query import DBQuery
from src.evaluation.helpers.dict_schema import extract_sqlite_schema
from src.evaluation.simple_sem import SemnticalBasedEvaluator
from src.evaluation.simple_syntax import QuerySyntaxBasedEvaluator
from src.evaluation.simple_exec import ResultBasedEvaluator
from src.workers.sql_worker import SQLWorker
from src.utils.utils import get_db_path
from src.templates.arctic import ArcticText2SQLTemplate
from src.templates.omnisql import OmniSQLPromptTemplate
from src.templates.default import DefaultPromptTemplate
from src.templates.json_template import JSONText2SQLTemplate
from manage import MongoDBConfig, BIRD_DATA_PATH
from sqlglot import parse_one

@dataclass
class ModelPrediction:
    prediction: DBQuery | None
    model_name: str = 'unknown_model'
    model_config: tuple | str = 'default'
    results: dict = None

    def print_query(self):
        if self.prediction and self.prediction.query:
            print(parse_one(self.prediction.query).sql(pretty=True))
        else:
            print("No prediction available.")


@dataclass
class ModelsPredictionsPerInstance:
    instance_id: str
    gold_query: DBQuery
    model_predictions: list[ModelPrediction]

    @property
    def db_path(self):
        return self.gold_query.db_path

    @property
    def schema(self):
        return extract_sqlite_schema(self.gold_query.db_path)
    
    def print_gold(self):
        if self.gold_query and self.gold_query.query:
            print(parse_one(self.gold_query.query).sql(pretty=True))
        else:
            print("No gold query available.")


def find_instance_by_id(instances: List[Tuple[DatasetInstance, str]], instance_id: str):
    for instance, path in instances:
        if instance.id == instance_id:
            return instance, path
    print(f"Instance id {instance_id} not found")
    return None


def extract_model_results(inference_results: list) -> dict:
    per_q_models_res = {}
    
    for res in inference_results:
        model = res.get('model', {}).get('model_name', 'unknown_model')
        config = res.get('model', {}).get('model_config', {})
        config = (config.get('temperature'), config.get('frequency_penalty'), 
                 config.get('presence_penalty')) if config else 'default'
        
        if not res.get('has_prediction', False):
            per_q_models_res[(model, config)] = None
            continue
            
        prediction = res.get('predicted_output', {}).get('raw_response')
        per_q_models_res[(model, config)] = prediction
    
    return per_q_models_res

class SimpleJsonTemplate:
    """A simple JSON template for extracting SQL from model responses."""
    
    def extract_sql(self, raw_response: str) -> str:
        try:
            response_json = json.loads(raw_response)
            sql_query = response_json.get("sql_query", None)
            if sql_query is None and "error" in response_json:
                print(f"Model returned an error: {response_json['error']}")
                return None
            return sql_query
        except json.JSONDecodeError:
            print("Failed to decode JSON from response.")
            return None


def prompt_template_selector(model_name: str) -> DefaultPromptTemplate:
    model_name = model_name.lower()
    if 'arctic' in model_name:
        return ArcticText2SQLTemplate()
    elif 'omnisql' in model_name:
        return OmniSQLPromptTemplate()
    elif 'qwen' in model_name:
        return JSONText2SQLTemplate()
    elif 'mixtral' in model_name:
        return SimpleJsonTemplate()
    return DefaultPromptTemplate()


def sql_extraction(raw_response: str, prompt_template: DefaultPromptTemplate) -> str:
    def timeout_handler(signum, frame):
        raise TimeoutError
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        sql_query = prompt_template.extract_sql(raw_response)
    except TimeoutError:
        print("Timeout during SQL extraction")
        sql_query = None
    finally:
        signal.alarm(0)
    return sql_query


def models_dict_to_predictions(instances: List[Tuple[DatasetInstance, str]], 
                               sample_instance: dict, 
                               models_dict: dict) -> ModelsPredictionsPerInstance:
    instance_id = sample_instance['id']
    dataset_instance, dataset_instance_path = find_instance_by_id(instances, instance_id)
    db_path = get_db_path(dataset_instance, dataset_instance_path)
    db_name = dataset_instance.database['name']
    gold_query = DBQuery(db_name=db_name, db_path=db_path, 
                        query_id=instance_id, query=sample_instance['sql'])
    
    model_predictions = []
    for (model_name, model_config), prediction in models_dict.items():
        prediction_id = hash((gold_query.query_id, model_name, model_config))
        
        if prediction is None:
            db_query = DBQuery(db_name=gold_query.db_name, db_path=gold_query.db_path,
                             query_id=prediction_id, query=None)
        else:
            prompt_template = prompt_template_selector(model_name)
            sql_query = sql_extraction(prediction, prompt_template)
            db_query = DBQuery(db_name=gold_query.db_name, db_path=gold_query.db_path,
                             query_id=prediction_id, query=sql_query)
        
        model_pred = ModelPrediction(prediction=db_query, model_name=model_name, 
                                    model_config=model_config)
        model_predictions.append(model_pred)
    
    return ModelsPredictionsPerInstance(instance_id=instance_id, gold_query=gold_query,
                                       model_predictions=model_predictions)


def semantic_evaluation(inst_predictions: ModelsPredictionsPerInstance, semaneval: SemnticalBasedEvaluator, sql_vocab):
    """Evaluate predictions using semantic similarity."""
    for mp in inst_predictions.model_predictions:
        if mp.prediction.query is None:
            continue
        mp.results = semaneval(inst_predictions.gold_query.query, mp.prediction.query,
                                vocabulary=sql_vocab, method='cosine')
        

def syntax_evaluation(inst_predictions: ModelsPredictionsPerInstance, syntax_evaluator: QuerySyntaxBasedEvaluator):
    """Evaluate predictions using syntax-based metrics."""
    for mp in inst_predictions.model_predictions:
        if mp.prediction.query is None:
            continue
        mp.results = syntax_evaluator(inst_predictions.gold_query, mp.prediction)


def execution_evaluation(inst_predictions: ModelsPredictionsPerInstance, 
                        sql_worker: SQLWorker, 
                        result_evaluator: ResultBasedEvaluator):
    """Evaluate predictions by executing queries and comparing results."""
    target_re = sql_worker.execute_single(inst_predictions.gold_query)
    predictions_re = sql_worker.execute_parallel([mp.prediction for mp in inst_predictions.model_predictions], 
                                                 do_tqdm=False)
    
    for pred_re in predictions_re:
        for mp in inst_predictions.model_predictions:
            if mp.prediction.query is None:
                continue
            if mp.prediction.query_id == pred_re.query_id:
                mp.results = result_evaluator.evaluate(target_re, pred_re)
                break


def save_to_mongo(instances_predictions: List[ModelsPredictionsPerInstance], 
                 mongo_config: MongoDBConfig, collection_name: str):
    """Save evaluation results to MongoDB."""
    mongo_loader = MongoDBLoader(db_name=mongo_config.db_name, 
                                    db_uri=mongo_config.db_uri)
    collection = mongo_loader.db[collection_name]
    
    for inst_pred in instances_predictions:
        doc = {
            "instance_id": inst_pred.instance_id,
            "gold_query": {
                "db_name": inst_pred.gold_query.db_name,
                "db_path": inst_pred.gold_query.db_path,
                "query_id": inst_pred.gold_query.query_id,
                "query": inst_pred.gold_query.query
            },
            "model_predictions": []
        }
        
        for mp in inst_pred.model_predictions:
            mp_doc = {
                "model_name": mp.model_name,
                "model_config": mp.model_config,
                "prediction": {
                    "query_id": mp.prediction.query_id,
                    "query": mp.prediction.query
                } if mp.prediction else None,
                "results": mp.results
            }
            doc["model_predictions"].append(mp_doc)
        
        collection.insert_one(doc)


def evaluate(dataset_path: str = BIRD_DATA_PATH,
            dataset_name: str = "bird",
            eval_type: Literal["semantic", "syntax", "execution"] = "semantic",
            collection_name: str = None,
            num_workers: int = 4,
            runs_per_query: int = 20,
            timeout: int = 10,
            max_try_timeout: int = 3):
    """
    Unified evaluation function supporting semantic, syntax, and execution evaluation.
    
    Args:
        dataset_path: Path to dataset
        dataset_name: Name of dataset (e.g., "bird", "spider")
        eval_type: Type of evaluation ("semantic", "syntax", "execution")
        collection_name: MongoDB collection name for saving results
        num_workers: Number of workers for parallel execution (execution eval only)
        runs_per_query: Number of runs per query (execution eval only)
        timeout: Query execution timeout (execution eval only)
        max_try_timeout: Max timeout retries (execution eval only)
    """
    # Set default collection name based on eval type
    if collection_name is None:
        collection_name = f"{dataset_name}_{eval_type}_evaluation_results"
    
    # Load dataset
    loader = DatasetLoader(data_path=dataset_path)
    dataset_instances = loader.load_instances()
    
    # Setup MongoDB
    mongo_config = MongoDBConfig()
    mongo_loader = MongoDBLoader(**mongo_config.model_dump())
    mongo_instances = mongo_loader.collection.find(filter={"dataset": dataset_name})
    
    # Initialize evaluators based on type
    syntax_evaluator = None
    sql_worker = None
    result_evaluator = None
    
    if eval_type == "syntax":
        syntax_evaluator = QuerySyntaxBasedEvaluator()
    elif eval_type == "execution":
        sql_worker = SQLWorker(num_workers=num_workers, 
                              runs_per_query=runs_per_query,
                              timeout=timeout, 
                              max_try_timeout=max_try_timeout)
        result_evaluator = ResultBasedEvaluator()
    elif eval_type == "semantic":
        semaneval = SemnticalBasedEvaluator()
    
    # Process instances
    instances_predictions = []
    progress = tqdm(mongo_instances, desc=f"Evaluating {dataset_name} instances ({eval_type})")
    
    for sample_instance in progress:
        progress.set_description(f"Evaluating {dataset_name} instance {sample_instance.get('id')} ({eval_type})")
        
        model_results = extract_model_results(sample_instance['inference_results'])
        inst_predictions = models_dict_to_predictions(dataset_instances, sample_instance, model_results)
        
        # Perform evaluation based on type
        if eval_type == "semantic":
            sql_vocab = semaneval.build_vocabulary([inst_predictions.gold_query.query] + 
                                                 [mp.prediction.query for mp in inst_predictions.model_predictions if mp.prediction.query is not None])
            semantic_evaluation(inst_predictions, semaneval, sql_vocab)
        elif eval_type == "syntax":
            syntax_evaluation(inst_predictions, syntax_evaluator)
        elif eval_type == "execution":
            execution_evaluation(inst_predictions, sql_worker, result_evaluator)
        
        instances_predictions.append(inst_predictions)
    
    # Save results
    save_to_mongo(instances_predictions, mongo_config, collection_name)
    mongo_loader.client.close()
    
    print(f"Evaluation complete. Results saved to {collection_name}")


if __name__ == "__main__":
    # Run semantic evaluation
    evaluate(eval_type="semantic")
    
    # Run syntax evaluation
    # evaluate(eval_type="syntax")
    
    # Run execution evaluation
    # evaluate(eval_type="execution", num_workers=4, runs_per_query=20)