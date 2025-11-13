import os
import sys
from src.utils.templates.base import BasePromptTemplate
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.dataloader import DataLoader, Text2SQLDataset
import argparse
from tqdm import tqdm

input_prompt_template = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}
{evidence}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.'''


class OmniSQLPromptTemplate(BasePromptTemplate):

    def create_prompt(self, question, schema, dialect=None, evidence = None):
        
        usr_msg = input_prompt_template.format(
            db_details = schema,
            question = question,
            evidence = '' if evidence is None else f'(PS :\n{evidence})'
        )

        return '', usr_msg, ''
    
    def extract_sql(self, response_text, clean = True):
        
        extractor = self.SQLExtractorHelper()

        sql = extractor._try_extraction_methods([
            extractor._try_code_block_extraction,
        ]
            ,response_text)
        
        if self._is_valid_sql(sql):
            return self._clean_sql(sql)
        
        return None

def generate_batch(batched_instances, llm ,tokenizer, sampling_params):
    # Extract instance IDs and user messages
    instances_ids = batched_instances['instance_id'].detach().cpu().tolist()
    user_messages = batched_instances['user_message']
        
    # Generate chat prompts for all instances
    chat_prompts = []
    for user_message in user_messages:
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True, 
            tokenize=False
        )
        chat_prompts.append(chat_prompt)
        
    # Batch generate for all prompts at once
    outputs = llm.generate(chat_prompts, sampling_params=sampling_params)

    # Map outputs back to instance IDs
    batch_responses = {}
    for inst_id, output in zip(instances_ids, outputs):
        # Get the first (and typically only) response
        batch_responses[inst_id] = output.outputs[0].text

    return batch_responses   

def downaload_model():
    from huggingface_hub import snapshot_download

    model_name = "seeklhy/OmniSQL-7B"
    local_dir = "./models/OmniSQL-7B"

    print("Downloading model...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("Download complete!")

    return local_dir

def save_batch_responses_json(batches_responses, save_path):
    import json

    all_responses = {}
    for batch_response in batches_responses:
        all_responses.update(batch_response)

    with open(save_path, 'w') as f:
        json.dump(all_responses, f, indent=4)

    print(f"Saved responses to {save_path}")

def argument_parser():
    parser = argparse.ArgumentParser(description="OmniSQL Runner")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file')
    # dialect
    parser.add_argument('--dialect', type=str, default='sqlite', help='SQL dialect to use (default: sqlite)')
    # dataloader params
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader (default: 4)')
    # do all in one batch
    parser.add_argument('--all_in_one_batch', action='store_true', help='Process all data in one batch (default: False)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader (default: 2)')
    # # is shuffle
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the dataset (default: False)') 
    # temperature, frequency_penalty, presence_penalty
    parser.add_argument('--temp', type=float, default=0.2, help='Temperature for model generation (default: 0.2)')
    parser.add_argument('--fp', type=float, default=0.0, help='Frequency penalty for model generation (default: 0.0)')
    parser.add_argument('--pp', type=float, default=0.0, help='Presence penalty for model generation (default: 0.0)')
    # add model path
    parser.add_argument('--model_path', type=str, default='./models/OmniSQL-7B', help='Path to the OmniSQL model (default: ./models/OmniSQL-7B)')
    # add examples to help
    parser.epilog = '''Example usage: \n
    python omnisql_runner.py --data_path ./Data/v3_claude/bird_set_stratified \
          --model_path ./models/OmniSQL-7B \
          --dialect sqlite --batch_size 4 \
          --num_workers 2 --shuffle --temp 0.2 --fp 0.0 --pp 0.0
    ''' 
    
    return parser

def main():

    args = argument_parser().parse_args()

    data_path = args.data_path
    model_path = args.model_path
    dialect = args.dialect 

    temperature = args.temp
    frequency_penalty = args.fp
    presence_penalty = args.pp

    prompt_template = OmniSQLPromptTemplate()

    dataset = Text2SQLDataset(data_path, template=prompt_template, dialect=dialect)

    batch_size = args.batch_size
    if args.all_in_one_batch:
        batch_size = len(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers
    )

    print("Dataset loaded. Number of samples:", len(dataset))

    llm = LLM(
        model = model_path,
        dtype = "float16", 
        tensor_parallel_size = 1,
        max_model_len = 8192,
        gpu_memory_utilization = 0.92,
        swap_space = 1,
        enforce_eager = True,
        disable_custom_all_reduce = True,
        trust_remote_code = True
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=2048
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Model and tokenizer loaded.")
    print("Model Sampling Parameters:", sampling_params)

    batches_reposenses = []

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):

        batch_responses = generate_batch(batch, llm, tokenizer, sampling_params)
        
        batches_reposenses.append(batch_responses)

        save_path = os.path.join(f"omisql_responses_batch_{batch_idx}.json")
        save_batch_responses_json([batch_responses], save_path)

    save_path = os.path.join("omisql_responses.json")
    save_batch_responses_json(batches_reposenses, save_path)

if __name__ == "__main__":
    main()


# next(iter(dataloader))
