import os
import sys
from src.utils.templates.default import DefaultPromptTemplate
from vllm import LLM, SamplingParams
from src.dataloader import DataLoader, Text2SQLDataset
import argparse
from tqdm import tqdm

    
def process_logprobs(logprobs, num_logprobs):

    if logprobs is None:
        return None
    
    _logprobs = []
    for token_logprobs in logprobs:
        ranked_tokens = {}
        for token_id, token_logprob in token_logprobs.items():
            ranked_tokens[token_id] = {
                "prob" : token_logprob.logprob,
                "decoded_token" : token_logprob.decoded_token
            }
        _logprobs.append(ranked_tokens)

    return _logprobs
        
            
def generate_batch(batched_instances, llm ,tokenizer, sampling_params):
    # Extract instance IDs and user messages
    instances_ids = batched_instances['instance_id'].detach().cpu().tolist()
    user_messages = batched_instances['user_message']
    system_messages = batched_instances['system_message']
        
    # Generate chat prompts for all instances
    chat_prompts = []
    for user_message, system_message in zip(user_messages, system_messages):
        chat_prompt = tokenizer.apply_chat_template(
            [   
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            add_generation_prompt=True, 
            tokenize=False
        )
        chat_prompts.append(chat_prompt)

    # Batch generate for all prompts at once
    outputs = llm.generate(chat_prompts, sampling_params=sampling_params)

    # Map outputs back to instance IDs
    batch_responses = {}
    for inst_id, output in zip(instances_ids, outputs):
        
        generated_output = output.outputs[0]
        batch_responses[inst_id] = {
            "text" : generated_output.text,
            "token_ids" : generated_output.token_ids,
            "logprobs" : process_logprobs(generated_output.logprobs, sampling_params.logprobs)
        }

    return batch_responses

def downaload_model(local_dir=""):
    from huggingface_hub import snapshot_download

    model_name = "Snowflake/metallama-Text2SQL-R1-7B"

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
    for batch_responses in batches_responses:
        all_responses.update(batch_responses)
    
    # Actually write to file
    with open(save_path, 'w') as f:
        json.dump(all_responses, f, indent=2)
    
    print(f"Saved responses to {save_path}")

def argument_parser():
    parser = argparse.ArgumentParser(description="metallama Runner")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset file')
    # dialect
    parser.add_argument('--dialect', type=str, default='sqlite', help='SQL dialect to use (default: sqlite)')
    # dataloader params
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for DataLoader (default: 4)')
    # do all in one batch
    parser.add_argument('--all-in-one-batch', action='store_true', help='Process all data in one batch (default: False)')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers for DataLoader (default: 2)')
    parser.add_argument('--num-tensor-parallel', type=int, default=1, help='Number of tensor parallelism (default: 1)')
    # # is shuffle
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the dataset (default: False)') 
    # temperature, frequency_penalty, presence_penalty
    parser.add_argument('--temp', type=float, default=0.2, help='Temperature for model generation (default: 0.2)')
    parser.add_argument('--fp', type=float, default=0.0, help='Frequency penalty for model generation (default: 0.0)')
    parser.add_argument('--pp', type=float, default=0.0, help='Presence penalty for model generation (default: 0.0)')
    parser.add_argument('--logprobs', type=int, default=5, help='Number of logprobs to return (default: 5)')
    # add model path
    parser.add_argument('--model-path', type=str, default='./models/metallama-7B', help='Path to the metallama model (default: ./models/metallama-7B)')
    # save directory
    parser.add_argument('--save-dir', type=str, default='./metallama_results', help='Directory to save results (default: ./metallama_results)')
    # add examples to help
    parser.epilog = '''Example usage: \n
    python metallamat_runner.py --data-path ./data/bird_dev.json --dialect sqlite --batch-size 4 --temp 0.2 --fp 0.0 --pp 0.0 --model-path ./models/metallama-Text2SQL-R1-7B --save-dir ./metallama_results
    '''
    
    return parser

# At the end of metallama_runner.py, replace the if __name__ == "__main__" section:

def run_metallama(data_path, model_path, dialect='sqlite', batch_size=4, 
               temperature=0.2, frequency_penalty=0.0, presence_penalty=0.0,
               logprobs=5, save_dir='./metallama_results', num_tensor_parallel=1,
               shuffle=False, num_workers=2, all_in_one_batch=False):
    """Run metallama inference with given parameters"""
    
    # make save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prompt_template = DefaultPromptTemplate()
    dataset = Text2SQLDataset(data_path, template=prompt_template, dialect=dialect)

    if all_in_one_batch:
        batch_size = len(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    print("Dataset loaded. Number of samples:", len(dataset))

    llm = LLM(
        model=model_path,
        dtype="float16", 
        tensor_parallel_size=num_tensor_parallel,
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        swap_space=1,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=1024,
        logprobs=logprobs 
    )

    tokenizer = llm.get_tokenizer()

    print("Model and tokenizer loaded.")
    print("Model Sampling Parameters:", sampling_params)

    batches_responses = []

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
        batch_responses = generate_batch(batch, llm, tokenizer, sampling_params)
        batches_responses.append(batch_responses)

        # Save batch responses
        save_path = os.path.join(save_dir, f"metallama_responses_batch_{batch_idx}.json")
        save_batch_responses_json([batch_responses], save_path)

    # Save all responses
    save_path = os.path.join(save_dir, f"metallama_responses_all_batches.json")
    save_batch_responses_json(batches_responses, save_path)


if __name__ == "__main__":
    args = argument_parser().parse_args()
    run_metallama(
        data_path=args.data_path,
        model_path=args.model_path,
        dialect=args.dialect,
        batch_size=args.batch_size,
        temperature=args.temp,
        frequency_penalty=args.fp,
        presence_penalty=args.pp,
        logprobs=args.logprobs,
        save_dir=args.save_dir,
        num_tensor_parallel=args.num_tensor_parallel,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        all_in_one_batch=args.all_in_one_batch
    )