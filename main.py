#!/usr/bin/env python3
"""
Text2SQL Inference Pipeline - Main Entry Point
Supports multiple datasets and model configurations via command-line arguments
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import DatasetLoader
from pipeline.text2sql_enricher import Text2SQLInferencePipeline


def parse_arguments():
    """Parse command-line arguments for the inference pipeline"""
    parser = argparse.ArgumentParser(
        description='Text2SQL Inference Pipeline with Multi-Model Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Together.ai model on BIRD dataset
  python main.py --dataset bird --data-path Data/bird_set --model-type together_ai --model-name "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
  
  # Run with Claude (Anthropic) with extended thinking
  python main.py --dataset spider --data-path Data/spider_set --model-type anthropic --model-name "claude-sonnet-4-20250514" --extended-thinking
  
  # Run with local model
  python main.py --dataset spider2-lite --data-path Data/spider2-lite_set --model-type local --model-path "./models/Qwen2.5-Coder-1.5B"
  
  # Specify API key directly
  python main.py --dataset bird --data-path Data/bird_set --model-type together_ai --model-name "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" --api-key "your-key-here"
        """
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['bird', 'spider', 'spider2-lite'],
        help='Dataset to use for inference (bird, spider, or spider2-lite)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Base path to data directory'
    )
    
    # Model configuration arguments
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['together_ai', 'anthropic', 'openai', 'local'],
        help='Type of model provider to use'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        help='Model name (for API-based models)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to local model (for local models only)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for the model provider (can also use environment variables)'
    )
    
    parser.add_argument(
        '--api-key-file',
        type=str,
        help='Path to file containing API key'
    )
    
    parser.add_argument(
        '--extended-thinking',
        action='store_true',
        help='Enable extended thinking/chain-of-thought reasoning'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Maximum tokens to generate (default: 1024)'
    )
    
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=512,
        help='Maximum new tokens for local models (default: 512)'
    )
    
    # Database configuration
    parser.add_argument(
        '--snowflake-config',
        type=str,
        help='Path to Snowflake configuration JSON file',
        required=False,
        default=None
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/app/output',
        help='Directory to save inference results (default: /app/output)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save updated instance files'
    )
    
    # Instance filtering
    parser.add_argument(
        '--instance-id',
        type=int,
        help='Process only a specific instance by ID'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of instances to process'
    )
    
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['simple', 'moderate', 'challenging'],
        help='Filter instances by difficulty level'
    )
    
    return parser.parse_args()


def load_api_key(args) -> Optional[str]:
    """Load API key from arguments, file, or environment variables"""
    # Priority: direct argument > file > environment variable
    
    if args.api_key:
        return args.api_key
    
    if args.api_key_file:
        try:
            with open(args.api_key_file, 'r') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read API key file: {e}")
    
    # Check environment variables based on model type
    env_var_map = {
        'together_ai': 'TOGETHER_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'openai': 'OPENAI_API_KEY'
    }
    
    env_var = env_var_map.get(args.model_type)
    if env_var:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key
    
    return None

def load_snowflake_config(config_path: str) -> Optional[Dict]:
    """Load Snowflake configuration from JSON file"""
    if not config_path or not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load Snowflake config: {e}")
        return None


def create_model_config(args) -> Dict:
    """Create model configuration dictionary from arguments"""
    config = {
        'type': args.model_type,
        'extended_thinking': args.extended_thinking
    }
    
    # Add model-specific parameters
    if args.model_type == 'local':
        if not args.model_path:
            raise ValueError("--model-path is required for local models")
        config['path'] = args.model_path
        config['name'] = Path(args.model_path).name
        config['max_new_tokens'] = args.max_new_tokens
    else:
        if not args.model_name:
            raise ValueError(f"--model-name is required for {args.model_type} models")
        config['name'] = args.model_name
        
        # Load API key
        api_key = load_api_key(args)
        if api_key:
            config['api_key'] = api_key
        else:
            print(f"Warning: No API key provided for {args.model_type}")
    
    # Add token limits
    if args.model_type in ['anthropic']:
        config['max_tokens'] = args.max_tokens
    elif args.model_type in ['together_ai']:
        config['max_tokens'] = args.max_tokens
    
    return config


def filter_instances(instances, args):
    """Filter instances based on arguments"""
    filtered = instances
    
    # Filter by instance ID
    if args.instance_id is not None:
        filtered = [inst for inst in filtered if inst[0].id == args.instance_id]
        if not filtered:
            print(f"Warning: No instance found with ID {args.instance_id}")
            return []
    
    # Filter by difficulty
    if args.difficulty:
        filtered = [(inst, path) for inst, path in filtered 
                    if inst.difficulty == args.difficulty]
    
    # Limit number of instances
    if args.limit:
        filtered = filtered[:args.limit]
    
    return filtered


def main():
    """Main entry point for the inference pipeline"""
    # Parse arguments
    args = parse_arguments()
    
    print("=" * 80)
    print("Text2SQL Inference Pipeline")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model Type: {args.model_type}")
    print(f"Model Name: {args.model_name or args.model_path}")
    print(f"Extended Thinking: {args.extended_thinking}")
    print("=" * 80)
    
    # Get dataset path
    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print(f"\nLoading dataset from: {dataset_path}")
    
    # Load dataset
    loader = DatasetLoader(dataset_path)
    instances = loader.load_instances()
    
    print(f"Loaded {len(instances)} instances")
    
    # Filter instances
    instances = filter_instances(instances, args)
    
    if not instances:
        print("Error: No instances to process after filtering")
        sys.exit(1)
    
    print(f"Processing {len(instances)} instances after filtering")
    
    # Create model configuration
    try:
        model_config = create_model_config(args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    snowflake_config = None
    if args.dataset == 'spider2-lite':
        # Load Snowflake configuration if provided
        snowflake_config = load_snowflake_config(args.snowflake_config)
        if args.snowflake_config and snowflake_config:
            print("Loaded Snowflake configuration")
        else:
            print("Warning: No valid Snowflake configuration provided; database execution will be skipped")
    
    # Initialize pipeline
    print("\nInitializing Text2SQL Inference Pipeline...")
    try:
        pipeline = Text2SQLInferencePipeline(
            model_config=model_config,
            snowflake_config=snowflake_config
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
    # Prepare output directory
    if not args.no_save:
        output_dir = os.path.join(
            args.output_dir,
            f"{args.model_type}_{args.model_name or Path(args.model_path).name}",
            args.dataset
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = None
        print("Results will not be saved (--no-save flag)")
    
    # Run pipeline
    print("\n" + "=" * 80)
    print("Starting Inference Pipeline")
    print("=" * 80 + "\n")
    
    try:
        metrics = pipeline.run_pipeline(
            instances=instances,
            save_updated_files=not args.no_save,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print final results
    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully")
    print("=" * 80)
    print(f"\nFinal Metrics:")
    print(f"  Total Evaluated: {metrics['num_evaluated']}")
    print(f"  Predictions Generated: {metrics['num_with_prediction']}")
    print(f"  Prediction Rate: {metrics['prediction_rate']:.2%}")
    print(f"  Execution Accuracy: {metrics['execution_accuracy']:.2%}")
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
    print(f"  Semantic Equivalence Accuracy: {metrics['semantic_equivalent_accuracy']:.2%}")
    
    # Save metrics to file
    if not args.no_save and output_dir:
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()