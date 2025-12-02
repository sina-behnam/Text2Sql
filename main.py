#!/usr/bin/env python3
"""
Text2SQL Inference Pipeline - Main Entry Point
Uses modular pipeline approach with external model and template configuration
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from loaders.dataloader import DatasetLoader
from pipeline.modular_pipeline import Text2SQLPipeline
from pipeline.steps import (
    GenerateStep, ExtractStep, EvaluateStep, MetricsStep, SaveStep, LoadStep
)


def parse_arguments():
    """Parse command-line arguments for the inference pipeline"""
    parser = argparse.ArgumentParser(
        description='Text2SQL Modular Inference Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --dataset bird --data-path Data/bird_set --config-module my_config

  # Resume from extracted SQLs
  python main.py --dataset bird --data-path Data/bird_set --config-module my_config --resume-from extracted_sqls

  # Run only metrics and save (from existing results)
  python main.py --dataset bird --data-path Data/bird_set --config-module my_config --load-only
        """
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['bird', 'spider', 'spider2-lite'],
        help='Dataset to use for inference'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to dataset directory'
    )

    # Configuration module
    parser.add_argument(
        '--config-module',
        type=str,
        required=True,
        help='Python module path containing get_model_provider() and get_prompt_template() functions (e.g., "config.my_config")'
    )

    # Pipeline control
    parser.add_argument(
        '--resume-from',
        type=str,
        choices=['raw_responses', 'extracted_sqls', 'evaluation_results'],
        help='Resume from intermediate results instead of running generation'
    )

    parser.add_argument(
        '--load-only',
        action='store_true',
        help='Only load existing results and compute metrics (no generation or evaluation)'
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step (only generate and extract SQL)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to instance files'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Base directory to save results (default: ./output)'
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

    # Logging
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (if not specified, logs to console)'
    )

    return parser.parse_args()


def load_config_module(module_path: str):
    """
    Dynamically import configuration module

    The module must provide:
    - get_model_provider() -> ModelProvider
    - get_prompt_template() -> BasePromptTemplate
    - get_model_info() -> dict (optional)
    """
    import importlib

    try:
        config_module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Error: Could not import config module '{module_path}': {e}")
        print(f"\nMake sure the module exists and provides:")
        print("  - get_model_provider() -> ModelProvider")
        print("  - get_prompt_template() -> BasePromptTemplate")
        print("  - get_model_info() -> dict (optional)")
        sys.exit(1)

    # Check required functions
    if not hasattr(config_module, 'get_model_provider'):
        print(f"Error: Config module must provide 'get_model_provider()' function")
        sys.exit(1)

    if not hasattr(config_module, 'get_prompt_template'):
        print(f"Error: Config module must provide 'get_prompt_template()' function")
        sys.exit(1)

    return config_module


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


def build_pipeline(args, config_module, output_dir: str) -> Text2SQLPipeline:
    """Build pipeline based on arguments and configuration"""

    steps = []

    # Get model provider and template
    model_provider = config_module.get_model_provider()
    template = config_module.get_prompt_template()

    # Get model info for saving
    model_info = None
    if hasattr(config_module, 'get_model_info'):
        model_info = config_module.get_model_info()
    else:
        # Default model info
        model_info = {
            'model_name': getattr(model_provider, 'model_name', 'unknown'),
            'model_config': getattr(model_provider.config, 'model_dump', lambda: {})()
        }

    # Build pipeline based on mode
    if args.load_only:
        # Only load existing results and compute metrics
        steps = [
            LoadStep('evaluation_results'),
            MetricsStep(),
        ]
    elif args.resume_from:
        # Resume from intermediate results
        if args.resume_from == 'raw_responses':
            steps.append(LoadStep('raw_responses'))
            steps.append(ExtractStep(template, save_intermediate=True))
            if not args.skip_evaluation:
                steps.append(EvaluateStep(model_provider, do_judge=True, save_intermediate=True))
        elif args.resume_from == 'extracted_sqls':
            steps.append(LoadStep('extracted_sqls'))
            if not args.skip_evaluation:
                steps.append(EvaluateStep(model_provider, do_judge=True, save_intermediate=True))
        elif args.resume_from == 'evaluation_results':
            steps.append(LoadStep('evaluation_results'))

        steps.append(MetricsStep())
    else:
        # Full pipeline
        steps = [
            GenerateStep(model_provider, template, save_intermediate=True),
            ExtractStep(template, save_intermediate=True),
        ]

        if not args.skip_evaluation:
            steps.append(EvaluateStep(model_provider, do_judge=True, save_intermediate=True))

        steps.append(MetricsStep())

    # Add save step if needed
    if not args.no_save:
        steps.append(SaveStep(model_info=model_info))

    # Setup logging config
    logging_config = None
    if args.log_file:
        logging_config = {
            'filename': args.log_file,
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }

    # Create pipeline
    pipeline = Text2SQLPipeline(
        steps=steps,
        output_dir=output_dir,
        logging_config=logging_config
    )

    return pipeline


def main():
    """Main entry point for the inference pipeline"""
    # Parse arguments
    args = parse_arguments()

    print("=" * 80)
    print("Text2SQL Modular Inference Pipeline")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Config Module: {args.config_module}")
    if args.resume_from:
        print(f"Resume Mode: {args.resume_from}")
    elif args.load_only:
        print("Mode: Load and compute metrics only")
    print("=" * 80)

    # Verify dataset path
    if not os.path.exists(args.data_path):
        print(f"Error: Dataset path does not exist: {args.data_path}")
        sys.exit(1)

    print(f"\nLoading dataset from: {args.data_path}")

    # Load dataset
    loader = DatasetLoader(args.data_path)
    instances = loader.load_instances()

    print(f"Loaded {len(instances)} instances")

    # Filter instances
    instances = filter_instances(instances, args)

    if not instances:
        print("Error: No instances to process after filtering")
        sys.exit(1)

    print(f"Processing {len(instances)} instances after filtering")

    # Load configuration module
    print(f"\nLoading configuration from: {args.config_module}")
    config_module = load_config_module(args.config_module)

    # Get model info for output directory naming
    if hasattr(config_module, 'get_model_info'):
        model_info = config_module.get_model_info()
        model_name = model_info.get('model_name', 'unknown')
    else:
        model_provider = config_module.get_model_provider()
        model_name = getattr(model_provider, 'model_name', 'unknown')

    # Prepare output directory
    output_dir = os.path.join(
        args.output_dir,
        model_name.replace('/', '_'),
        args.dataset
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Build pipeline
    print("\nBuilding pipeline...")
    try:
        pipeline = build_pipeline(args, config_module, output_dir)
    except Exception as e:
        print(f"Error building pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Pipeline steps: {len(pipeline)} steps")
    for i, step in enumerate(pipeline.steps):
        print(f"  {i+1}. {step.name}")

    # Run pipeline
    print("\n" + "=" * 80)
    print("Starting Pipeline")
    print("=" * 80 + "\n")

    try:
        context = pipeline.fit_transform(instances)
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print final results
    metrics = context.metadata.get('metrics', {})

    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully")
    print("=" * 80)

    if metrics:
        print(f"\nFinal Metrics:")
        print(f"  Total Evaluated: {metrics.get('num_evaluated', 0)}")
        print(f"  Predictions Generated: {metrics.get('num_with_prediction', 0)}")
        print(f"  Prediction Rate: {metrics.get('prediction_rate', 0):.2%}")
        print(f"  Execution Accuracy: {metrics.get('execution_accuracy', 0):.2%}")
        print(f"  Exact Match Accuracy: {metrics.get('exact_match_accuracy', 0):.2%}")
        print(f"  Semantic Equivalence Accuracy: {metrics.get('semantic_equivalent_accuracy', 0):.2%}")

        # Save metrics to file
        if not args.no_save:
            metrics_file = os.path.join(output_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to: {metrics_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
