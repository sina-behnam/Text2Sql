from processors import DataProcessor
import argparse

def main():
    """Main function with argument parsing for dataset processing"""
    parser = argparse.ArgumentParser(description='Process text-to-SQL datasets')
    
    # Common arguments
    parser.add_argument('--dataset', required=True, choices=['spider', 'bird', 'spider2'],
                       help='Dataset to process')
    parser.add_argument('--dataset-path', required=True,
                       help='Path to the dataset')
    parser.add_argument('--split', default='dev',
                       help='Dataset split to process (default: dev)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of instances to process')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for processed data')
    parser.add_argument('--save-to-file', action='store_true',
                       help='Save processed data to file')
    
    # Spider2 specific argument
    parser.add_argument('--dataset-type', default='lite',
                       help='Dataset type for Spider2 (default: lite)')
    
    args = parser.parse_args()
    
    processor = DataProcessor()
    
    # Build kwargs based on dataset
    kwargs = {
        'dataset_name': args.dataset,
        'dataset_path': args.dataset_path,
        'limit': args.limit,
        'save_to_file': args.save_to_file,
        'output_dir': args.output_dir
    }
    
    if args.dataset in ['spider', 'bird']:
        kwargs['split'] = args.split
    elif args.dataset == 'spider2':
        kwargs['dataset_type'] = args.dataset_type
    
    instances = processor.process_dataset(**kwargs)
    
    print(f"Processed {len(instances)} instances from {args.dataset} dataset.")
    
    # Optionally save instances to file
    if args.save_to_file:
        output_file = f"{args.output_dir}/{args.dataset}_{args.split}_instances.json"
        with open(output_file, 'w') as f:
            for instance in instances:
                f.write(f"{instance.to_dict()}\n")
        print(f"Saved processed instances to {output_file}")


if __name__ == "__main__":
    main()