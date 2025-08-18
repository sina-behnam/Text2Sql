
from dataset_readers import (
    SpiderDatasetReader,
    BIRDDatasetReader,
    Spider2DatasetReader
)
from instance import StandardizedInstance

from typing import List, Optional, Dict
from pathlib import Path
import json
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataProcessor:
    """Main processor class that orchestrates dataset reading and processing"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.readers = {
            'spider': SpiderDatasetReader,
            'bird': BIRDDatasetReader,
            'spider2': Spider2DatasetReader
        }
    
    def process_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        split: str = 'dev',
        limit: Optional[int] = None,
        save_to_file: bool = False,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> List[StandardizedInstance]:
        """
        Process a dataset and return standardized instances
        
        Args:
            dataset_name: Name of the dataset ('spider', 'bird', 'spider2')
            dataset_path: Path to the dataset directory
            split: Dataset split to process
            limit: Limit number of instances to process
            save_to_file: Whether to save processed instances to files
            output_dir: Directory to save processed instances
            **kwargs: Additional arguments for specific readers
            
        Returns:
            List of standardized instances
        """
        if dataset_name not in self.readers:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(self.readers.keys())}")
        
        # Create reader instance
        if dataset_name == 'spider2':
            reader = self.readers[dataset_name](
                dataset_path,
                dataset_type=kwargs.get('dataset_type', 'lite')
            )
        else:
            reader = self.readers[dataset_name](dataset_path, split)
        
        # Load instances
        instances = reader.load_instances(limit)
        
        # Save to files if requested
        if save_to_file:
            self._save_instances(instances, dataset_name, output_dir)
        
        return instances
    
    def _save_instances(
        self,
        instances: List[StandardizedInstance],
        dataset_name: str,
        output_dir: Optional[str] = None
    ):
        """Save processed instances to JSON files"""
        if not output_dir:
            output_dir = f"processed_{dataset_name}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each instance as a separate JSON file
        for instance in instances:
            filename = f"instance_{dataset_name}_{instance.id:04d}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(instance.to_dict(), f, indent=2)
        
        logger.info(f"Saved {len(instances)} instances to {output_path}")
        
        # Also save a summary file
        summary_file = output_path / f"{dataset_name}_summary.json"
        summary = {
            'dataset': dataset_name,
            'total_instances': len(instances),
            'difficulties': {
                'simple': sum(1 for i in instances if i.difficulty == 'simple'),
                'moderate': sum(1 for i in instances if i.difficulty == 'moderate'),
                'challenging': sum(1 for i in instances if i.difficulty == 'challenging')
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")

