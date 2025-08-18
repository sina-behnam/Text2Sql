This is an unified implementation that provide all three dataset readers (Spider, BIRD, and Spider2) into an organized architecture and transform the original dataset into standard unified structure where the rest of implmenetation will work with output of this dataset preparation pipeline. Here are the key features:

## Architecture Overview

### 1. **`StandardizedInstance` Dataclass** (located in `instance.py`)
- A unified data structure for all datasets
- Contains all required fields with optional fields for dataset-specific data
- Easily convertible to dictionary for JSON serialization

### 2. **`BaseDatasetReader` Abstract Class** (located in `base_reader.py`)
- Defines the interface that all dataset readers must implement
- Contains common functionality like DDL extraction and difficulty calculation
- Ensures consistency across different dataset implementations

### 3. **Concrete Reader Implementations** (located in `dataset_readers.py`)
- **`SpiderDatasetReader`**: Handles Spider dataset with its specific structure
- **`BIRDDatasetReader`**: Processes BIRD dataset with evidence and database descriptions
- **`Spider2DatasetReader`**: Manages Spider2-lite with multiple database types (SQLite, Snowflake, BigQuery)

### 4. **`DataProcessor` Class** (located in `processors.py`)
- Main orchestrator that manages all readers
- Provides a unified interface for processing any dataset
- Handles saving processed instances to files

### 5. **Command-Line Interface** (located in `main.py`)
- Main entry point for the script


## Key Features

1. **Standardized Output Format**: All datasets are converted to the same structure:
   ```python
   {
       "id": int,
       "dataset": str,
       "question": str,
       "sql": str,
       "database": {...},
       "schemas": [...],
       "difficulty": str,
       "original_instance_id": str (optional),
       "evidence": str (optional)
   }
   ```

## Usage Example

```python
# Command line usage examples:

# Process Spider dataset (development split, first 100 instances)
python main.py --dataset spider --dataset_path /path/to/spider/dataset --split dev --limit 100 --save_to_file --output_dir processed_data/spider

# Process BIRD dataset (development split, all instances)
python main.py --dataset bird --dataset_path /path/to/bird/dataset --split dev --save_to_file --output_dir processed_data/bird

# Process Spider2-lite dataset (all instances)
python main.py --dataset spider2 --dataset_path /path/to/spider2/dataset --dataset_type lite --save_to_file --output_dir processed_data/spider2

# Programmatic usage:
from unified_dataset_processor import DataProcessor

processor = DataProcessor()

# Process datasets programmatically
spider_instances = processor.process_dataset(
    dataset_name='spider',
    dataset_path='/path/to/spider/dataset',
    split='dev',
    limit=100
)
```
