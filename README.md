# Multi-Agent LLMs in Text2SQL Application

A comprehensive framework for evaluating and comparing different Large Language Models (LLMs) on Text2SQL tasks, supporting both API-based and local models with advanced evaluation metrics.

## 📁 Project Structure

```
Text2SQL-Multi-Agent-LLMs/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 src/
│   ├── 📄 __init__.py
│   ├── 📄 dataloader.py          # Dataset loading and processing
│   ├── 📄 models.py              # Model providers (API & Local)
│   ├── 📄 utils.py               # SQL processing and evaluation utilities
│   └── 📄 logger.py              # Centralized logging system
│
├── 📂 pipeline/
│   └── 📄 text2sql_enricher.py   # Main inference pipeline
│
├── 📂 tests/
│   ├── 📄 test_cuda.py           # CUDA availability testing
│   ├── 📄 test_spacy.py          # NLP processing testing
│   ├── 📄 test_transformers.py   # Transformers library testing
│   └── 📄 model_download.py      # Model downloading utilities
│
├── 📂 Data/                      # Dataset storage (gitignored)
│   ├── 📂 spider2-lite/
│   ├── 📂 bird/
│   └── 📄 instance_*.json        # Individual dataset instances
│
├── 📂 models/                    # Local model storage (gitignored)
│   ├── 📂 SmolLM3-3B/
│   ├── 📂 Qwen2.5-Coder-1.5B/
│   └── 📂 other-models/
│
├── 📂 notebooks/                 # Jupyter notebooks (gitignored)
├── 📂 logs/                      # Log files
└── 📂 venv/                      # Virtual environment (gitignored)
```

## 🚀 Features

### Multi-Model Support
- **API Models**: Together.ai, OpenAI, Anthropic (Claude)
- **Local Models**: HuggingFace Transformers with GPU/CPU support
- **Extended Thinking**: Chain-of-Thought reasoning for complex queries

### Advanced Evaluation Metrics
- **Execution Accuracy**: Verifies if generated SQL produces correct results
- **Exact Match**: String-based comparison after normalization
- **Semantic Equivalence**: LLM-based evaluation for logically equivalent queries.

#### Semantic Equivalence Explanation
Semantic equivalence is evaluated using LLM as a judge to determine if the predicted SQL query is logically equivalent to the ground truth. As an example if the ground truth SQL is:
```sql  
SELECT player_name, AVG(runs) AS batting_avg FROM IPL WHERE season = 5 GROUP BY player_name ORDER BY batting_avg DESC LIMIT 5;
```
and the predicted SQL is:
```sql
SELECT player_name, AVG(runs) AS batting_avg FROM IPL WHERE season = 5 GROUP
BY player_name ORDER BY batting_avg DESC LIMIT 5;
```
the LLM will determine that these two queries are semantically equivalent, even though they may differ in syntax or formatting in naming and terminology differences.

### Database Support
- **SQLite**: Local database files
- **Snowflake**: Cloud data warehouse (with connector)
- Extensible architecture for additional database types

## 📋 Prerequisites

### API Keys (Optional)
- **Together.ai**: For API-based model access
- **OpenAI**: For GPT models
- **Anthropic**: For Claude models
- **Snowflake**: For cloud database access

## 📊 Dataset Format

The project uses JSON files for individual instances:

```json
{
    "id": 409,
    "question": "Find the top 5 players with highest average runs per match in season 5",
    "sql": "SELECT p.player_name, AVG(runs) as batting_avg FROM ...",
    "database": {
        "name": "IPL",
        "path": ["databases/IPL.sqlite"],
        "type": "sqlite"
    },
    "schemas": [...],
    "difficulty": "challenging",
    "dataset": "spider2-lite"
}
```

## 🔧 Usage

### Basic Usage

```python
from src.dataloader import DatasetLoader
from pipeline.text2sql_enricher import Text2SQLInferencePipeline

# Load dataset
loader = DatasetLoader("./Data")
instances = loader.load_instances("instance_*.json")

# Configure model
model_config = {
    "type": "anthropic",  # or "together_ai", "openai", "local"
    "name": "claude-3-opus-20240229",
    "api_key": "your_api_key",
    "extended_thinking": True
}

# Run pipeline
pipeline = Text2SQLInferencePipeline(model_config)
results = pipeline.run_pipeline(instances)

print(f"Execution Accuracy: {results['execution_accuracy']:.2f}")
print(f"Semantic Equivalence: {results['semantic_equivalent_accuracy']:.2f}")
```

### Model Configuration Examples

**API Models:**
```python
# Together.ai
config = {"type": "together_ai", "name": "meta-llama/Llama-2-70b-chat-hf"}

# OpenAI
config = {"type": "openai", "name": "gpt-4"}

# Anthropic (with thinking)
config = {"type": "anthropic", "name": "claude-3-opus-20240229", "extended_thinking": True}
```

**Local Models:**
```python
config = {
    "type": "local",
    "path": "./models/Qwen2.5-Coder-1.5B",
    "max_new_tokens": 512,
    "extended_thinking": False
}
```

### Snowflake Database
```python
snowflake_config = {
    "user": "your_username",
    "password": "your_password",
    "account": "RSRSBDK-YDB67606"
}

pipeline = Text2SQLInferencePipeline(model_config, snowflake_config)
```

## 📈 Evaluation Metrics

The framework provides three levels of evaluation:

### 1. Execution Accuracy
$$\text{Execution Accuracy} = \frac{\text{Number of queries with correct results}}{\text{Total number of valid queries}}$$

### 2. Exact Match
$$\text{Exact Match} = \frac{\text{Number of syntactically identical queries}}{\text{Total number of predictions}}$$

### 3. Semantic Equivalence
$$\text{Semantic Equivalence} = \frac{\text{Number of logically equivalent queries}}{\text{Total number of predictions}}$$

## 🙏 Acknowledgments

- Built for Master's thesis research in "Multi-Agent LLMs in Text2SQL application"
- Supports Spider, Spider2-lite, and BIRD datasets
- Integrates with major LLM providers and local model frameworks

## 📞 Contact

For questions about this research project, please contact Sina Behnam at sina.behnam.ity@gmail.com

---

*This framework is part of ongoing research in Data Science and Engineering, focusing on the application of Multi-Agent Large Language Models to Text2SQL tasks.*