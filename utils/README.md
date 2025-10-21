# Text2SQL Prompt Engineering Utilities

This directory contains utilities for prompt engineering in the Text2SQL pipeline, featuring a flexible, extensible system for creating model-specific prompts and extracting SQL queries from model responses.

## Architecture

The system uses **Object-Oriented Programming (OOP)** with the following design patterns:

- **Base Class Pattern**: `BasePromptTemplate` defines the interface that all templates must implement
- **Template Pattern**: Subclasses implement specific prompt strategies for different models
- **Registry Pattern**: `PromptTemplateRegistry` manages model-to-template mappings
- **Factory Pattern**: `get_prompt_template()` returns the appropriate template for a given model

## Files

- **`prompt_engineering.py`**: Core module with base class and built-in templates
- **`custom_prompt_example.py`**: Examples of creating custom templates for fine-tuned models
- **`README.md`**: This file

## Quick Start

### Basic Usage

```python
from utils.prompt_engineering import get_prompt_template

# Get a template for your model
template = get_prompt_template("claude-3-5-sonnet-20241022")

# Create prompts
system_msg, user_msg = template.create_prompt(
    question="Get all active users",
    schema="CREATE TABLE users (id INT, name TEXT, status TEXT);"
)

# Extract SQL from model response
model_response = '{"sql": "SELECT * FROM users WHERE status = \'active\'"}'
sql = template.extract_sql(model_response)
print(sql)  # SELECT * FROM users WHERE status = 'active'
```

### With Few-Shot Examples

```python
examples = [
    {
        "question": "Get all customers",
        "schema": "CREATE TABLE customers (id INT, name TEXT);",
        "sql": "SELECT * FROM customers"
    }
]

system_msg, user_msg = template.create_prompt(
    question="Get premium customers",
    schema="CREATE TABLE customers (id INT, name TEXT, tier TEXT);",
    few_shot_examples=examples
)
```

### With Evidence/Context

```python
system_msg, user_msg = template.create_prompt(
    question="Get recent orders",
    schema="CREATE TABLE orders (id INT, date TEXT);",
    evidence="Recent means within the last 30 days"
)
```

## Built-in Templates

The following models have pre-configured templates:

### Anthropic Models
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-sonnet-20240620`
- `claude-3-5-haiku-20241022`

### OpenAI Models
- `gpt-4`
- `gpt-4-turbo`
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-3.5-turbo`

### Model Types (Fallback)
- `anthropic`: Uses `AnthropicPromptTemplate`
- `openai`: Uses `OpenAIPromptTemplate`
- `together_ai`: Uses `TogetherAIPromptTemplate`
- `local`: Uses `LocalModelPromptTemplate`
- `default`: Uses `DefaultPromptTemplate`

## Creating Custom Templates

For fine-tuned models with specific prompt requirements, extend `BasePromptTemplate`:

### Step 1: Create Your Template Class

```python
from utils.prompt_engineering import BasePromptTemplate

class MyFineTunedTemplate(BasePromptTemplate):
    """Custom template for my fine-tuned model."""

    def __init__(self, model_name="my-model"):
        super().__init__(model_name, "custom")

    def create_prompt(self, question, schema, evidence=None, few_shot_examples=None):
        """Create a prompt in your model's expected format."""

        # Your custom system message
        system_message = "Your custom system prompt here..."

        # Your custom user message format
        user_message = f"Custom format: {question} | Schema: {schema}"

        return system_message, user_message

    def extract_sql(self, response_text, clean=True):
        """Extract SQL from your model's response format."""
        import re

        # Your custom extraction logic
        match = re.search(r'YOUR_PATTERN: (SELECT.*)', response_text, re.IGNORECASE)
        if match:
            sql = match.group(1)
            return self._clean_sql(sql) if clean else sql

        # Fallback to generic extraction
        return self._extract_sql_generic(response_text, clean)
```

### Step 2: Register Your Template

```python
from utils.prompt_engineering import register_prompt_template

register_prompt_template("my-finetuned-model-v1", MyFineTunedTemplate)
```

### Step 3: Use It

```python
from utils.prompt_engineering import get_prompt_template

# Now your template is automatically available
template = get_prompt_template("my-finetuned-model-v1")
system_msg, user_msg = template.create_prompt(question, schema)
```

### Step 4: Use in Pipeline

```python
from pipeline.text2sql_enricher import Text2SQLInferencePipeline

# Register your template first
register_prompt_template("my-model", MyFineTunedTemplate)

# Configure pipeline with your model name
model_config = {
    "type": "local",
    "name": "my-model",  # This will automatically use your custom template!
    "path": "/path/to/model",
    "device": "cuda"
}

# Initialize pipeline - it automatically uses the registered template
pipeline = Text2SQLInferencePipeline(model_config=model_config)
```

## Advanced: Custom Extraction Patterns

The base class provides helper methods for common extraction patterns:

```python
class MyTemplate(BasePromptTemplate):
    def extract_sql(self, response_text, clean=True):
        # Remove thinking tags
        text = self._remove_thinking_tags(response_text)

        # Try different extraction methods
        sql = self._try_json_extraction(text)
        if sql and self._is_valid_sql(sql):
            return self._clean_sql(sql) if clean else sql

        sql = self._try_code_block_extraction(text)
        if sql and self._is_valid_sql(sql):
            return self._clean_sql(sql) if clean else sql

        # Add your custom patterns...

        return ""
```

### Available Helper Methods

- `_remove_thinking_tags(text)`: Remove `<think>` and `<thinking>` tags
- `_try_json_extraction(text)`: Extract SQL from JSON format
- `_try_code_block_extraction(text)`: Extract from markdown code blocks
- `_try_xml_extraction(text)`: Extract from XML tags
- `_try_intro_phrase_extraction(text)`: Extract after phrases like "The SQL is:"
- `_try_direct_sql_extraction(text)`: Direct pattern matching for SQL keywords
- `_is_valid_sql(text)`: Check if text is valid SQL
- `_clean_sql(sql)`: Remove comments, extra whitespace, semicolons
- `_build_few_shot_examples(examples)`: Format few-shot examples

## Examples

See `custom_prompt_example.py` for complete examples including:

1. **CodeLlama Fine-tuned Model**: Custom format with `[INST]` tags
2. **XML Format Model**: Model that outputs XML instead of JSON
3. **Domain-Specific Model**: Healthcare SQL with specialized guidelines

## Testing

Run the test suite:

```bash
python test_prompt_engineering.py
```

This tests:
- Template retrieval for different models
- Prompt creation
- SQL extraction from various formats
- Few-shot examples
- Custom template registration
- Registry listing
- Evidence integration

## Integration with Text2SQL Pipeline

The pipeline automatically uses the appropriate template based on your model configuration:

```python
from pipeline.text2sql_enricher import Text2SQLInferencePipeline

# The pipeline automatically selects the right template
pipeline = Text2SQLInferencePipeline(
    model_config={
        "type": "anthropic",
        "name": "claude-3-5-sonnet-20241022",  # Uses AnthropicPromptTemplate
        "api_key": "your-key"
    },
    few_shot_examples=[...]  # Optional
)

# Process instances - prompts are automatically created using the template
results = pipeline.run_pipeline(instances)
```

## Design Principles

1. **Extensibility**: Easily add new templates without modifying existing code
2. **Type Safety**: Abstract base class ensures all templates implement required methods
3. **Flexibility**: Support for model-specific prompts and extraction patterns
4. **Reusability**: Common extraction patterns available as helper methods
5. **Maintainability**: Centralized prompt management
6. **Fallback**: Generic extraction works even if custom patterns fail

## API Reference

### BasePromptTemplate

**Abstract Methods:**
- `create_prompt(question, schema, evidence=None, few_shot_examples=None) -> Tuple[str, str]`
- `extract_sql(response_text, clean=True) -> str`

**Helper Methods:**
- `_build_few_shot_examples(examples) -> str`
- `_is_valid_sql(text) -> bool` (static)
- `_clean_sql(sql) -> str` (static)
- `_extract_sql_generic(text, clean) -> str`
- Various extraction helpers (see above)

### Functions

- **`get_prompt_template(model_name, model_type=None) -> BasePromptTemplate`**
  - Get template instance for a model

- **`register_prompt_template(model_name, template_class)`**
  - Register a custom template

- **`PromptTemplateRegistry.list_registered() -> List[str]`**
  - List all registered model names

## Contributing

To add support for a new model:

1. Determine if existing templates work for your model
2. If not, create a custom template extending `BasePromptTemplate`
3. Implement `create_prompt()` and `extract_sql()` methods
4. Register your template using `register_prompt_template()`
5. Add tests to verify functionality
6. Update this README with your model

## License

Part of the Text2SQL project.
