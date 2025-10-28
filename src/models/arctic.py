from vllm import LLM, SamplingParams
import re
from pydantic import Field
from src.models.models import ModelProvider, ModelConfig
from typing import Optional

class ArcticConfig(ModelConfig):
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory utilization for vLLM")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallel size for vLLM")
    dtype: str = Field(default="float16", description="Data type for model weights")
    max_model_len: int = Field(default=8192, description="Maximum model context length")
    stop_sequences: list[str] = Field(default_factory=lambda: ["</answer>"])


class ArcticText2SQLInference(ModelProvider):
    """Arctic Text2SQL model inference using vLLM."""
    
    def __init__(self, model_name: str = "Snowflake/Arctic-Text2SQL-R1-7B", 
                 config: Optional[ArcticConfig] = None, **config_kwargs):
        super().__init__(model_name, config=config, **config_kwargs)
        '''
         Initialize the Arctic Text2SQL model for inference using vLLM.
         Args:
             model_name: Name or path of the pre-trained model (default: "Snowflake/Arctic-Text2SQL-R1-7B")
             config: Optional ArcticConfig instance
             config_kwargs: Additional keyword arguments for configuration
        '''
        
        if config is not None:
            self.config = config
        elif config_kwargs:
            self.config = ArcticConfig(**config_kwargs)
        else:
            self.config = ArcticConfig()
        

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=True,
            dtype=self.config.dtype,
            max_model_len=self.config.max_model_len,
        )
        
        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            seed=self.config.seed,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
            stop=self.config.stop_sequences
        )
    
    def reformat_prompt(self, system_message, user_message, assistant_prefix):
        """Reformat prompt into vLLM chat format"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_prefix}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return prompt

    def generate(self,system_message, user_message, assistant_prefix):

        prompt = self.reformat_prompt(system_message, user_message, assistant_prefix)
        
        outputs = self.llm.generate([prompt], self.sampling_params)

        generated_text = outputs[0].outputs[0].text

        full_response = assistant_prefix + generated_text

        return full_response

    def generate_sql_batch(self, questions, schemas, evidence_list=None):
        """Generate SQL for multiple questions in batch (much faster!)"""
        pass
    

