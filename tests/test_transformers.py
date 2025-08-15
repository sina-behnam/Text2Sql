from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch
import os
from transformers import TextStreamer

model_name = "HuggingFaceTB/SmolLM3-3B"
local_dir = "./models/SmolLM3-3B"
# Step 2: Load from local directory
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Step 3: Inference (same as before)
def generate_response(prompt, max_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# Test
# Test with streaming output
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Create input tensor
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Create a streamer
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

# Generate with streaming
print("Response:")
_ = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
    streamer=streamer
)