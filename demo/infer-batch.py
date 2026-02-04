import datetime
import os
import time
import torch
from vllm import LLM, SamplingParams
import re

PREFIX_CACHING = False
CONTINUOUS_BATCHING = True
SPECULATIVE_DECODING = False

def sanitize_filename(s: str, replacement: str = "_") -> str:
    # Keep letters, numbers, dash, underscore, dot
    return re.sub(r'[^A-Za-z0-9._-]', replacement, s)

def format_prompt(system_message, instruction, input_text, model_name):
    templates = {
        "Qwen/Qwen2.5-32B-Instruct-AWQ": """<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{instruction}

{input}
<|im_end|>
<|im_start|>assistant
""",
        "unsloth/gemma-3-27b-it-bnb-4bit": """<start_of_turn>user
{system}
{instruction}

{input}
<end_of_turn>
<start_of_turn>model
""",
    }
    if model_name in templates:
        template = templates[model_name]
        return template.format(system=system_message, instruction=instruction, input=input_text)
    else:
        # Default to old Mistral style
        return f"[INST] {system_message}\n\n{instruction}\n\n{input_text} [/INST]"


def main():
    # Configuration
    #model = "mistralai/Mistral-7B-Instruct-v0.3"
    #model = "ReBatch/Llama-3-8B-dutch"
    #model = "Qwen/Qwen2.5-32B-Instruct-AWQ" # 19GB disk, 45GB VRAM
    model = "unsloth/gemma-3-27b-it-bnb-4bit" # 16GB disk, 45GB VRAN
    input_dir = "/hfcache/input"
    output_dir = "/hfcache/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get GPU model
    gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    # Read prefix
    prefix_file = os.path.join(os.path.dirname(__file__), '..', 'prompt-prefix.txt')
    if not os.path.exists(prefix_file):
        print(f"Error: '{prefix_file}' not found.")
        return
    with open(prefix_file, "r", encoding="utf-8") as f:
        prefix = f.read().strip()
    
    # Get list of txt files
    if not os.path.exists(input_dir):
        print(f"Error: '{input_dir}' not found.")
        return
    
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if not txt_files:
        print(f"No .txt files found in '{input_dir}'.")
        return
    
    # 1. Initialize the Model
    start_init = time.time()
    
    # ------
    # PERFORMANCE IMPROVEMENT: PREFIX CACHING
    if PREFIX_CACHING:
        enable_prefix_caching = True
    else:
        enable_prefix_caching = False
    # ------
    # PERFORMANCE IMPROVEMENT: SPECULATIVE DECODING
    if SPECULATIVE_DECODING:
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
        }
    else:
        speculative_config = None
    # ------
    llm = LLM(model=model, download_dir="/hfcache/hub/", enable_prefix_caching=enable_prefix_caching, speculative_config=speculative_config)
    end_init = time.time()
    init_seconds = end_init - start_init
    
    warmup_duration = 0  # Initialize warmup duration
    # Warm-up inference
    warmup_prompt = "Warm-up request"
    warmup_sampling_params = SamplingParams(temperature=0, max_tokens=1)
    warmup_start = time.time()
    llm.generate([warmup_prompt], warmup_sampling_params)
    warmup_end = time.time()
    warmup_duration = warmup_end - warmup_start
    
    # For each input file
    all_prompts = []
    all_txt_files = []
    for txt_file in txt_files:
        full_path = os.path.join(input_dir, txt_file)
        with open(full_path, "r", encoding="utf-8") as f:
            user_input = f.read().strip()
        
        # 2. Format Prompt
        system_message = "Je bent een tekstredacteur."
        prompt_text = format_prompt(system_message, prefix, user_input, model)
        
        # Write prompt to file
        prompt_basename = os.path.splitext(txt_file)[0] + "_prompt.txt"
        prompt_path = os.path.join(output_dir, prompt_basename)
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        
        all_prompts.append(prompt_text)
        all_txt_files.append(txt_file)
    
    # 3. Run Inference
    print("Running inference...")
    sampling_params = SamplingParams(temperature=0, max_tokens=2048)
    start_infer = time.time()
    # ------
    # PERFORMANCE IMPROVEMENT: CONTINUOUS BATCHING
    if CONTINUOUS_BATCHING:
        outputs = llm.generate(all_prompts, sampling_params)
    else:
        outputs = []
        for prompt in all_prompts:
            output = llm.generate([prompt], sampling_params)
            outputs.append(output[0])
    # ------
    end_infer = time.time()
    total_infer_seconds = end_infer - start_infer
    print(f"Inference completed in {total_infer_seconds:.2f} seconds")
    
    log_file_path = os.path.join(output_dir, "inference_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for txt_file, output in zip(all_txt_files, outputs):
            # Print usage statistics
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            total_tokens = prompt_tokens + completion_tokens
            print(f"Token Usage for {txt_file}: Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            
            # vLLM returns only the generated text, so we prepend the prompt to match echo=True behavior
            full_text = output.outputs[0].text
            
            # 4. Write to file
            output_path = os.path.join(output_dir, txt_file)  # Use input filename for output
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
                
            print(f"Success! Output written to {output_path}")
        # Log summary
        log_file.write(f"{datetime.datetime.now()},{model},{gpu_model},{init_seconds:.2f},{warmup_duration:.2f},{total_infer_seconds:.2f},{total_prompt_tokens},{total_completion_tokens},PREFIX_CACHING={PREFIX_CACHING};CONTINUOUS_BATCHING={CONTINUOUS_BATCHING};SPECULATIVE_DECODING={SPECULATIVE_DECODING}\n")

if __name__ == "__main__":
    main()