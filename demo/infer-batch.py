import datetime
import os
import time
import torch
from vllm import LLM, SamplingParams
import re

def sanitize_filename(s: str, replacement: str = "_") -> str:
    # Keep letters, numbers, dash, underscore, dot
    return re.sub(r'[^A-Za-z0-9._-]', replacement, s)

def format_prompt(messages):
    eos_token = "</s>"
    formatted_text = ""
    for i, message in enumerate(messages):
        if message["role"] == "user":
            if i % 2 != 0:
                raise Exception("Conversation roles must alternate user/assistant/user/assistant/...")
            formatted_text += "[INST] " + message["content"] + " [/INST]"
        elif message["role"] == "assistant":
            if i % 2 == 0:
                raise Exception("Conversation roles must alternate user/assistant/user/assistant/...")
            formatted_text += message["content"] + eos_token
        else:
            raise Exception("Only user and assistant roles are supported!")
    return formatted_text


def main():
    # Configuration
    model = "mistralai/Mistral-7B-Instruct-v0.3"
    #model = "ReBatch/Llama-3-8B-dutch"
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
    
    llm = LLM(model=model, download_dir="/hfcache/hub/")
    end_init = time.time()
    init_seconds = end_init - start_init
    
    # Warm-up inference
    warmup_prompt = "Warm-up request"
    warmup_sampling_params = SamplingParams(temperature=0, max_tokens=1)
    llm.generate([warmup_prompt], warmup_sampling_params)
    
    # For each input file
    for txt_file in txt_files:
        full_path = os.path.join(input_dir, txt_file)
        with open(full_path, "r", encoding="utf-8") as f:
            user_prompt = f.read().strip()
        user_prompt = prefix + "\n\n" + user_prompt
        
        # 2. Format Prompt (ChatML style)
        system_message = "Je bent een tekstredacteur."
        messages = [
            {"role": "user", "content": f"{system_message}\n\n{user_prompt}"}
        ]
        prompt_text = format_prompt(messages)
        
        # 3. Run Inference
        print(f"Running inference for {txt_file}...")
        sampling_params = SamplingParams(temperature=0, max_tokens=2048)
        start_infer = time.time()
        outputs = llm.generate([prompt_text], sampling_params)
        end_infer = time.time()
        infer_seconds = end_infer - start_infer
        
        # Print usage statistics
        output = outputs[0]
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens
        print(f"Token Usage: Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
        
        # vLLM returns only the generated text, so we prepend the prompt to match echo=True behavior
        full_text = prompt_text + output.outputs[0].text
        
        # 4. Write to file
        base_name = os.path.basename(txt_file).rsplit('.', 1)[0]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        output_filename = f"{base_name}-{timestamp}-{sanitize_filename(model)}-{sanitize_filename(gpu_model)}-{init_seconds:.2f}-{infer_seconds:.2f}-response.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {model}\n")
            f.write(full_text)
            
        print(f"Success! Output written to {output_path}")

if __name__ == "__main__":
    main()