import datetime
import os
import sys
import time
import torch
from vllm import LLM, SamplingParams
import re

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

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
    #model = "mistralai/Mistral-7B-Instruct-v0.3"
    #model = "ReBatch/Llama-3-8B-dutch"
    model = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    input_dir = "/hfcache/input"
    output_dir = "/hfcache/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Redirect stdout and stderr to a file
    output_log = os.path.join(output_dir, 'script_output.txt')
    with open(output_log, 'w') as file_out:
        sys.stdout = Tee(sys.stdout, file_out)
        sys.stderr = Tee(sys.stderr, file_out)
        
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
        #llm = LLM(model=model, download_dir="/hfcache/hub/")
        llm = LLM(model=model, download_dir="/hfcache/hub/", enable_prefix_caching=True)
        # ------
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
            full_text = output.outputs[0].text
            
            # 4. Write to file
            output_path = os.path.join(output_dir, txt_file)  # Use input filename for output
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # Log metadata
            log_file_path = os.path.join(output_dir, "inference_log.txt")
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{datetime.datetime.now()},{model},{gpu_model},{init_seconds:.2f},{warmup_duration:.2f},{infer_seconds:.2f},{prompt_tokens},{completion_tokens}\n")
                
            print(f"Success! Output written to {output_path}")

if __name__ == "__main__":
    main()