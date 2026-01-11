import datetime
import os
import time
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
    prompt_file = os.path.join(os.path.dirname(__file__), '..', 'prompt.txt')
    
    # 1. Read the prompt from file
    if not os.path.exists(prompt_file):
        print(f"Error: '{prompt_file}' not found in current working directory.")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        user_prompt = f.read().strip()

    # 2. Initialize the Model
    start_init = time.time()
    
    llm = LLM(model=model, download_dir="/hfcache/hub/")
    end_init = time.time()
    init_seconds = end_init - start_init

    # Warm-up inference
    warmup_prompt = "Warm-up request"
    warmup_sampling_params = SamplingParams(temperature=0, max_tokens=1)
    llm.generate([warmup_prompt], warmup_sampling_params)

    # 3. Format Prompt (ChatML style)
    system_message = "You are a text editor. You strictly preserve original wording and only correct spelling."
    messages = [
        {"role": "user", "content": f"{system_message}\n\n{user_prompt}"}
    ]
    prompt_text = format_prompt(messages)

    # 4. Run Inference
    print("Running inference...")
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

    # 5. Write to timestamped file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_filename = f"infer-raw-{timestamp}-{sanitize_filename(model)}-{init_seconds:.2f}-{infer_seconds:.2f}-response.txt"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"# {model}\n")
        f.write(full_text)
        
    print(f"Success! Output written to {output_filename}")

if __name__ == "__main__":
    main()