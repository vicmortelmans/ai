import argparse
import datetime
import os
import time
import re
import sys

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

def get_multiline_input():
    lines = []
    while True:
        try:
            line = input()
            if line == ".":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()

def main():
    # Configuration
    model = "unsloth/gemma-3-27b-it-bnb-4bit"
    
    parser = argparse.ArgumentParser(description="Live interactive inference runner")
    parser.add_argument("--prefix-caching", dest="prefix_caching", action="store_true", help="Enable prefix caching")
    parser.add_argument("--speculative-decoding", dest="speculative_decoding", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--prompt-file", dest="prompt_file", type=str, default=None, help="Reads base instruction from this txt file")
    parser.add_argument("--output-prefix", dest="output_prefix", type=str, default=".", help="Directory to save prompt/response files")

    args = parser.parse_args()
    os.makedirs(args.output_prefix, exist_ok=True)

    # Read base instruction if provided (the "prefix" from infer-batch.py)
    instruction = ""
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            instruction = f.read().strip()

    # 1. Initialize the Model
    print(f"Initializing model {model}...")
    from vllm import LLM, SamplingParams
    
    speculative_config = None
    if args.speculative_decoding:
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
        }

    llm = LLM(model=model, download_dir="/hfcache/hub/", 
              enable_prefix_caching=args.prefix_caching, 
              speculative_config=speculative_config)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    print("\n--- Live Inference Mode ---")
    print("Instructions: Enter your text. End with a single '.' on its own line or Ctrl-D to process.")
    print("Each input is processed as a fresh prompt (no conversation history).")
    print("Press Ctrl-C or enter an empty prompt (Ctrl-D immediately) to exit.")

    try:
        while True:
            print("\nInput > ", end="", flush=True)
            user_input = get_multiline_input()
            
            if not user_input:
                break

            system_message = "Je bent een tekstredacteur."
            prompt_text = format_prompt(system_message, instruction, user_input, model)
            
            print("\nGenerating...")
            outputs = llm.generate([prompt_text], sampling_params)
            response_text = outputs[0].outputs[0].text

            print("\n" + "="*20 + " RESPONSE " + "="*20)
            print(response_text)
            print("="*50)

            # Save files with shared timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            with open(os.path.join(args.output_prefix, f"prompt-{timestamp}.txt"), "w", encoding="utf-8") as f:
                f.write(prompt_text)
            with open(os.path.join(args.output_prefix, f"response-{timestamp}.txt"), "w", encoding="utf-8") as f:
                f.write(response_text)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()