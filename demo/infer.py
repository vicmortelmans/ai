import datetime
import os
import time
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

def main():
    # Configuration
    # Assumes script is run from project root where ./models exists
    model_path = f"./models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    #draft_model_path = f"./models/Ministral-3b-instruct.Q8_0.gguf"
    model = os.path.basename(model_path)
    prompt_file = "prompt.txt"
    
    # 1. Read the prompt from file
    if not os.path.exists(prompt_file):
        print(f"Error: '{prompt_file}' not found in current working directory.")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        user_prompt = f.read().strip()

    # 2. Initialize the Model
    start_init = time.time()
    n_ctx = 4096
    
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=8,
        temperature=0,
        chat_format="mistral-instruct",  # NOTE: this is actually WRONG for my model!
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10),
        draft_k=10,
        logits_all=True,
    )
    end_init = time.time()
    init_seconds = end_init - start_init

    # 3. Format Prompt (ChatML style)
    system_message = "You are a text editor. You strictly preserve original wording and only correct spelling."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    # 4. Run Inference
    print("Running inference...")
    start_infer = time.time()
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=2048,
    )
    end_infer = time.time()
    infer_seconds = end_infer - start_infer
    
    # Print usage statistics
    usage = output['usage']
    print(f"Token Usage: Prompt: {usage['prompt_tokens']}, Completion: {usage['completion_tokens']}, Total: {usage['total_tokens']}")

    full_text = output['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": full_text})

    # 5. Write to timestamped file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_filename = f"infer-{timestamp}-{model}-{init_seconds:.2f}-{infer_seconds:.2f}-response.txt"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"# {model}\n")
        for msg in messages:
            f.write(f"\n<|{msg['role']}|>\n")
            f.write(f"{msg['content']}\n")
        
    print(f"Success! Output written to {output_filename}")

if __name__ == "__main__":
    main()