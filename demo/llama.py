from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path="./models/tinyllama-1.1/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",  # Download the model file first
  n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,           # The number of CPU threads to use, tailor to your system and the resulting performance
  chat_format="chatml"   # Set chat_format according to the model you are using (TinyLlama uses ChatML)
)

# Simple inference example
system_message = "You are a helpful assistant."
prompt = "Hello there!"
output = llm(
  f"<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>", # Prompt
  max_tokens=512,  # Generate up to 512 tokens
  stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
  echo=True        # Whether to echo the prompt
)
print(output['choices'][0]['text'])

# Chat Completion API

response = llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {
            "role": "user",
            "content": "What does the Summa Theologiae teach about the nature of Christ?"
        }
    ]
)
print(response['choices'][0]['message']['content'])
