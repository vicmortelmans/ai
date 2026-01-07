## Purpose
This repository is a minimal demo for running a local GGUF LLM with `llama_cpp`.
These instructions capture the project's runtime patterns, debugging tips, and conventions
so an AI coding agent can be immediately productive without guessing project intent.

## Big picture
- **Single small demo:** The main runnable is [demo/llama.py](demo/llama.py#L1-L40). It demonstrates loading a GGUF model and two usage styles: direct inference and chat completion.
- **Model files:** Models live under `models/` (example: [models/tinyllama-1.1/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf](models/tinyllama-1.1/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)). Model binaries are not tracked here and must be downloaded separately.

## Key files and patterns (examples)
- [demo/llama.py](demo/llama.py#L1-L40): Uses `from llama_cpp import Llama`. Look for `model_path` set to a relative path and a `pdb.set_trace()` that intentionally drops into the debugger.
- Model usage patterns:
  - Constructor options: `model_path`, `n_ctx`, `n_threads`, `gpu_layers`.
  - Two calling conventions: direct call `llm(...)` and `llm.create_chat_completion(...)` with `chat_format`.
  - Prompt formatting is manual in this demo; stop tokens and `echo` are set explicitly.

## Running & debugging
- Run from the repository root where `demo/` is the working folder, e.g.: `python demo/llama.py`.
- Important: `demo/llama.py` contains `import pdb; pdb.set_trace()` on startup — comment or remove that line if you want the script to run non-interactively.
- Recommended install (detects the imported module name): `pip install "llama-cpp-python"` (the package exposes `llama_cpp`).
- If GPU offload is desired, set `gpu_layers` in the `Llama(...)` constructor; the demo defaults to CPU (no explicit `gpu_layers` in the file).

## Agent-specific guidance (what an AI agent should/shouldn't change)
- Preserve relative `model_path` usage — agents should not hardcode absolute paths; use the `models/` folder.
- Do not remove the two usage examples (direct call vs chat API) — they document supported patterns.
- When modifying run flags (`n_threads`, `n_ctx`), keep changes small and document why they were changed (performance vs memory).
- If adding automation (scripts, tests), ensure they do not commit model binaries.

## Missing/absent conventions
- No CI, tests, or packaging files were found for agent workflows — add them only with explicit user approval.
- No existing agent instruction files were found in the repo root prior to this addition.

## Quick tasks an agent can do next
- Remove or guard the `pdb.set_trace()` behind a `if __name__ == "__main__":` block for non-interactive runs.
- Add a `requirements.txt` with `llama-cpp-python` and a short `README.md` usage snippet.

If any of these details are incomplete or you want additional automation (CI, Docker, or a starter `requirements.txt`), tell me which one to add and I'll update this file.
