import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_CONFIG = {
    "llama3.1:8b": {"backend": "ollama"},
    "qwen3:4b": {"backend": "ollama"},
    "gemma3:latest": {"backend": "ollama"},
    "meta-llama/Llama-3.1-8B-Instruct-Turbo": {"backend": "together"},
    "meta-llama/Llama-3.1-70B-Instruct-Turbo": {"backend": "together"},
}


def get_client(backend: str):
    if backend == "ollama":
        return OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # required by the library but ignored by Ollama
        )
    elif backend == "together":
        from together import Together
        return Together(api_key=os.getenv("TOGETHER_API_KEY"))
    else:
        raise ValueError(f"Unknown backend: {backend}")


def query_model(prompt: str, model: str, system_prompt: str = None) -> str:
    config = MODEL_CONFIG[model]
    client = get_client(config["backend"])

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content
