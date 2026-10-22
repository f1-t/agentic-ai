"""Backend selection from environment variables."""
import os


def create_backend():
    backend = os.environ.get("LLM_BACKEND", "anthropic").strip().lower()
    if backend in {"anthropic", "claude"}:
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend()
    if backend in {"huggingface", "hf", "local"}:
        from .huggingface_backend import HuggingFaceBackend
        return HuggingFaceBackend()
    raise ValueError(
        f"Unknown LLM_BACKEND={backend!r}. Choose 'anthropic' or 'huggingface'."
    )
