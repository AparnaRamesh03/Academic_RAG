"""
brain/llm_config.py
--------------------
Provider-aware LLM factory for all final_arch nodes and the MADDPG adapter.

Provider is selected via env var LLM_PROVIDER, with sensible per-provider
model defaults that can be individually overridden:

  LLM_PROVIDER=groq      (default)   model env: GROQ_MODEL
  LLM_PROVIDER=openai                model env: OPENAI_MODEL  (default: gpt-4o-mini)

`build_groq_llm()` is preserved as a backward-compatible alias for the modules
that still import it by name; it now dispatches through `build_llm()`.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


def _provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "groq").strip().lower()


# Default model per provider; users can override via the provider-specific env.
_DEFAULT_MODELS = {
    "groq":   "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}


def current_model() -> str:
    p = _provider()
    if p == "openai":
        return os.getenv("OPENAI_MODEL", _DEFAULT_MODELS["openai"])
    return os.getenv("GROQ_MODEL", _DEFAULT_MODELS["groq"])


# Kept for compatibility with existing imports across final_arch nodes.
GROQ_MODEL = current_model()


def build_llm(temperature: float = 0.0, max_tokens: Optional[int] = None) -> Any:
    """
    Build a chat model for the active provider.

    Returns whatever LangChain chat object matches the provider; both ChatGroq
    and ChatOpenAI support `.invoke([HumanMessage(...)])` so callers do not
    need to special-case the type.
    """
    p = _provider()
    kwargs: dict[str, Any] = {"model": current_model(), "temperature": float(temperature)}
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**kwargs)

    if p == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(**kwargs)

    raise ValueError(
        f"Unknown LLM_PROVIDER='{p}'. Supported: groq, openai. "
        f"Set the LLM_PROVIDER env var in brain/.env."
    )


def build_groq_llm(temperature: float = 0.0) -> Any:
    """
    Backward-compatible alias used by brain/final_arch nodes.

    Despite the name, this now returns whichever provider LLM_PROVIDER selects.
    Kept under the old name so we do not have to edit every node file.
    """
    return build_llm(temperature=temperature)
