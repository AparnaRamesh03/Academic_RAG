import os
from dotenv import load_dotenv
from pathlib import Path

# Load from brain/.env if not already loaded
_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    load_dotenv(dotenv_path=_env)

GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.1-8b-instant")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_use_openai = bool(os.getenv("OPENAI_API_KEY"))


def build_llm(temperature: float = 0.0, **kwargs):
    """Return ChatOpenAI if OPENAI_API_KEY is set, otherwise ChatGroq."""
    if _use_openai:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature, **kwargs)
    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, **kwargs)


def build_groq_llm(temperature: float = 0.0, **kwargs):
    """Legacy alias — now delegates to build_llm() so all nodes get OpenAI automatically."""
    return build_llm(temperature=temperature, **kwargs)
