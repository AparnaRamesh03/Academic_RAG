import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

# Shared Groq configuration for all LLM-backed nodes.
# Note: this is Groq provider, not xAI Grok.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Groq free tier: 30 RPM for llama-3.3-70b-versatile.
# The supervisor pipeline fires ~13 LLM calls per question, so we throttle
# to 0.4 req/s (~24 RPM) to stay safely under the limit.
# This limiter is a module-level singleton shared across all nodes.
_groq_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.4,
    check_every_n_seconds=0.05,
    max_bucket_size=10,
)


def build_groq_llm(temperature: float = 0.0) -> ChatGroq:
    """
    Create a Groq-backed chat model with the shared default model.
    Rate-limited to ~24 RPM to avoid Groq 429 errors during benchmarking.
    """
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        rate_limiter=_groq_rate_limiter,
    )