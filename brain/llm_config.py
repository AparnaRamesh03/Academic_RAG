import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Shared Groq configuration for all LLM-backed nodes.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def build_groq_llm(temperature: float = 0.0) -> ChatGroq:
    """
    Create a Groq-backed chat model with the shared default model.
    """
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )