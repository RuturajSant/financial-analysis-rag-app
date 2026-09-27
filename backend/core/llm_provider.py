"""
LLM provider factory — returns the right LangChain chat model for the configured provider.

Usage:
    from backend.core.llm_provider import get_chat_llm
    llm = get_chat_llm(temperature=0, max_tokens=5)

Supports: openai, anthropic, google.
"""
from __future__ import annotations

from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_PROVIDERS = {"openai", "anthropic", "google"}


def extract_text(content) -> str:
    """Normalise LLM response content to a plain string.

    Google GenAI models may return content as a list of typed blocks, e.g.
    [{'type': 'text', 'text': '...', ...}].  This helper extracts and
    concatenates only the 'text' blocks.  For providers that return a plain
    string, it passes through unchanged.
    """
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return content


def get_chat_llm(**overrides):
    """
    Return a LangChain chat model for the configured provider.

    Args:
        **overrides: Keyword arguments forwarded to the chat model constructor
                     (e.g. temperature, max_tokens). Provider-specific keys are
                     mapped automatically.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the configured provider is not supported.
        ImportError: If the required provider package is not installed.
    """
    provider = settings.llm_provider.lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            f"Choose from: {', '.join(sorted(_SUPPORTED_PROVIDERS))}"
        )

    model = overrides.pop("model", settings.chat_model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url="http://localhost:20128/v1",
            model=model,
            openai_api_key=settings.openai_api_key,
            **overrides,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=model,
            anthropic_api_url="http://localhost:20128/v1",
            anthropic_api_key=settings.anthropic_api_key,
            **overrides,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            **overrides,
        )

    logger.info(
        "llm_created",
        extra={"provider": provider, "model": model},
    )
    return llm
