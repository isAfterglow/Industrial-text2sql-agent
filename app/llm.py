from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """创建 OpenAI-compatible Chat Model。"""

    settings = get_settings()

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=0,
        max_retries=1,
        timeout=30,
    )