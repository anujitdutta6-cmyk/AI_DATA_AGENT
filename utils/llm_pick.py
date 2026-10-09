import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def pick_llm(level: str):
    """Select a configured Gemini-compatible model through the OpenAI API interface.

    Each level prefers its matching environment variable. Fallbacks preserve
    compatibility with older .env files that only define MEDIUM and HIGH.
    """
    level = level.lower().strip()
    if level not in {"low", "medium", "high"}:
        raise ValueError(f"Unsupported level: {level}. Supported levels: low, medium, high.")

    api_key = os.getenv("Gemini_Api_Key") or os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_URL")
    if not api_key:
        raise ValueError("Gemini_Api_Key (or GEMINI_API_KEY) is not configured.")
    if not base_url:
        raise ValueError("GEMINI_URL is not configured.")

    fallback_names = {
        "low": ("GEMINI_LOW_MODEL", "GEMINI_MEDIUM_MODEL", "GEMINI_HIGH_MODEL"),
        "medium": ("GEMINI_MEDIUM_MODEL", "GEMINI_HIGH_MODEL"),
        "high": ("GEMINI_HIGH_MODEL",),
    }
    model = next((os.getenv(name) for name in fallback_names[level] if os.getenv(name)), None)
    if not model:
        raise ValueError(
            f"No model is configured for level '{level}'. Set one of: "
            f"{', '.join(fallback_names[level])}."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


if __name__ == "__main__":
    llm_obj = pick_llm("high")
    response = llm_obj.invoke("What is the capital of France?")
    print(response.content)
