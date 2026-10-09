import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def pick_llm(level: str):
    """
    Select the Gemini model based on the requested capability level.

    Supported levels:
        low    -> GEMINI_MEDIUM_MODEL
        medium -> GEMINI_HIGH_MODEL
        high   -> GEMINI_MEDIUM_MODEL
    """

    level = level.lower().strip()

    api_key = os.getenv("Gemini_Api_Key")
    base_url = os.getenv("GEMINI_URL")

    if not api_key:
        raise ValueError(
            "Gemini_Api_Key is not configured in the .env file."
        )

    if not base_url:
        raise ValueError(
            "GEMINI_URL is not configured in the .env file."
        )

    if level == "low":
        model = os.getenv("GEMINI_MEDIUM_MODEL")

    elif level == "medium":
        model = os.getenv("GEMINI_HIGH_MODEL")

    elif level == "high":
        model = os.getenv("GEMINI_MEDIUM_MODEL")

    else:
        raise ValueError(
            f"Unsupported level: {level}. "
            "Supported levels: low, medium, high."
        )

    if not model:
        raise ValueError(
            f"Model is not configured for level '{level}'."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    llm_obj = pick_llm("high")

    response = llm_obj.invoke(
        "What is the capital of France?"
    )

    print(response.content)