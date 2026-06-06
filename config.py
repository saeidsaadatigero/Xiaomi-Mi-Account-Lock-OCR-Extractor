"""unlock_code_extractor/config.py — Application configuration via python-decouple."""

from decouple import config


class AppConfig:
    """Centralized application configuration loaded from environment variables."""

    OPENROUTER_API_KEY: str = config("OPENROUTER_API_KEY", default="")
    OPENROUTER_BASE_URL: str = config(
        "OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1"
    )
    PRIMARY_API_MODEL: str = config(
        "PRIMARY_API_MODEL", default="google/gemma-4-26b-a4b-it:free"
    )
    SECONDARY_API_MODEL: str = config(
        "SECONDARY_API_MODEL", default="nvidia/nemotron-nano-12b-v2-vl:free"
    )
    OCR_CONFIDENCE_THRESHOLD: float = config(
        "OCR_CONFIDENCE_THRESHOLD", default=0.70, cast=float
    )
    EASYOCR_GPU: bool = config("EASYOCR_GPU", default=False, cast=bool)


app_config = AppConfig()
