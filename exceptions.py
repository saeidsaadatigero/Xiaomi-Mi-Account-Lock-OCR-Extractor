"""unlock_code_extractor/exceptions.py — Custom exceptions for the unlock code extractor."""


class InvalidImageError(Exception):
    """Raised when the input image file is invalid, missing, or unreadable."""

    def __init__(self, file_path: str, reason: str = "Unknown error") -> None:
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Invalid image at '{file_path}': {reason}")


class PatternNotFoundError(Exception):
    """Raised when no unlock code pattern is found after all OCR attempts."""

    def __init__(self, models_tried: list[str], raw_output: str = "") -> None:
        self.models_tried = models_tried
        self.raw_output = raw_output
        super().__init__(
            f"No unlock code pattern found. Models tried: {models_tried}. "
            f"Raw output: {raw_output[:200]}"
        )


class OCRServiceError(Exception):
    """Raised when an OCR service (EasyOCR or Vision API) fails unexpectedly."""

    def __init__(self, model_name: str, reason: str, http_status: int | None = None) -> None:
        self.model_name = model_name
        self.reason = reason
        self.http_status = http_status
        status_info = f" (HTTP {http_status})" if http_status else ""
        super().__init__(f"OCR service '{model_name}' failed{status_info}: {reason}")
