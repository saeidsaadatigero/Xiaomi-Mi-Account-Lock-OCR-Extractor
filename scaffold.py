"""scaffold.py — Windows-compatible scaffold script for unlock_code_extractor."""

import os
import sys


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    print(f"  📁 Created: {path}")


def write_file(file_path: str, content: str) -> None:
    """Write content to a file, creating parent dirs if needed."""
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  📄 Created: {file_path}")


def main() -> None:
    """Build the entire project scaffold."""
    print("=" * 60)
    print("  Building unlock_code_extractor scaffold...")
    print("=" * 60)
    print()

    # ── Create directories ──
    print("[1/4] Creating directories...")
    create_directory("services")
    create_directory("utils")
    create_directory("tests")
    print()

    # ── Create __init__.py files ──
    print("[2/4] Creating __init__.py files...")
    write_file("__init__.py", "")
    write_file("services/__init__.py", "")
    write_file("utils/__init__.py", "")
    write_file("tests/__init__.py", "")
    print()

    # ── Create project files ──
    print("[3/4] Creating project files...")

    write_file(".env.example", """OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PRIMARY_API_MODEL=google/gemma-4-26b-a4b-it:free
SECONDARY_API_MODEL=nvidia/nemotron-nano-12b-v2-vl:free
OCR_CONFIDENCE_THRESHOLD=0.70
EASYOCR_GPU=false
""")

    write_file("requirements.txt", """opencv-python-headless==4.9.0.80
Pillow==10.3.0
easyocr==1.7.1
openai==1.30.0
structlog==24.1.0
python-decouple==3.8
pytest==8.2.0
pytest-mock==3.14.0
""")

    write_file("exceptions.py", '''"""unlock_code_extractor/exceptions.py — Custom exceptions for the unlock code extractor."""


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
''')

    write_file("config.py", '''"""unlock_code_extractor/config.py — Application configuration via python-decouple."""

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
''')

    write_file("utils/logger.py", '''"""unlock_code_extractor/utils/logger.py — Structlog configuration for structured logging."""

import logging
import sys

import structlog


def setup_logging(verbose: bool = False) -> structlog.BoundLogger:
    """
    Configure and return a structlog logger instance.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.

    Returns:
        Configured structlog BoundLogger.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()
''')

    write_file("utils/image_preprocessor.py", '''"""unlock_code_extractor/utils/image_preprocessor.py — OpenCV image preprocessing pipeline."""

import os

import cv2
import numpy as np
from PIL import Image


def load_and_validate_image(file_path: str) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Load an image from disk and validate it is a readable JPG/PNG.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        Tuple of (image_array, file_size_kb, resolution).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid image.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    file_size_kb = os.path.getsize(file_path) / 1024.0

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported file extension '{ext}'. Allowed: {allowed_extensions}")

    pil_image = Image.open(file_path)
    pil_image.verify()

    pil_image = Image.open(file_path)
    image_array = np.array(pil_image)

    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    elif image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    height, width = image_array.shape[:2]
    resolution = (width, height)

    return image_array, file_size_kb, resolution


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, bool, tuple[int, int, int, int], tuple[int, int]]:
    """
    Detect phone screen, crop bottom 30%, enhance contrast, denoise, and upscale if needed.

    Args:
        image: Input BGR image as numpy array.

    Returns:
        Tuple of (preprocessed_crop, screen_detected, crop_region, final_resolution).
        crop_region is (x, y, w, h).
        final_resolution is (width, height).
    """
    screen_detected = False
    crop_region = (0, 0, image.shape[1], image.shape[0])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > (image.shape[0] * image.shape[1] * 0.05):
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                screen_detected = True
                crop_region = (x, y, w, h)

    img_height = image.shape[0]

    if screen_detected:
        _, y, _, h = crop_region
        bottom_start = y + int(h * 0.7)
        bottom_end = y + h
    else:
        bottom_start = int(img_height * 0.7)
        bottom_end = img_height

    bottom_start = max(0, bottom_start)
    bottom_end = min(img_height, bottom_end)

    cropped = image[bottom_start:bottom_end, :]

    if cropped.size == 0:
        cropped = image[int(img_height * 0.7):, :]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cropped_yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
    cropped_yuv[:, :, 0] = clahe.apply(cropped_yuv[:, :, 0])
    cropped = cv2.cvtColor(cropped_yuv, cv2.COLOR_YUV2BGR)

    cropped = cv2.fastNlMeansDenoising(cropped, h=10)

    crop_height = cropped.shape[0]
    if crop_height < 100:
        cropped = cv2.resize(
            cropped,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )

    final_height, final_width = cropped.shape[:2]
    final_resolution = (final_width, final_height)

    return cropped, screen_detected, crop_region, final_resolution
''')

    write_file("services/extractor_service.py", '''"""unlock_code_extractor/services/extractor_service.py — Core business logic for unlock code extraction."""

import base64
import re
import time
from io import BytesIO

import cv2
import easyocr
import numpy as np
from openai import OpenAI
from PIL import Image

from config import app_config
from exceptions import InvalidImageError, OCRServiceError, PatternNotFoundError
from utils.image_preprocessor import load_and_validate_image, preprocess_image
from utils.logger import setup_logging

logger = setup_logging()

UNLOCK_CODE_PATTERN = re.compile(r"\\b[A-Z0-9]{2,4}(-[A-Z0-9]{2,4}){3,4}\\b")

VISION_PROMPT = (
    "Look at this image carefully. Find and extract ONLY the unlock code. "
    "The unlock code is a sequence of 4 to 5 groups of uppercase letters and "
    "digits separated by hyphens, for example: 42MS-ETCO-R5MM-SWG "
    "The image may contain Persian or English text — ignore all of it. "
    "Return ONLY the raw code with no explanation, no label, no extra text. "
    "If you cannot find it, return exactly: NOT_FOUND"
)

_easyocr_reader: easyocr.Reader | None = None


def _get_easyocr_reader() -> easyocr.Reader:
    """Return a singleton EasyOCR Reader instance (lazy initialization)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        use_gpu = app_config.EASYOCR_GPU
        _easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
    return _easyocr_reader


def _encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a numpy BGR image to base64 JPEG string."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    b64_string = base64.b64encode(buffer.read()).decode("utf-8")
    return b64_string


def _call_vision_api(
    client: OpenAI,
    model_id: str,
    base64_image: str,
    use_reasoning: bool = False,
) -> tuple[str, int]:
    """
    Call OpenRouter Vision API with the given model and base64 image.

    Returns:
        Tuple of (response_text, http_status).
    """
    content = [
        {"type": "text", "text": VISION_PROMPT},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        },
    ]

    request_body: dict = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
    }

    if use_reasoning:
        request_body["reasoning"] = {"enabled": True}

    response = client.chat.completions.create(**request_body)

    response_text = response.choices[0].message.content.strip()
    return response_text, 200


def _extract_pattern_from_text(raw_text: str) -> tuple[list[str], str | None, int]:
    """
    Apply regex to extract unlock code candidates from raw text.

    Returns:
        Tuple of (candidates_list, selected_code_or_none, segment_count).
    """
    candidates: list[str] = []

    for match in UNLOCK_CODE_PATTERN.finditer(raw_text):
        candidates.append(match.group())

    candidates = list(set(candidates))

    valid_candidates = []
    for candidate in candidates:
        segments = candidate.split("-")
        if 4 <= len(segments) <= 5:
            valid_candidates.append(candidate)

    if not valid_candidates:
        return [], None, 0

    if len(valid_candidates) == 1:
        selected = valid_candidates[0]
    else:
        selected = max(valid_candidates, key=lambda c: len(c.split("-")))

    selected = selected.upper().strip()
    segment_count = len(selected.split("-"))

    return valid_candidates, selected, segment_count


def extract_unlock_code(
    image_path: str,
    force_model: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Main extraction pipeline: load image → preprocess → OCR cascade → pattern match.

    Args:
        image_path: Path to the input image file.
        force_model: Optional model override ('easyocr', 'gemma4', 'nemotron').
        verbose: Enable debug logging.

    Returns:
        Dict with keys: unlock_code, model, confidence, duration_ms.

    Raises:
        InvalidImageError: If the image file is invalid.
        PatternNotFoundError: If no code is found after all attempts.
    """
    pipeline_start = time.time()

    if verbose:
        global logger
        logger = setup_logging(verbose=True)

    log = logger.bind(module="extractor_service")

    try:
        image, file_size_kb, resolution = load_and_validate_image(image_path)
        log.info(
            "image_load",
            file_path=image_path,
            file_size_kb=round(file_size_kb, 2),
            resolution=f"{resolution[0]}x{resolution[1]}",
        )
    except FileNotFoundError as exc:
        log.error("image_load_failed", file_path=image_path, reason=str(exc))
        raise InvalidImageError(image_path, str(exc)) from exc
    except (ValueError, OSError, SyntaxError) as exc:
        log.error("image_load_failed", file_path=image_path, reason=str(exc))
        raise InvalidImageError(image_path, str(exc)) from exc

    try:
        preprocessed_crop, screen_detected, crop_region, final_resolution = preprocess_image(image)
        log.info(
            "preprocess",
            screen_detected=screen_detected,
            crop_region=str(crop_region),
            final_resolution=f"{final_resolution[0]}x{final_resolution[1]}",
        )
    except Exception as exc:
        log.error("preprocess_failed", reason=str(exc))
        raise InvalidImageError(image_path, f"Preprocessing failed: {exc}") from exc

    base64_crop = _encode_image_to_base64(preprocessed_crop)

    models_tried: list[str] = []
    last_raw_output: str = ""

    if force_model is None or force_model == "easyocr":
        models_tried.append("easyocr")
        easyocr_start = time.time()
        try:
            reader = _get_easyocr_reader()
            ocr_results = reader.readtext(preprocessed_crop)

            raw_text_parts: list[str] = []
            max_confidence = 0.0
            for (_, text, confidence_score) in ocr_results:
                raw_text_parts.append(text)
                if confidence_score > max_confidence:
                    max_confidence = confidence_score

            raw_text = " ".join(raw_text_parts)
            easyocr_duration_ms = round((time.time() - easyocr_start) * 1000, 2)

            log.info(
                "ocr_attempt",
                model_used="easyocr",
                raw_text=raw_text[:500],
                confidence=round(max_confidence, 4),
                duration_ms=easyocr_duration_ms,
            )

            candidates, selected_code, segment_count = _extract_pattern_from_text(raw_text)
            log.info(
                "pattern_match",
                candidates=candidates,
                selected_code=selected_code,
                segment_count=segment_count,
            )

            if selected_code is not None:
                total_duration_ms = round((time.time() - pipeline_start) * 1000, 2)
                log.info(
                    "result",
                    unlock_code=selected_code,
                    model_used="easyocr",
                    total_duration_ms=total_duration_ms,
                    success=True,
                )
                return {
                    "unlock_code": selected_code,
                    "model": "easyocr",
                    "confidence": round(max_confidence, 4),
                    "duration_ms": total_duration_ms,
                }

            log.warning(
                "easyocr_low_confidence_or_no_match",
                max_confidence=round(max_confidence, 4),
                threshold=app_config.OCR_CONFIDENCE_THRESHOLD,
            )

        except Exception as exc:
            log.error("easyocr_failed", reason=str(exc))
            easyocr_duration_ms = round((time.time() - easyocr_start) * 1000, 2)
            log.info(
                "ocr_attempt",
                model_used="easyocr",
                raw_text="",
                confidence=0.0,
                duration_ms=easyocr_duration_ms,
            )

    client = OpenAI(
        base_url=app_config.OPENROUTER_BASE_URL,
        api_key=app_config.OPENROUTER_API_KEY,
    )

    if force_model is None or force_model == "gemma4":
        models_tried.append("gemma4")
        gemma_start = time.time()
        try:
            response_text, http_status = _call_vision_api(
                client=client,
                model_id=app_config.PRIMARY_API_MODEL,
                base64_image=base64_crop,
                use_reasoning=False,
            )
            gemma_duration_ms = round((time.time() - gemma_start) * 1000, 2)
            last_raw_output = response_text

            log.info(
                "ocr_attempt",
                model_used="gemma4",
                raw_text=response_text[:500],
                confidence=None,
                duration_ms=gemma_duration_ms,
                http_status=http_status,
            )

            if response_text.strip() == "NOT_FOUND":
                log.warning("gemma4_returned_not_found")
            else:
                candidates, selected_code, segment_count = _extract_pattern_from_text(response_text)
                log.info(
                    "pattern_match",
                    candidates=candidates,
                    selected_code=selected_code,
                    segment_count=segment_count,
                )

                if selected_code is not None:
                    total_duration_ms = round((time.time() - pipeline_start) * 1000, 2)
                    log.info(
                        "result",
                        unlock_code=selected_code,
                        model_used="gemma4",
                        total_duration_ms=total_duration_ms,
                        success=True,
                    )
                    return {
                        "unlock_code": selected_code,
                        "model": "gemma4",
                        "confidence": None,
                        "duration_ms": total_duration_ms,
                    }

        except Exception as exc:
            log.error("gemma4_failed", reason=str(exc))
            gemma_duration_ms = round((time.time() - gemma_start) * 1000, 2)
            log.info(
                "ocr_attempt",
                model_used="gemma4",
                raw_text="",
                confidence=None,
                duration_ms=gemma_duration_ms,
                http_status=getattr(exc, "status_code", None),
            )

    if force_model is None or force_model == "nemotron":
        models_tried.append("nemotron")
        nemotron_start = time.time()
        try:
            response_text, http_status = _call_vision_api(
                client=client,
                model_id=app_config.SECONDARY_API_MODEL,
                base64_image=base64_crop,
                use_reasoning=True,
            )
            nemotron_duration_ms = round((time.time() - nemotron_start) * 1000, 2)
            last_raw_output = response_text

            log.info(
                "ocr_attempt",
                model_used="nemotron",
                raw_text=response_text[:500],
                confidence=None,
                duration_ms=nemotron_duration_ms,
                http_status=http_status,
            )

            if response_text.strip() == "NOT_FOUND":
                log.warning("nemotron_returned_not_found")
            else:
                candidates, selected_code, segment_count = _extract_pattern_from_text(response_text)
                log.info(
                    "pattern_match",
                    candidates=candidates,
                    selected_code=selected_code,
                    segment_count=segment_count,
                )

                if selected_code is not None:
                    total_duration_ms = round((time.time() - pipeline_start) * 1000, 2)
                    log.info(
                        "result",
                        unlock_code=selected_code,
                        model_used="nemotron",
                        total_duration_ms=total_duration_ms,
                        success=True,
                    )
                    return {
                        "unlock_code": selected_code,
                        "model": "nemotron",
                        "confidence": None,
                        "duration_ms": total_duration_ms,
                    }

        except Exception as exc:
            log.error("nemotron_failed", reason=str(exc))
            nemotron_duration_ms = round((time.time() - nemotron_start) * 1000, 2)
            log.info(
                "ocr_attempt",
                model_used="nemotron",
                raw_text="",
                confidence=None,
                duration_ms=nemotron_duration_ms,
                http_status=getattr(exc, "status_code", None),
            )

    total_duration_ms = round((time.time() - pipeline_start) * 1000, 2)
    log.error(
        "all_models_failed",
        models_tried=models_tried,
        total_duration_ms=total_duration_ms,
    )

    raise PatternNotFoundError(
        models_tried=models_tried,
        raw_output=last_raw_output,
    )
''')

    write_file("main.py", '''"""unlock_code_extractor/main.py — CLI entry point for unlock code extraction."""

import argparse
import json
import sys

from exceptions import InvalidImageError, PatternNotFoundError
from services.extractor_service import extract_unlock_code
from utils.logger import setup_logging


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract Xiaomi Mi Account unlock code from an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file (JPG/PNG).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["easyocr", "gemma4", "nemotron"],
        default=None,
        help="Force a specific OCR model (default: auto-cascade).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: parse args → extract → output."""
    args = parse_arguments()

    logger = setup_logging(verbose=args.verbose)
    log = logger.bind(module="main")

    try:
        result = extract_unlock_code(
            image_path=args.image,
            force_model=args.model,
            verbose=args.verbose,
        )
        print(json.dumps(result, indent=2))
        sys.exit(0)

    except InvalidImageError as exc:
        error_output = {
            "error": "InvalidImageError",
            "message": str(exc),
            "raw_ocr_output": "",
            "models_tried": [],
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)

    except PatternNotFoundError as exc:
        error_output = {
            "error": "PatternNotFoundError",
            "message": str(exc),
            "raw_ocr_output": exc.raw_output,
            "models_tried": exc.models_tried,
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(2)

    except Exception as exc:
        error_output = {
            "error": type(exc).__name__,
            "message": str(exc),
            "raw_ocr_output": "",
            "models_tried": [],
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
''')

    write_file("tests/test_extractor_service.py", '''"""unlock_code_extractor/tests/test_extractor_service.py — Unit tests for the extractor service."""

import pytest

from exceptions import InvalidImageError, PatternNotFoundError
from services.extractor_service import (
    _extract_pattern_from_text,
    _get_easyocr_reader,
    extract_unlock_code,
)


class TestExtractPatternFromText:
    """Tests for the _extract_pattern_from_text utility function."""

    def test_single_valid_code(self) -> None:
        raw = "Your unlock code is 42MS-ETCO-R5MM-SWG thank you"
        candidates, selected, count = _extract_pattern_from_text(raw)
        assert selected == "42MS-ETCO-R5MM-SWG"
        assert count == 4

    def test_five_segment_code(self) -> None:
        raw = "Code: 4YEV-QJCO-K6XE-FOO-BAR"
        candidates, selected, count = _extract_pattern_from_text(raw)
        assert selected == "4YEV-QJCO-K6XE-FOO-BAR"
        assert count == 5

    def test_no_pattern_found(self) -> None:
        raw = "This text has no unlock code at all"
        candidates, selected, count = _extract_pattern_from_text(raw)
        assert selected is None
        assert count == 0
        assert candidates == []

    def test_multiple_candidates_selects_most_segments(self) -> None:
        raw = "First: AB12-CD34-EF56-GH78 and Second: WXYZ-1234-ABCD-EFGH-IJKL"
        candidates, selected, count = _extract_pattern_from_text(raw)
        assert selected == "WXYZ-1234-ABCD-EFGH-IJKL"
        assert count == 5

    def test_lowercase_input_is_uppercased(self) -> None:
        raw = "code: 42ms-etco-r5mm-swg"
        candidates, selected, count = _extract_pattern_from_text(raw)
        assert selected == "42MS-ETCO-R5MM-SWG"

    def test_three_segments_rejected(self) -> None:
        raw = "AB12-CD34-EF56"
        candidates, selected, count = _extract_pattern_from_text(raw)
        assert selected is None

    def test_six_segments_rejected(self) -> None:
        raw = "AB12-CD34-EF56-GH78-IJ90-KL12"
        candidates, selected, count = _extract_pattern_from_text(raw)
        assert selected is None


class TestExtractUnlockCode:
    """Tests for the main extract_unlock_code pipeline function."""

    def _create_test_image(self, tmp_path: str, filename: str = "test_unlock.jpg") -> str:
        """Create a minimal valid JPEG image for testing."""
        from PIL import Image as PILImage

        image_path = f"{tmp_path}/{filename}"
        dummy_image = PILImage.new("RGB", (640, 480), color=(0, 0, 0))
        dummy_image.save(image_path, "JPEG")
        return image_path

    def test_invalid_file_path_raises_invalid_image_error(self) -> None:
        with pytest.raises(InvalidImageError):
            extract_unlock_code(image_path="/nonexistent/path/image.jpg")

    def test_invalid_file_extension_raises_invalid_image_error(self, tmp_path) -> None:
        text_file = f"{tmp_path}/test_file.txt"
        with open(text_file, "w") as f:
            f.write("not an image")
        with pytest.raises(InvalidImageError):
            extract_unlock_code(image_path=text_file)

    def test_extract_english_ui_returns_correct_code(self, mocker, tmp_path) -> None:
        image_path = self._create_test_image(tmp_path)

        mock_reader = mocker.MagicMock()
        mock_reader.readtext.return_value = [
            ((0, 0, 100, 30), "Unlock code:", 0.95),
            ((0, 40, 200, 70), "42MS-ETCO-R5MM-SWG", 0.94),
        ]
        mocker.patch(
            "services.extractor_service._get_easyocr_reader",
            return_value=mock_reader,
        )

        result = extract_unlock_code(image_path=image_path)
        assert result["unlock_code"] == "42MS-ETCO-R5MM-SWG"
        assert result["model"] == "easyocr"
        assert result["confidence"] == 0.95

    def test_extract_persian_ui_returns_correct_code(self, mocker, tmp_path) -> None:
        image_path = self._create_test_image(tmp_path)

        mock_reader = mocker.MagicMock()
        mock_reader.readtext.return_value = [
            ((0, 0, 100, 30), "Unlock code:", 0.92),
            ((0, 40, 200, 70), "4YEV-QJCO-K6XE-FOO", 0.91),
        ]
        mocker.patch(
            "services.extractor_service._get_easyocr_reader",
            return_value=mock_reader,
        )

        result = extract_unlock_code(image_path=image_path)
        assert result["unlock_code"] == "4YEV-QJCO-K6XE-FOO"
        assert result["model"] == "easyocr"

    def test_easyocr_low_confidence_triggers_gemma_fallback(self, mocker, tmp_path) -> None:
        image_path = self._create_test_image(tmp_path)

        mock_reader = mocker.MagicMock()
        mock_reader.readtext.return_value = [
            ((0, 0, 100, 30), "some blurry text", 0.45),
        ]
        mocker.patch(
            "services.extractor_service._get_easyocr_reader",
            return_value=mock_reader,
        )

        mock_openai_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "ABCD-1234-EFGH-5678"
        mock_openai_client.chat.completions.create.return_value = mock_response
        mocker.patch("services.extractor_service.OpenAI", return_value=mock_openai_client)

        result = extract_unlock_code(image_path=image_path)
        assert result["unlock_code"] == "ABCD-1234-EFGH-5678"
        assert result["model"] == "gemma4"

    def test_gemma_http_error_triggers_nemotron_fallback(self, mocker, tmp_path) -> None:
        image_path = self._create_test_image(tmp_path)

        mock_reader = mocker.MagicMock()
        mock_reader.readtext.return_value = [
            ((0, 0, 100, 30), "blurry", 0.30),
        ]
        mocker.patch(
            "services.extractor_service._get_easyocr_reader",
            return_value=mock_reader,
        )

        mock_openai_client = mocker.MagicMock()

        from openai import APIError

        mock_openai_client.chat.completions.create.side_effect = [
            APIError("Server error", response=mocker.MagicMock(status_code=500), body=None),
            mocker.MagicMock(
                choices=[mocker.MagicMock(message=mocker.MagicMock(content="WXYZ-9998-7776-5554"))]
            ),
        ]
        mocker.patch("services.extractor_service.OpenAI", return_value=mock_openai_client)

        result = extract_unlock_code(image_path=image_path)
        assert result["unlock_code"] == "WXYZ-9998-7776-5554"
        assert result["model"] == "nemotron"

    def test_all_models_fail_raises_pattern_not_found_error(self, mocker, tmp_path) -> None:
        image_path = self._create_test_image(tmp_path)

        mock_reader = mocker.MagicMock()
        mock_reader.readtext.return_value = [
            ((0, 0, 100, 30), "no code here", 0.20),
        ]
        mocker.patch(
            "services.extractor_service._get_easyocr_reader",
            return_value=mock_reader,
        )

        mock_openai_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "NOT_FOUND"
        mock_openai_client.chat.completions.create.return_value = mock_response
        mocker.patch("services.extractor_service.OpenAI", return_value=mock_openai_client)

        with pytest.raises(PatternNotFoundError) as exc_info:
            extract_unlock_code(image_path=image_path)

        assert "easyocr" in exc_info.value.models_tried
        assert "gemma4" in exc_info.value.models_tried
        assert "nemotron" in exc_info.value.models_tried

    def test_singleton_reader_not_reinitialized_on_second_call(self, mocker, tmp_path) -> None:
        image_path = self._create_test_image(tmp_path)

        mock_reader = mocker.MagicMock()
        mock_reader.readtext.return_value = [
            ((0, 0, 100, 30), "ABCD-1234-EFGH-5678", 0.99),
        ]

        mock_reader_class = mocker.patch(
            "services.extractor_service.easyocr.Reader",
            return_value=mock_reader,
        )

        extract_unlock_code(image_path=image_path)
        extract_unlock_code(image_path=image_path)

        assert mock_reader_class.call_count == 1

    def test_multiple_candidates_selects_most_segments(self, mocker, tmp_path) -> None:
        image_path = self._create_test_image(tmp_path)

        mock_reader = mocker.MagicMock()
        mock_reader.readtext.return_value = [
            ((0, 0, 100, 30), "AB12-CD34-EF56-GH78 and WXYZ-1234-ABCD-EFGH-IJKL", 0.90),
        ]
        mocker.patch(
            "services.extractor_service._get_easyocr_reader",
            return_value=mock_reader,
        )

        result = extract_unlock_code(image_path=image_path)
        assert result["unlock_code"] == "WXYZ-1234-ABCD-EFGH-IJKL"
''')

    print()

    # ── Summary ──
    print("[4/4] Scaffold complete!")
    print()
    print("=" * 60)
    print("  ✅ Project structure:")
    print("=" * 60)

    for root, dirs, files in os.walk("."):
        # Skip venv and __pycache__
        dirs[:] = [d for d in dirs if d not in ("venv", "__pycache__", ".git")]
        level = root.replace(".", "").count(os.sep)
        indent = "  " * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for file in sorted(files):
            if file.endswith((".py", ".txt", ".env", ".example")):
                print(f"{sub_indent}📄 {file}")

    print()
    print("=" * 60)
    print("  Next steps:")
    print("=" * 60)
    print("  1. pip install -r requirements.txt")
    print("  2. copy .env.example .env  (then edit with your API key)")
    print("  3. pytest tests/ -v")
    print("  4. python main.py --image path/to/photo.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()
