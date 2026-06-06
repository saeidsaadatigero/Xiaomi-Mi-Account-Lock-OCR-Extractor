"""unlock_code_extractor/services/extractor_service.py — Core business logic for unlock code extraction."""

import base64
import re
import time
from io import BytesIO

import cv2
import easyocr
import numpy as np
from openai import OpenAI, APIError
from PIL import Image

from config import app_config
from exceptions import InvalidImageError, OCRServiceError, PatternNotFoundError
from utils.image_preprocessor import load_and_validate_image, preprocess_image
from utils.logger import setup_logging

logger = setup_logging()

UNLOCK_CODE_PATTERN = re.compile(r"(?:^|[^A-Z0-9])([A-Z0-9]{2,4}(?:-[A-Z0-9]{2,4}){3,4})(?:[^A-Z0-9]|$)")

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
    Converts text to uppercase before matching.

    Returns:
        Tuple of (candidates_list, selected_code_or_none, segment_count).
    """
    upper_text = raw_text.upper()

    # Split by whitespace and common delimiters to get tokens
    import re as re_module
    tokens = re_module.split(r'[\s,;]+', upper_text)

    candidates: list[str] = []

    for token in tokens:
        # Check if token matches exactly 4-5 segments
        segment_count = len(token.split("-"))
        if 4 <= segment_count <= 5:
            # Verify each segment is 2-4 chars of uppercase letters/digits
            segments = token.split("-")
            all_valid = all(
                re_module.fullmatch(r"[A-Z0-9]{2,4}", seg)
                for seg in segments
            )
            if all_valid and token not in candidates:
                candidates.append(token)

    if not candidates:
        return [], None, 0

    if len(candidates) == 1:
        selected = candidates[0]
    else:
        selected = max(candidates, key=lambda c: len(c.split("-")))

    selected = selected.strip()
    segment_count = len(selected.split("-"))

    return candidates, selected, segment_count



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
