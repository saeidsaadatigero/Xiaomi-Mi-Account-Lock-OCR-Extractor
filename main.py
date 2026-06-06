"""unlock_code_extractor/main.py — CLI entry point for unlock code extraction."""

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
