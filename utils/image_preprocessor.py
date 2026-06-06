"""unlock_code_extractor/utils/image_preprocessor.py — OpenCV image preprocessing pipeline."""

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

    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    cropped = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Denoise
    cropped = cv2.fastNlMeansDenoising(cropped, h=10)

    # Convert to grayscale and apply adaptive threshold for better OCR
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(gray_crop, cv2.MORPH_CLOSE, kernel)

    # Adaptive threshold to handle variable lighting
    thresh = cv2.adaptiveThreshold(
        morph, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # Convert back to BGR for consistency
    cropped = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Upscale if too small
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
