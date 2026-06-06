"""unlock_code_extractor/utils/ocr_corrector.py — OCR output correction utilities."""

import re


# Common OCR mistakes for unlock codes - digit to letter mappings
DIGIT_TO_LETTER = {
    "0": "O",  # Zero → O
    "1": "I",  # One → I
    "4": "A",  # Four → A
    "5": "S",  # Five → S
    "6": "G",  # Six → G
    "8": "B",  # Eight → B
    "9": "g",  # Nine → g (rare)
}

# Letter to digit mappings (reverse)
LETTER_TO_DIGIT = {v: k for k, v in DIGIT_TO_LETTER.items() if v.upper() == v}

# Characters that should be letters in unlock codes (not digits)
# In unlock codes, segments are usually letters mixed with some digits
# But first character of each segment is usually a letter
COMMON_OCR_ERRORS = {
    "QU": "QJ",
    "QUC": "QJC",
    "0": "O",
    "1": "I",
}


def correct_ocr_text(raw_text: str) -> str:
    """
    Apply common OCR corrections to text.
    
    Args:
        raw_text: Raw OCR output text.
        
    Returns:
        Corrected text.
    """
    corrected = raw_text.upper()
    
    # Fix common OCR mistakes
    corrected = corrected.replace("QU", "QJ")
    corrected = corrected.replace("QUC", "QJC")
    
    return corrected


def fix_code_segments(code: str) -> str:
    """
    Fix common OCR mistakes in unlock code segments.
    Tries to convert digits that should be letters.
    
    Args:
        code: Unlock code like "4YE-QJC0-K6XE-F00"
        
    Returns:
        Fixed code like "AYE-QJCO-K6XE-FOO"
    """
    segments = code.split("-")
    fixed_segments = []
    
    for segment in segments:
        fixed = ""
        for i, char in enumerate(segment):
            # First character of segment is usually a letter
            if i == 0 and char in DIGIT_TO_LETTER:
                fixed += DIGIT_TO_LETTER[char]
            # Last characters in segment are often letters
            elif i == len(segment) - 1 and char in DIGIT_TO_LETTER:
                fixed += DIGIT_TO_LETTER[char]
            # Middle characters - check if surrounded by letters
            elif char in DIGIT_TO_LETTER:
                # Check neighbors
                prev_letter = i > 0 and segment[i-1].isalpha()
                next_letter = i < len(segment) - 1 and segment[i+1].isalpha()
                if prev_letter or next_letter:
                    fixed += DIGIT_TO_LETTER[char]
                else:
                    fixed += char
            else:
                fixed += char
        fixed_segments.append(fixed)
    
    return "-".join(fixed_segments)


def generate_code_variations(code: str) -> list[str]:
    """
    Generate possible variations of a code by replacing ambiguous characters.
    
    Args:
        code: Potential unlock code.
        
    Returns:
        List of possible code variations.
    """
    variations = [code]
    
    # Try fixing the code
    fixed = fix_code_segments(code)
    if fixed != code and fixed not in variations:
        variations.append(fixed)
    
    # Try replacing each digit with letter
    for i, char in enumerate(code):
        if char in DIGIT_TO_LETTER:
            new_code = code[:i] + DIGIT_TO_LETTER[char] + code[i+1:]
            if new_code not in variations:
                variations.append(new_code)
    
    return variations


def is_valid_code_format(code: str) -> bool:
    """
    Check if a string matches the unlock code format.
    
    Args:
        code: String to validate.
        
    Returns:
        True if valid format.
    """
    pattern = re.compile(r"^[A-Z0-9]{2,4}(-[A-Z0-9]{2,4}){3,4}$")
    return bool(pattern.match(code))


def find_code_in_text(raw_text: str) -> list[str]:
    """
    Find potential unlock codes in text, including variations.
    
    Args:
        raw_text: Raw text to search.
        
    Returns:
        List of potential codes found.
    """
    upper_text = raw_text.upper()
    candidates = []
    
    # First: try to find codes with hyphens
    pattern = re.compile(r"\b[A-Z0-9]{2,4}(-[A-Z0-9]{2,4}){3,4}\b")
    for match in pattern.finditer(upper_text):
        code = match.group()
        if code not in candidates:
            candidates.append(code)
            # Also add fixed version
            fixed = fix_code_segments(code)
            if fixed != code and fixed not in candidates:
                candidates.append(fixed)
    
    # Second: try to find space-separated codes
    if not candidates:
        groups = re.findall(r"\b([A-Z0-9]{2,4})\b", upper_text)
        
        for i in range(len(groups)):
            for length in [5, 4]:
                if i + length <= len(groups):
                    potential = "-".join(groups[i:i + length])
                    if is_valid_code_format(potential) and potential not in candidates:
                        candidates.append(potential)
                        fixed = fix_code_segments(potential)
                        if fixed != potential and fixed not in candidates:
                            candidates.append(fixed)
    
    # Third: try with corrected text
    if not candidates:
        corrected = correct_ocr_text(raw_text)
        for match in pattern.finditer(corrected):
            code = match.group()
            if code not in candidates:
                candidates.append(code)
                fixed = fix_code_segments(code)
                if fixed != code and fixed not in candidates:
                    candidates.append(fixed)
    
    return candidates
