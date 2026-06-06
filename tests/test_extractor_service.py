"""unlock_code_extractor/tests/test_extractor_service.py — Unit tests for the extractor service."""

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

        # Create a proper mock exception
        mock_error = Exception("API Error")
        mock_error.status_code = 500

        mock_openai_client.chat.completions.create.side_effect = [
            mock_error,
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
