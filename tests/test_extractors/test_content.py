"""Tests for content extractor."""

from unittest.mock import MagicMock, patch

import pytest
from py_web_text_extractor.exception.exceptions import TextExtractionError, UrlIsNotValidException

from py_search_helper.exceptions import URLError, URLNotFoundError, URLTimeoutError
from py_search_helper.extractors import extract_content, extract_content_safe, get_extractor_service


def test_get_extractor_service_creates_singleton() -> None:
    """Test that get_extractor_service returns singleton instance."""
    service1 = get_extractor_service()
    service2 = get_extractor_service()

    assert service1 is service2
    assert service1 is not None


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_success(mock_create_service: MagicMock) -> None:
    """Test successful content extraction."""
    # Reset the singleton for this test
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    # Mock the service
    mock_service = MagicMock()
    mock_service.extract_text_from_page.return_value = "# Test Content\n\nThis is test content."
    mock_create_service.return_value = mock_service

    result = extract_content("https://example.com")

    assert result == "# Test Content\n\nThis is test content."
    mock_service.extract_text_from_page.assert_called_once_with("https://example.com")


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_invalid_url(mock_create_service: MagicMock) -> None:
    """Test that invalid URL raises URLError."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page.side_effect = UrlIsNotValidException("Invalid URL")
    mock_create_service.return_value = mock_service

    with pytest.raises(URLError, match="Invalid URL"):
        extract_content("invalid-url")


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_not_found(mock_create_service: MagicMock) -> None:
    """Test that 404 errors raise URLNotFoundError."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page.side_effect = TextExtractionError("404 not found")
    mock_create_service.return_value = mock_service

    with pytest.raises(URLNotFoundError, match="URL not found"):
        extract_content("https://example.com/notfound")


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_timeout(mock_create_service: MagicMock) -> None:
    """Test that timeout errors raise URLTimeoutError."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page.side_effect = TextExtractionError("Request timed out")
    mock_create_service.return_value = mock_service

    with pytest.raises(URLTimeoutError, match="Request timed out"):
        extract_content("https://slow-example.com")


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_timeout_alternative_message(mock_create_service: MagicMock) -> None:
    """Test that timeout errors with different message format raise URLTimeoutError."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page.side_effect = TextExtractionError("Connection timeout")
    mock_create_service.return_value = mock_service

    with pytest.raises(URLTimeoutError, match="Request timed out"):
        extract_content("https://slow-example.com")


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_extraction_error(mock_create_service: MagicMock) -> None:
    """Test that general extraction errors raise URLError."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page.side_effect = TextExtractionError("Failed to extract")
    mock_create_service.return_value = mock_service

    with pytest.raises(URLError, match="Content extraction failed"):
        extract_content("https://example.com")


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_unexpected_exception(mock_create_service: MagicMock) -> None:
    """Test that unexpected exceptions raise URLError."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page.side_effect = ValueError("Unexpected error")
    mock_create_service.return_value = mock_service

    with pytest.raises(URLError, match="Unexpected error extracting content"):
        extract_content("https://example.com")


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_safe_success(mock_create_service: MagicMock) -> None:
    """Test successful safe mode extraction."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page_safe.return_value = "Safe content"
    mock_create_service.return_value = mock_service

    result = extract_content_safe("https://example.com")

    assert result == "Safe content"
    mock_service.extract_text_from_page_safe.assert_called_once_with("https://example.com")


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_safe_returns_empty_on_error(mock_create_service: MagicMock) -> None:
    """Test that safe mode returns empty string on error."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page_safe.return_value = ""
    mock_create_service.return_value = mock_service

    result = extract_content_safe("https://invalid-url.com")

    assert result == ""


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_extract_content_safe_handles_exceptions(mock_create_service: MagicMock) -> None:
    """Test that safe mode handles unexpected exceptions."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page_safe.side_effect = Exception("Unexpected error")
    mock_create_service.return_value = mock_service

    result = extract_content_safe("https://example.com")

    assert result == ""


@patch("py_search_helper.extractors.content.create_extractor_service")
def test_service_reuse(mock_create_service: MagicMock) -> None:
    """Test that service is reused across calls."""
    import py_search_helper.extractors.content as content_module

    content_module._extractor_service = None

    mock_service = MagicMock()
    mock_service.extract_text_from_page.return_value = "Content"
    mock_create_service.return_value = mock_service

    # First call
    extract_content("https://example.com/1")

    # Second call
    extract_content("https://example.com/2")

    # Service should only be created once
    mock_create_service.assert_called_once()

    # Service should be used twice
    assert mock_service.extract_text_from_page.call_count == 2
