"""Tests for open_url API function."""

from unittest.mock import MagicMock, patch

import pytest

from py_search_helper import open_url
from py_search_helper.exceptions import URLError, URLNotFoundError, URLTimeoutError


def test_open_url_validates_empty_url() -> None:
    """Test that open_url validates empty URL."""
    with pytest.raises(ValueError, match="URL cannot be empty"):
        open_url("")


def test_open_url_validates_url_format() -> None:
    """Test that open_url validates URL format."""
    with pytest.raises(ValueError, match="URL must start with http://"):
        open_url("not-a-url")

    with pytest.raises(ValueError, match="URL must start with http://"):
        open_url("ftp://example.com")


def test_open_url_validates_max_chars() -> None:
    """Test that open_url validates max_chars parameter."""
    with pytest.raises(ValueError, match="max_chars must be at least 1"):
        open_url("https://example.com", max_chars=0)

    with pytest.raises(ValueError, match="max_chars must be at least 1"):
        open_url("https://example.com", max_chars=-1)


def test_open_url_accepts_none_max_chars() -> None:
    """Test that open_url accepts None for max_chars."""
    # Should not raise ValueError
    with patch("py_search_helper.api.open.extract_content") as mock_extract:
        mock_extract.return_value = "content"
        result = open_url("https://example.com", max_chars=None)
        assert result == "content"


@patch("py_search_helper.api.open.extract_content")
def test_open_url_calls_extract_content(mock_extract: MagicMock) -> None:
    """Test that open_url calls extract_content."""
    mock_extract.return_value = "Test content"

    result = open_url("https://example.com", max_chars=None)

    mock_extract.assert_called_once_with("https://example.com")
    assert result == "Test content"


@patch("py_search_helper.api.open.extract_content")
def test_open_url_truncates_with_max_chars(mock_extract: MagicMock) -> None:
    """Test that open_url truncates content when max_chars is specified."""
    mock_extract.return_value = "This is a very long content that should be truncated"

    result = open_url("https://example.com", max_chars=20)

    assert len(result) == 23  # 20 chars + "..."
    assert result == "This is a very long ..."


@patch("py_search_helper.api.open.extract_content")
def test_open_url_does_not_truncate_short_content(mock_extract: MagicMock) -> None:
    """Test that open_url doesn't truncate content shorter than max_chars."""
    mock_extract.return_value = "Short content"

    result = open_url("https://example.com", max_chars=100)

    assert result == "Short content"
    assert "..." not in result


@patch("py_search_helper.api.open.extract_content")
def test_open_url_uses_default_max_chars(mock_extract: MagicMock) -> None:
    """Test that open_url uses default max_chars=500."""
    long_content = "x" * 1000
    mock_extract.return_value = long_content

    result = open_url("https://example.com")

    assert len(result) == 503  # 500 + "..."
    assert result.endswith("...")


@patch("py_search_helper.api.open.extract_content")
def test_open_url_propagates_url_not_found_error(mock_extract: MagicMock) -> None:
    """Test that URLNotFoundError is propagated."""
    mock_extract.side_effect = URLNotFoundError("URL not found")

    with pytest.raises(URLNotFoundError, match="URL not found"):
        open_url("https://example.com/notfound")


@patch("py_search_helper.api.open.extract_content")
def test_open_url_propagates_url_timeout_error(mock_extract: MagicMock) -> None:
    """Test that URLTimeoutError is propagated."""
    mock_extract.side_effect = URLTimeoutError("Request timed out")

    with pytest.raises(URLTimeoutError, match="Request timed out"):
        open_url("https://slow-example.com")


@patch("py_search_helper.api.open.extract_content")
def test_open_url_propagates_url_error(mock_extract: MagicMock) -> None:
    """Test that URLError is propagated."""
    mock_extract.side_effect = URLError("Extraction failed")

    with pytest.raises(URLError, match="Extraction failed"):
        open_url("https://example.com")


@patch("py_search_helper.api.open.extract_content")
def test_open_url_wraps_unexpected_exceptions(mock_extract: MagicMock) -> None:
    """Test that unexpected exceptions are wrapped in URLError."""
    mock_extract.side_effect = RuntimeError("Unexpected error")

    with pytest.raises(URLError, match="Failed to open URL"):
        open_url("https://example.com")


@patch("py_search_helper.api.open.extract_content")
def test_open_url_accepts_http_and_https(mock_extract: MagicMock) -> None:
    """Test that open_url accepts both http:// and https:// URLs."""
    mock_extract.return_value = "content"

    # Should not raise
    open_url("http://example.com")
    open_url("https://example.com")

    assert mock_extract.call_count == 2


@patch("py_search_helper.api.open.extract_content")
def test_open_url_truncation_boundary(mock_extract: MagicMock) -> None:
    """Test truncation at exact boundary."""
    # Exactly max_chars length - should not truncate
    mock_extract.return_value = "x" * 100
    result = open_url("https://example.com", max_chars=100)
    assert result == "x" * 100
    assert "..." not in result

    # One char over max_chars - should truncate
    mock_extract.return_value = "x" * 101
    result = open_url("https://example.com", max_chars=100)
    assert len(result) == 103
    assert result == "x" * 100 + "..."
