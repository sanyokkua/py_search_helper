"""Tests for DDGS provider."""

import time
from unittest.mock import MagicMock, patch

import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from py_search_helper.exceptions import SearchProviderError
from py_search_helper.models import EngineInfo, SearchResult
from py_search_helper.providers import DDGSProvider


@pytest.fixture
def provider() -> DDGSProvider:
    """Provide a DDGS provider instance."""
    return DDGSProvider()


def test_ddgs_provider_initialization() -> None:
    """Test DDGS provider initialization."""
    provider = DDGSProvider()
    assert provider.min_interval == 10.0  # 60/6 = 10 seconds
    assert provider.last_request_time == 0.0


def test_ddgs_provider_custom_rate_limit() -> None:
    """Test DDGS provider with custom rate limit."""
    provider = DDGSProvider(requests_per_minute=12)
    assert provider.min_interval == 5.0  # 60/12 = 5 seconds


def test_get_info(provider: DDGSProvider) -> None:
    """Test getting provider information."""
    info = provider.get_info()

    assert isinstance(info, EngineInfo)
    assert info.id == "ddgs"
    assert info.name == "DuckDuckGo Search"
    assert info.description == "General web search (DuckDuckGo)"
    assert info.scope == "General web content from multiple sources"
    assert info.base_url == "https://duckduckgo.com"


@patch("py_search_helper.providers.ddgs_provider.DDGS")
def test_search_success(mock_ddgs_class: MagicMock, provider: DDGSProvider) -> None:
    """Test successful search."""
    # Mock DDGS response
    mock_ddgs = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.text.return_value = [
        {"title": "Result 1", "href": "https://example.com/1", "body": "Description 1"},
        {"title": "Result 2", "href": "https://example.com/2", "body": "Description 2"},
    ]

    results = provider.search("test query", max_results=2)

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].title == "Result 1"
    assert results[0].url == "https://example.com/1"
    assert results[0].description == "Description 1"
    assert results[1].title == "Result 2"
    assert results[1].url == "https://example.com/2"
    assert results[1].description == "Description 2"


@patch("py_search_helper.providers.ddgs_provider.DDGS")
def test_search_respects_max_results_limit(mock_ddgs_class: MagicMock, provider: DDGSProvider) -> None:
    """Test that search limits max_results to 30."""
    mock_ddgs = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.text.return_value = []

    provider.search("test", max_results=100)

    # Verify that DDGS was called with max 30 results
    mock_ddgs.text.assert_called_once_with("test", max_results=30)


@patch("py_search_helper.providers.ddgs_provider.DDGS")
@patch("time.sleep")
def test_search_rate_limit_retry(mock_sleep: MagicMock, mock_ddgs_class: MagicMock, provider: DDGSProvider) -> None:
    """Test that search retries on rate limit error."""
    mock_ddgs = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

    # First call raises RatelimitException, second succeeds
    mock_ddgs.text.side_effect = [
        RatelimitException("Rate limited"),
        [{"title": "Result", "href": "https://example.com", "body": "Description"}],
    ]

    results = provider.search("test", max_results=5)

    # Verify retry happened
    assert mock_ddgs.text.call_count == 2
    mock_sleep.assert_called_with(60)  # Should sleep 60 seconds
    assert len(results) == 1


@patch("py_search_helper.providers.ddgs_provider.DDGS")
@patch("time.sleep")
def test_search_rate_limit_retry_fails(mock_sleep: MagicMock, mock_ddgs_class: MagicMock, provider: DDGSProvider) -> None:
    """Test that search raises error if retry also fails."""
    mock_ddgs = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

    # Both calls raise RatelimitException
    mock_ddgs.text.side_effect = [
        RatelimitException("Rate limited"),
        RatelimitException("Still rate limited"),
    ]

    with pytest.raises(SearchProviderError, match="DDGS rate limit retry failed"):
        provider.search("test", max_results=5)

    mock_sleep.assert_called_with(60)


@patch("py_search_helper.providers.ddgs_provider.DDGS")
def test_search_timeout_error(mock_ddgs_class: MagicMock, provider: DDGSProvider) -> None:
    """Test that timeout errors are handled."""
    mock_ddgs = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.text.side_effect = TimeoutException("Timeout")

    with pytest.raises(SearchProviderError, match="DDGS search timeout"):
        provider.search("test", max_results=5)


@patch("py_search_helper.providers.ddgs_provider.DDGS")
def test_search_ddgs_exception(mock_ddgs_class: MagicMock, provider: DDGSProvider) -> None:
    """Test that DDGS exceptions are handled."""
    mock_ddgs = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.text.side_effect = DDGSException("DDGS error")

    with pytest.raises(SearchProviderError, match="DDGS search failed"):
        provider.search("test", max_results=5)


@patch("py_search_helper.providers.ddgs_provider.DDGS")
def test_search_unexpected_exception(mock_ddgs_class: MagicMock, provider: DDGSProvider) -> None:
    """Test that unexpected exceptions are handled."""
    mock_ddgs = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.text.side_effect = ValueError("Unexpected error")

    with pytest.raises(SearchProviderError, match="Unexpected error during DDGS search"):
        provider.search("test", max_results=5)


@patch("time.time")
@patch("time.sleep")
@patch("py_search_helper.providers.ddgs_provider.DDGS")
def test_rate_limiting_enforced(
    mock_ddgs_class: MagicMock, mock_sleep: MagicMock, mock_time: MagicMock, provider: DDGSProvider
) -> None:
    """Test that rate limiting is enforced between requests."""
    mock_ddgs = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.text.return_value = []

    # Mock time progression: provider.min_interval = 10 seconds
    # First search: time=100, elapsed=100-0=100 (no sleep), then last_request_time=100
    # Second search: time=102, elapsed=102-100=2 < 10, sleep(8), then last_request_time=102
    mock_time.side_effect = [100.0, 100.0, 102.0, 102.0]

    # First search
    provider.search("test1", max_results=1)

    # Second search shortly after (2 seconds later)
    provider.search("test2", max_results=1)

    # Should have slept once for remaining time (10 - 2 = 8 seconds)
    mock_sleep.assert_called_once_with(8.0)


@patch("py_search_helper.providers.ddgs_provider.DDGS")
def test_search_handles_missing_fields(mock_ddgs_class: MagicMock, provider: DDGSProvider) -> None:
    """Test that search handles results with missing fields."""
    mock_ddgs = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.text.return_value = [
        {},  # All fields missing
        {"title": "Only Title"},  # Missing href and body
        {"href": "https://example.com"},  # Missing title and body
    ]

    results = provider.search("test", max_results=3)

    assert len(results) == 3
    assert results[0].title == ""
    assert results[0].url == ""
    assert results[0].description == ""
    assert results[1].title == "Only Title"
    assert results[1].url == ""
    assert results[2].url == "https://example.com"
