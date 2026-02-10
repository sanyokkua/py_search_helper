"""Tests for Wikipedia provider."""

from unittest.mock import Mock, patch

import pytest

from py_search_helper.models import EngineInfo, SearchResult
from py_search_helper.providers import WikipediaProvider


@pytest.fixture
def provider() -> WikipediaProvider:
    """Provide a Wikipedia provider instance."""
    return WikipediaProvider()


def test_wikipedia_provider_initialization(provider: WikipediaProvider) -> None:
    """Test Wikipedia provider initialization."""
    assert isinstance(provider, WikipediaProvider)


def test_get_info(provider: WikipediaProvider) -> None:
    """Test getting provider information."""
    info = provider.get_info()

    assert isinstance(info, EngineInfo)
    assert info.id == "wikipedia"
    assert info.name == "Wikipedia"
    assert info.description == "Wikipedia encyclopedia"
    assert info.scope == "Wikipedia articles and encyclopedia content"
    assert info.base_url == "https://en.wikipedia.org"


@patch("py_search_helper.providers.wikipedia_provider.DDGSProvider")
def test_search_constructs_site_filter(mock_ddgs_class: Mock, provider: WikipediaProvider) -> None:
    """Test that search adds site: filter to query."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    provider.search("Python programming", max_results=10)

    # Verify DDGS was called with site filter
    mock_ddgs.search.assert_called_once_with("Python programming site:wikipedia.org", max_results=10)


@patch("py_search_helper.providers.wikipedia_provider.DDGSProvider")
def test_search_returns_results(mock_ddgs_class: Mock, provider: WikipediaProvider) -> None:
    """Test that search returns list of SearchResult objects."""
    # Arrange
    mock_results = [
        SearchResult(
            title="Python (programming language)",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            description="Python is a high-level, interpreted programming language.",
        ),
        SearchResult(
            title="History of Python",
            url="https://en.wikipedia.org/wiki/History_of_Python",
            description="The history of the Python programming language.",
        ),
    ]
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = mock_results
    mock_ddgs_class.return_value = mock_ddgs

    # Act
    results = provider.search("Python programming", max_results=5)

    # Assert
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].title == "Python (programming language)"
    assert "wikipedia.org" in results[0].url


@patch("py_search_helper.providers.wikipedia_provider.DDGSProvider")
def test_search_with_max_results(mock_ddgs_class: Mock, provider: WikipediaProvider) -> None:
    """Test that search respects max_results parameter."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    provider.search("Python", max_results=3)

    # Verify max_results is passed through
    mock_ddgs.search.assert_called_once_with("Python site:wikipedia.org", max_results=3)


@patch("py_search_helper.providers.wikipedia_provider.DDGSProvider")
def test_search_empty_query_validation(mock_ddgs_class: Mock, provider: WikipediaProvider) -> None:
    """Test that empty query validation is handled by DDGS provider."""
    mock_ddgs = Mock()
    mock_ddgs.search.side_effect = ValueError("Query cannot be empty")
    mock_ddgs_class.return_value = mock_ddgs

    with pytest.raises(ValueError, match="Query cannot be empty"):
        provider.search("", max_results=10)


# Site filter handling tests


@patch("py_search_helper.providers.wikipedia_provider.DDGSProvider")
def test_wikipedia_preserves_existing_site_filter(mock_ddgs_class: Mock, provider: WikipediaProvider) -> None:
    """Test that Wikipedia preserves site filter if already present."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    provider.search("Python site:fr.wikipedia.org", max_results=10)

    # Should NOT add second site filter
    mock_ddgs.search.assert_called_once_with("Python site:fr.wikipedia.org", max_results=10)


@patch("py_search_helper.providers.wikipedia_provider.DDGSProvider")
def test_wikipedia_adds_default_site_when_missing(mock_ddgs_class: Mock, provider: WikipediaProvider) -> None:
    """Test that Wikipedia adds default site when not present."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    provider.search("Python", max_results=10)

    # Should add Wikipedia default site filter
    mock_ddgs.search.assert_called_once_with("Python site:wikipedia.org", max_results=10)


@patch("py_search_helper.providers.wikipedia_provider.DDGSProvider")
def test_wikipedia_allows_language_specific_search(mock_ddgs_class: Mock, provider: WikipediaProvider) -> None:
    """Test that Wikipedia allows language-specific searches with site: filter."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    # Test various language editions
    provider.search("Python site:de.wikipedia.org", max_results=10)
    mock_ddgs.search.assert_called_with("Python site:de.wikipedia.org", max_results=10)

    provider.search("Python site:ja.wikipedia.org", max_results=10)
    mock_ddgs.search.assert_called_with("Python site:ja.wikipedia.org", max_results=10)


@patch("py_search_helper.providers.wikipedia_provider.DDGSProvider")
def test_wikipedia_allows_api_site_parameter(mock_ddgs_class: Mock, provider: WikipediaProvider) -> None:
    """Test that Wikipedia preserves site: in query passed from API."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    # Simulate API call with explicit site parameter
    provider.search("Python site:es.wikipedia.org", max_results=10)

    # Should preserve the API-provided site filter
    mock_ddgs.search.assert_called_once_with("Python site:es.wikipedia.org", max_results=10)


@patch("py_search_helper.providers.wikipedia_provider.DDGSProvider")
def test_wikipedia_site_filter_exact_match(mock_ddgs_class: Mock, provider: WikipediaProvider) -> None:
    """Test that Wikipedia site filter is added with exact format."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    provider.search("Machine learning", max_results=10)

    # Verify exact format: "{query} site:wikipedia.org"
    call_args = mock_ddgs.search.call_args
    assert call_args[0][0] == "Machine learning site:wikipedia.org"
    assert call_args[1]["max_results"] == 10
