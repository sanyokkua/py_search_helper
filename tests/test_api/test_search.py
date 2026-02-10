"""Tests for search API function."""

from unittest.mock import MagicMock, patch

import pytest

from py_search_helper import search
from py_search_helper.api.search import _build_query
from py_search_helper.exceptions import EngineNotFoundError, SearchProviderError
from py_search_helper.models import SearchResult


def test_search_validates_empty_query() -> None:
    """Test that search validates empty query."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        search(engine="ddgs", query="")


def test_search_validates_whitespace_only_query() -> None:
    """Test that search validates whitespace-only query."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        search(engine="ddgs", query="   ")


def test_search_validates_empty_engine() -> None:
    """Test that search validates empty engine ID."""
    with pytest.raises(ValueError, match="Engine ID cannot be empty"):
        search(engine="", query="test")


def test_search_validates_max_results() -> None:
    """Test that search validates max_results parameter."""
    with pytest.raises(ValueError, match="max_results must be at least 1"):
        search(engine="ddgs", query="test", max_results=0)

    with pytest.raises(ValueError, match="max_results must be at least 1"):
        search(engine="ddgs", query="test", max_results=-1)


@patch("py_search_helper.api.search.get_registry")
def test_search_raises_engine_not_found(mock_get_registry: MagicMock) -> None:
    """Test that search raises EngineNotFoundError for unknown engine."""
    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = None
    mock_get_registry.return_value = mock_registry

    with pytest.raises(EngineNotFoundError, match="Engine 'unknown' not found"):
        search(engine="unknown", query="test")


@patch("py_search_helper.api.search.get_registry")
def test_search_calls_provider_search(mock_get_registry: MagicMock) -> None:
    """Test that search calls provider's search method."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = [
        SearchResult(title="Result 1", url="https://example.com/1", description="Description 1"),
        SearchResult(title="Result 2", url="https://example.com/2", description="Description 2"),
    ]

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    result = search(engine="test", query="python", max_results=5)

    mock_registry.get_provider.assert_called_once_with("test")
    mock_provider.search.assert_called_once_with("python", max_results=5)
    assert isinstance(result, str)
    assert "Result 1" in result
    assert "Result 2" in result


@patch("py_search_helper.api.search.get_registry")
def test_search_formats_results_as_markdown(mock_get_registry: MagicMock) -> None:
    """Test that search formats results as Markdown."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = [
        SearchResult(title="Test Result", url="https://example.com", description="Test description"),
    ]

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    result = search(engine="test", query="python")

    # Check Markdown formatting
    assert result.startswith("# Search Results for")
    assert "## Result 1" in result
    assert "**Title:**" in result
    assert "**URL:**" in result
    assert "**Description:**" in result
    assert "Test Result" in result
    assert "https://example.com" in result
    assert "Test description" in result


@patch("py_search_helper.api.search.get_registry")
def test_search_handles_no_results(mock_get_registry: MagicMock) -> None:
    """Test that search handles empty results gracefully."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = []

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    result = search(engine="test", query="python")

    assert "No results found" in result


@patch("py_search_helper.api.search.get_registry")
def test_search_propagates_search_provider_error(mock_get_registry: MagicMock) -> None:
    """Test that SearchProviderError is propagated."""
    mock_provider = MagicMock()
    mock_provider.search.side_effect = SearchProviderError("Provider failed")

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    with pytest.raises(SearchProviderError, match="Provider failed"):
        search(engine="test", query="python")


@patch("py_search_helper.api.search.get_registry")
def test_search_wraps_unexpected_exceptions(mock_get_registry: MagicMock) -> None:
    """Test that unexpected exceptions are wrapped in SearchProviderError."""
    mock_provider = MagicMock()
    mock_provider.search.side_effect = RuntimeError("Unexpected error")

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    with pytest.raises(SearchProviderError, match="Search failed for engine 'test'"):
        search(engine="test", query="python")


@patch("py_search_helper.api.search.get_registry")
def test_search_respects_max_results(mock_get_registry: MagicMock) -> None:
    """Test that search passes max_results to provider."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = []

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    search(engine="test", query="python", max_results=20)

    mock_provider.search.assert_called_once_with("python", max_results=20)


@patch("py_search_helper.api.search.get_registry")
def test_search_uses_default_max_results(mock_get_registry: MagicMock) -> None:
    """Test that search uses default max_results=10."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = []

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    search(engine="test", query="python")

    mock_provider.search.assert_called_once_with("python", max_results=10)


@patch("py_search_helper.api.search.get_registry")
def test_search_includes_query_and_engine_in_output(mock_get_registry: MagicMock) -> None:
    """Test that search output includes query and engine."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = []

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    result = search(engine="test_engine", query="test query")

    assert "test query" in result
    assert "test_engine" in result


# Site parameter tests


def test_search_validates_empty_site() -> None:
    """Test that search validates empty site parameter."""
    with pytest.raises(ValueError, match="Site parameter cannot be empty"):
        search(engine="ddgs", query="test", site="")


def test_search_validates_whitespace_only_site() -> None:
    """Test that search validates whitespace-only site parameter."""
    with pytest.raises(ValueError, match="Site parameter cannot be empty"):
        search(engine="ddgs", query="test", site="   ")


@patch("py_search_helper.api.search.get_registry")
def test_search_accepts_none_site(mock_get_registry: MagicMock) -> None:
    """Test that search accepts None as site parameter."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = []

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    # Should not raise
    search(engine="ddgs", query="test", site=None)


@patch("py_search_helper.api.search.get_registry")
def test_search_builds_query_with_site_filter(mock_get_registry: MagicMock) -> None:
    """Test that search constructs query with site filter."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = []

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    search(engine="ddgs", query="python", site="python.org")

    # Verify provider received query with site filter
    mock_provider.search.assert_called_once_with("python site:python.org", max_results=10)


@patch("py_search_helper.api.search.get_registry")
def test_search_without_site_filter(mock_get_registry: MagicMock) -> None:
    """Test that search passes query unchanged when no site filter."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = []

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    search(engine="ddgs", query="python")

    # Verify provider received original query
    mock_provider.search.assert_called_once_with("python", max_results=10)


@patch("py_search_helper.api.search.get_registry")
def test_search_includes_site_in_markdown_output(mock_get_registry: MagicMock) -> None:
    """Test that search output includes site filter in header."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = [
        SearchResult(title="Test", url="https://python.org/test", description="Test"),
    ]

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    result = search(engine="ddgs", query="asyncio", site="python.org")

    assert "asyncio" in result
    assert "python.org" in result
    assert "site:" in result


@patch("py_search_helper.api.search.get_registry")
def test_search_backward_compatible_without_site_param(mock_get_registry: MagicMock) -> None:
    """Test that existing code without site parameter still works."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = []

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_get_registry.return_value = mock_registry

    # Old-style call without site parameter
    result = search("ddgs", "python", 5)

    assert isinstance(result, str)
    mock_provider.search.assert_called_once_with("python", max_results=5)


def test_search_requires_keyword_for_site() -> None:
    """Test that site must be passed as keyword argument."""
    with (
        patch("py_search_helper.api.search.get_registry"),
        pytest.raises(TypeError),
    ):
        # This should raise TypeError (positional argument)
        search("ddgs", "python", 5, "python.org")  # type: ignore


# Query builder helper tests


def test_build_query_without_site() -> None:
    """Test query building without site filter."""
    result = _build_query("python asyncio", None)
    assert result == "python asyncio"


def test_build_query_with_site() -> None:
    """Test query building with site filter."""
    result = _build_query("python asyncio", "python.org")
    assert result == "python asyncio site:python.org"
