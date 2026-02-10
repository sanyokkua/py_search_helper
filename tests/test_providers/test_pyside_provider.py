"""Tests for PySide provider."""

from unittest.mock import Mock, patch

import pytest

from py_search_helper.models import EngineInfo, SearchResult
from py_search_helper.providers import PySideProvider


@pytest.fixture
def provider() -> PySideProvider:
    """Provide a PySide provider instance."""
    return PySideProvider()


def test_pyside_provider_initialization(provider: PySideProvider) -> None:
    """Test PySide provider initialization."""
    assert isinstance(provider, PySideProvider)


def test_get_info(provider: PySideProvider) -> None:
    """Test getting provider information."""
    info = provider.get_info()

    assert isinstance(info, EngineInfo)
    assert info.id == "pyside"
    assert info.name == "PySide Documentation"
    assert info.description == "Qt for Python official documentation"
    assert info.scope == "PySide6/Qt for Python API documentation and guides"
    assert info.base_url == "https://doc.qt.io/qtforpython-6"


@patch("py_search_helper.providers.pyside_provider.DDGSProvider")
def test_search_constructs_site_filter(mock_ddgs_class: Mock, provider: PySideProvider) -> None:
    """Test that search adds site: filter to query."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    provider.search("QPushButton", max_results=10)

    # Verify DDGS was called with site filter
    mock_ddgs.search.assert_called_once_with("QPushButton site:doc.qt.io/qtforpython-6", max_results=10)


@patch("py_search_helper.providers.pyside_provider.DDGSProvider")
def test_search_returns_results(mock_ddgs_class: Mock, provider: PySideProvider) -> None:
    """Test that search returns list of SearchResult objects."""
    # Arrange
    mock_results = [
        SearchResult(
            title="QPushButton Class",
            url="https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QPushButton.html",
            description="The QPushButton widget provides a command button.",
        ),
        SearchResult(
            title="QLabel Class",
            url="https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QLabel.html",
            description="The QLabel widget provides a text or image display.",
        ),
    ]
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = mock_results
    mock_ddgs_class.return_value = mock_ddgs

    # Act
    results = provider.search("QPushButton", max_results=5)

    # Assert
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].title == "QPushButton Class"
    assert "doc.qt.io/qtforpython-6" in results[0].url


@patch("py_search_helper.providers.pyside_provider.DDGSProvider")
def test_search_with_max_results(mock_ddgs_class: Mock, provider: PySideProvider) -> None:
    """Test that search respects max_results parameter."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    provider.search("QLabel", max_results=3)

    # Verify max_results is passed through
    mock_ddgs.search.assert_called_once_with("QLabel site:doc.qt.io/qtforpython-6", max_results=3)


@patch("py_search_helper.providers.pyside_provider.DDGSProvider")
def test_search_empty_query_validation(mock_ddgs_class: Mock, provider: PySideProvider) -> None:
    """Test that empty query validation is handled by DDGS provider."""
    mock_ddgs = Mock()
    mock_ddgs.search.side_effect = ValueError("Query cannot be empty")
    mock_ddgs_class.return_value = mock_ddgs

    with pytest.raises(ValueError, match="Query cannot be empty"):
        provider.search("", max_results=10)


# Site filter handling tests


@patch("py_search_helper.providers.pyside_provider.DDGSProvider")
def test_pyside_preserves_existing_site_filter(mock_ddgs_class: Mock, provider: PySideProvider) -> None:
    """Test that PySide preserves site filter if already present."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    provider.search("QPushButton site:doc.qt.io", max_results=10)

    # Should NOT add second site filter
    mock_ddgs.search.assert_called_once_with("QPushButton site:doc.qt.io", max_results=10)


@patch("py_search_helper.providers.pyside_provider.DDGSProvider")
def test_pyside_adds_default_site_when_missing(mock_ddgs_class: Mock, provider: PySideProvider) -> None:
    """Test that PySide adds default site when not present."""
    mock_ddgs = Mock()
    mock_ddgs.search.return_value = []
    mock_ddgs_class.return_value = mock_ddgs

    provider.search("QPushButton", max_results=10)

    # Should add PySide default site filter
    mock_ddgs.search.assert_called_once_with("QPushButton site:doc.qt.io/qtforpython-6", max_results=10)
