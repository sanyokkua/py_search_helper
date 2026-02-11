import pytest
from typer.testing import CliRunner
from unittest.mock import Mock

from py_search_helper.cli import app
from py_search_helper.exceptions import EngineNotFoundError, URLError, SearchProviderError

runner = CliRunner()

@pytest.fixture
def mock_get_search_engines(mocker):
    """Fixture to mock py_search_helper.get_search_engines."""
    mock = mocker.patch("py_search_helper.cli.get_search_engines")
    mock.return_value = [
        ("ddgs", "General web search (DuckDuckGo)"),
        ("wikipedia", "Wikipedia encyclopedia"),
    ]
    return mock

@pytest.fixture
def mock_search(mocker):
    """Fixture to mock py_search_helper.search."""
    mock = mocker.patch("py_search_helper.cli.search")
    mock.return_value = """## Search Results
- Result 1
- Result 2"""
    return mock

@pytest.fixture
def mock_open_url(mocker):
    """Fixture to mock py_search_helper.open_url."""
    mock = mocker.patch("py_search_helper.cli.open_url")
    mock.return_value = "This is some extracted content."
    return mock

def test_get_engines_success(mock_get_search_engines):
    result = runner.invoke(app, ["get-engines"])
    assert result.exit_code == 0
    assert "Available Search Engines:" in result.stdout
    assert "ddgs: General web search (DuckDuckGo)" in result.stdout
    mock_get_search_engines.assert_called_once()

def test_get_engines_error(mock_get_search_engines):
    mock_get_search_engines.side_effect = EngineNotFoundError("Test error")
    result = runner.invoke(app, ["get-engines"])
    assert result.exit_code == 1
    assert "Error: Test error" in result.stderr
    mock_get_search_engines.assert_called_once()

def test_search_success(mock_search):
    result = runner.invoke(app, ["search", "python typer", "-e","ddgs"])
    assert result.exit_code == 0
    assert "## Search Results" in result.stdout
    assert "- Result 1" in result.stdout
    mock_search.assert_called_once_with(engine="ddgs", query="python typer", max_results=10, site=None)

def test_search_with_options_success(mock_search):
    result = runner.invoke(app, ["search", "LLM", "--max-results", "5", "-s", "en.wikipedia.org", "-e","wikipedia"])
    assert result.exit_code == 0
    assert "## Search Results" in result.stdout
    mock_search.assert_called_once_with(engine="wikipedia", query="LLM", max_results=5, site="en.wikipedia.org")

def test_search_engine_not_found_error(mock_search):
    mock_search.side_effect = EngineNotFoundError("Engine 'invalid' not found")
    result = runner.invoke(app, ["search", "query", "-e", "invalid"])
    assert result.exit_code == 1
    assert "Error: Engine 'invalid' not found" in result.stderr
    mock_search.assert_called_once()

def test_search_provider_error(mock_search):
    mock_search.side_effect = SearchProviderError("DDGS search failed")
    result = runner.invoke(app, ["search", "query", "-e","ddgs"])
    assert result.exit_code == 1
    assert "Error: DDGS search failed" in result.stderr
    mock_search.assert_called_once()

def test_open_page_success(mock_open_url):
    result = runner.invoke(app, ["open-page", "https://example.com"])
    assert result.exit_code == 0
    assert "This is some extracted content." in result.stdout
    mock_open_url.assert_called_once_with(url="https://example.com", max_chars=500)

def test_open_page_with_options_success(mock_open_url):
    result = runner.invoke(app, ["open-page", "https://example.com/full", "--max-chars", "0"])
    assert result.exit_code == 0
    assert "This is some extracted content." in result.stdout
    mock_open_url.assert_called_once_with(url="https://example.com/full", max_chars=None)

def test_open_page_url_error(mock_open_url):
    mock_open_url.side_effect = URLError("Invalid URL")
    result = runner.invoke(app, ["open-page", "invalid-url"])
    assert result.exit_code == 1
    assert "Error: Invalid URL" in result.stderr
    mock_open_url.assert_called_once()

def test_help_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "A CLI for searching the internet and extracting web content." in result.stdout
    assert "Commands" in result.stdout  # Check for "Commands" within the section header
    assert "get-engines" in result.stdout
    assert "search" in result.stdout
    assert "open-page" in result.stdout

def test_search_help():
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "Search the web using a specified engine." in result.stdout
    assert "Arguments" in result.stdout
    assert "query" in result.stdout # Assert more specific argument description
    assert "Options" in result.stdout
    assert "--max-results" in result.stdout
    assert "--engine" in result.stdout
