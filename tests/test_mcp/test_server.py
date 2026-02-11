"""Tests for MCP server."""

from unittest.mock import MagicMock, patch

import pytest

from py_search_helper.mcp.server import main, mcp


def test_mcp_server_instance() -> None:
    """Test that MCP server instance exists."""
    assert mcp is not None
    assert hasattr(mcp, "tool")
    assert hasattr(mcp, "run")


def test_mcp_server_has_tools() -> None:
    """Test that MCP server has registered tools."""
    # Test that we can import the tool functions
    from py_search_helper.mcp.server import get_engines, open_page, search_web, search_web_ddg

    assert get_engines is not None
    assert search_web is not None
    assert open_page is not None
    assert search_web_ddg is not None


@patch("py_search_helper.mcp.server.get_search_engines")
def test_get_engines_integration(mock_get_search_engines: MagicMock) -> None:
    """Test get_engines through MCP integration."""
    from py_search_helper.mcp.server import get_engines

    mock_get_search_engines.return_value = [
        ("test1", "Test Engine 1"),
        ("test2", "Test Engine 2"),
    ]

    # Call the function that's wrapped by @mcp.tool()
    # We need to get the actual function, not the FunctionTool wrapper
    if hasattr(get_engines, "fn"):
        result = get_engines.fn()
    else:
        result = get_engines()

    mock_get_search_engines.assert_called_once()
    assert result == [("test1", "Test Engine 1"), ("test2", "Test Engine 2")]


@patch("py_search_helper.mcp.server.search")
def test_search_web_integration(mock_search: MagicMock) -> None:
    """Test search_web through MCP integration."""
    from py_search_helper.mcp.server import search_web

    mock_search.return_value = "Search results"

    # Call the function
    if hasattr(search_web, "fn"):
        result = search_web.fn(engine="ddgs", query="python", max_results=5)
    else:
        result = search_web(engine="ddgs", query="python", max_results=5)

    mock_search.assert_called_once_with(engine="ddgs", query="python", max_results=5, site=None)
    assert result == "Search results"


@patch("py_search_helper.mcp.server.search")
def test_search_web_with_site_parameter(mock_search: MagicMock) -> None:
    """Test search_web with site parameter."""
    from py_search_helper.mcp.server import search_web

    mock_search.return_value = "Search results"

    # Call the function
    if hasattr(search_web, "fn"):
        result = search_web.fn(engine="ddgs", query="python", max_results=5, site="python.org")
    else:
        result = search_web(engine="ddgs", query="python", max_results=5, site="python.org")

    mock_search.assert_called_once_with(engine="ddgs", query="python", max_results=5, site="python.org")
    assert result == "Search results"


@patch("py_search_helper.mcp.server.search")
def test_search_web_without_site_parameter(mock_search: MagicMock) -> None:
    """Test search_web without site parameter (backward compatibility)."""
    from py_search_helper.mcp.server import search_web

    mock_search.return_value = "Search results"

    # Call the function
    if hasattr(search_web, "fn"):
        result = search_web.fn(engine="ddgs", query="python", max_results=5)
    else:
        result = search_web(engine="ddgs", query="python", max_results=5)

    mock_search.assert_called_once_with(engine="ddgs", query="python", max_results=5, site=None)


@patch("py_search_helper.mcp.server.search")
def test_search_web_ddg_integration(mock_search: MagicMock) -> None:
    """Test search_web_ddg through MCP integration."""
    from py_search_helper.mcp.server import search_web_ddg

    mock_search.return_value = "Search results"

    # Call the function
    if hasattr(search_web_ddg, "fn"):
        result = search_web_ddg.fn(query="python", max_results=5)
    else:
        result = search_web_ddg(query="python", max_results=5)

    mock_search.assert_called_once_with(engine="ddgs", query="python", max_results=5, site=None)
    assert result == "Search results"


@patch("py_search_helper.mcp.server.search")
def test_search_web_ddg_with_site_parameter(mock_search: MagicMock) -> None:
    """Test search_web_ddg with site parameter."""
    from py_search_helper.mcp.server import search_web_ddg

    mock_search.return_value = "Search results"

    # Call the function
    if hasattr(search_web_ddg, "fn"):
        result = search_web_ddg.fn(query="python", max_results=5, site="python.org")
    else:
        result = search_web_ddg(query="python", max_results=5, site="python.org")

    mock_search.assert_called_once_with(engine="ddgs", query="python", max_results=5, site="python.org")
    assert result == "Search results"


@patch("py_search_helper.mcp.server.search")
def test_search_web_ddg_without_site_parameter(mock_search: MagicMock) -> None:
    """Test search_web_ddg without site parameter (backward compatibility)."""
    from py_search_helper.mcp.server import search_web_ddg

    mock_search.return_value = "Search results"

    # Call the function
    if hasattr(search_web_ddg, "fn"):
        result = search_web_ddg.fn(query="python", max_results=5)
    else:
        result = search_web_ddg(query="python", max_results=5)

    mock_search.assert_called_once_with(engine="ddgs", query="python", max_results=5, site=None)


def test_get_engines_real_call() -> None:
    """Test get_engines returns real data."""
    from py_search_helper.mcp.server import get_engines

    # Get the actual function
    fn = get_engines.fn if hasattr(get_engines, "fn") else get_engines
    result = fn()

    assert isinstance(result, list)
    assert all(isinstance(item, tuple) for item in result)
    assert all(len(item) == 2 for item in result)
    # DDGS should be registered by default
    assert any(item[0] == "ddgs" for item in result)


def test_search_web_real_call() -> None:
    """Test search_web returns real data."""
    from py_search_helper.mcp.server import search_web

    # Get the actual function
    fn = search_web.fn if hasattr(search_web, "fn") else search_web
    result = fn(engine="ddgs", query="test", max_results=1)

    assert isinstance(result, str)
    assert len(result) > 0


def test_search_web_ddg_real_call() -> None:
    """Test search_web_ddg returns real data."""
    from py_search_helper.mcp.server import search_web_ddg

    # Get the actual function
    fn = search_web_ddg.fn if hasattr(search_web_ddg, "fn") else search_web_ddg
    result = fn(query="test", max_results=1)

    assert isinstance(result, str)
    assert len(result) > 0


def test_open_page_real_call() -> None:
    """Test open_page returns real data."""
    from py_search_helper.mcp.server import open_page

    # Get the actual function
    fn = open_page.fn if hasattr(open_page, "fn") else open_page
    result = fn(url="https://example.com", max_chars=100)

    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result) <= 103  # 100 + "..." if truncated


@patch("py_search_helper.mcp.server.mcp")
def test_main_calls_mcp_run(mock_mcp: MagicMock) -> None:
    """Test that main() calls mcp.run()."""
    main()

    mock_mcp.run.assert_called_once()


def test_tool_descriptions() -> None:
    """Test that MCP tools have proper descriptions."""
    from py_search_helper.mcp.server import get_engines, open_page, search_web, search_web_ddg

    # FastMCP wraps functions in FunctionTool objects with description attribute
    assert hasattr(get_engines, "description")
    assert "search engines" in get_engines.description.lower()

    assert hasattr(search_web, "description")
    assert "search" in search_web.description.lower()

    assert hasattr(search_web_ddg, "description")
    assert "search" in search_web_ddg.description.lower()

    assert hasattr(open_page, "description")
    assert "url" in open_page.description.lower()


def test_tool_functionality() -> None:
    """Test that MCP tools work correctly."""
    from py_search_helper.mcp.server import get_engines, open_page, search_web, search_web_ddg

    # Test get_engines
    fn = get_engines.fn if hasattr(get_engines, "fn") else get_engines
    engines = fn()
    assert isinstance(engines, list)
    assert len(engines) > 0

    # Test search_web
    fn = search_web.fn if hasattr(search_web, "fn") else search_web
    result = fn(engine="ddgs", query="test", max_results=1)
    assert isinstance(result, str)

    # Test search_web_ddg
    fn = search_web_ddg.fn if hasattr(search_web_ddg, "fn") else search_web_ddg
    result = fn(query="test", max_results=1)
    assert isinstance(result, str)

    # Test open_page
    fn = open_page.fn if hasattr(open_page, "fn") else open_page
    result = fn(url="https://example.com", max_chars=50)
    assert isinstance(result, str)
