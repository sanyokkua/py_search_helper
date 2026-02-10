"""MCP server for py-search-helper.

Exposes three tools:
- get_search_engines: List available search engines
- search: Search using a specific engine
- open_url: Extract content from a URL
"""

from fastmcp import FastMCP

from py_search_helper import get_search_engines, open_url, search

# Create MCP server
mcp = FastMCP("py-search-helper")


@mcp.tool()
def get_engines() -> list[tuple[str, str]]:
    """Get list of available search engines.

    Returns:
        List of (engine_id, description) tuples showing all available search sources.

    Example:
        [("ddgs", "General web search"), ("pyside", "Qt docs"), ...]
    """
    return get_search_engines()


@mcp.tool()
def search_web(engine: str, query: str, max_results: int = 10, site: str | None = None) -> str:
    """Search using specified engine.

    Args:
        engine: Engine ID (use get_engines() to see available engines)
        query: Search query string
        max_results: Maximum number of results to return (default: 10)
        site: Optional domain to restrict search (e.g., "python.org", "github.com")

    Returns:
        Markdown-formatted search results with titles, URLs, and descriptions.

    Examples:
        # General search
        search_web(engine="ddgs", query="python asyncio tutorial", max_results=5)

        # Search specific domain
        search_web(engine="ddgs", query="asyncio", site="docs.python.org", max_results=5)

        # Search Stack Overflow
        search_web(engine="ddgs", query="python threading", site="stackoverflow.com")
    """
    return search(engine=engine, query=query, max_results=max_results, site=site)


@mcp.tool()
def open_page(url: str, max_chars: int | None = 500) -> str:
    """Open a URL and extract its content.

    Args:
        url: Web URL to open (must start with http:// or https://)
        max_chars: Maximum characters to return (default: 500, None for unlimited)

    Returns:
        Markdown-formatted page content. Truncated if max_chars specified.

    Example:
        open_page(url="https://docs.python.org/3/library/asyncio.html", max_chars=1000)
    """
    return open_url(url=url, max_chars=max_chars)


def main() -> None:
    """Main entry point for MCP server."""
    mcp.run()


# Main entry point
if __name__ == "__main__":
    main()
