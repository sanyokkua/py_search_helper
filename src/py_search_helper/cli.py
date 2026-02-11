"""Command-line interface for web search and content extraction.

This module provides a Typer-based CLI wrapper around py-search-helper's
core functionality. It exposes three commands for listing search engines,
performing web searches, and extracting content from URLs.

Exports:
    app: Typer application instance configured with CLI commands.
    get_engines_cli: List available search engines.
    search_cli: Perform web searches with configurable parameters.
    open_page_cli: Extract text content from web pages.
"""

import typer

from py_search_helper import get_search_engines, open_url, search
from py_search_helper.exceptions import PySearchHelperError

__all__ = ["app", "get_engines_cli", "open_page_cli", "search_cli"]

app = typer.Typer(
    name="py-search-helper",
    help="A CLI for searching the internet and extracting web content.",
    no_args_is_help=True,
)


@app.command(name="get-engines", help="List all available search engines.")
def get_engines_cli() -> None:
    """Display all available search engines and their descriptions.

    Queries the underlying search helper library to retrieve the list
    of supported search engines with their identifiers and descriptions.

    Raises:
        PySearchHelperError: If the search engine registry cannot be accessed.
        typer.Exit: With code 1 if an error occurs during engine retrieval.
    """
    try:
        engines = get_search_engines()
        typer.echo("Available Search Engines:")
        for engine_id, description in engines:
            typer.echo(f"  - {engine_id}: {description}")
    except PySearchHelperError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command(name="search", help="Search the web using a specified engine.")
def search_cli(
    query: str = typer.Argument(..., help="The search query string."),
    engine: str = typer.Option(
        "ddgs", "--engine", "-e", help="The ID of the search engine to use (e.g., 'ddgs', 'wikipedia')."
    ),
    max_results: int = typer.Option(10, "--max-results", "-m", help="Maximum number of results to return."),
    site: str | None = typer.Option(
        None, "--site", "-s", help="Optional domain to restrict the search (e.g., 'docs.python.org')."
    ),
) -> None:
    """Perform a web search using the specified search engine.

    Executes a search query against the configured search engine and
    outputs formatted results to stdout.

    Args:
        query: Search terms to query the engine with. Required.
        engine: Search engine identifier. Defaults to 'ddgs' (DuckDuckGo).
        max_results: Maximum number of results to return. Must be positive.
            Defaults to 10.
        site: Optional domain filter to restrict results to a specific site.
            If None, search is unrestricted.

    Raises:
        PySearchHelperError: If the search engine is unavailable, the query
            fails, or results cannot be retrieved.
        typer.Exit: With code 1 if an error occurs during search execution.
    """
    try:
        results = search(engine=engine, query=query, max_results=max_results, site=site)
        typer.echo(results)
    except PySearchHelperError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command(name="open-page", help="Extract content from a given URL.")
def open_page_cli(
    url: str = typer.Argument(..., help="The URL from which to extract content."),
    max_chars: int | None = typer.Option(
        500, "--max-chars", "-c", help="Maximum number of characters to return. Use 0 for unlimited."
    ),
) -> None:
    """Extract and display text content from a web page.

    Retrieves the main textual content from the specified URL and outputs
    it to stdout, optionally truncated to a character limit.

    Args:
        url: Target URL to extract content from. Must be a valid HTTP/HTTPS URL.
        max_chars: Maximum character count for output. Value of 0 indicates
            unlimited output. Defaults to 500 characters. If None, uses default.

    Raises:
        PySearchHelperError: If the URL cannot be fetched, parsed, or content
            extraction fails.
        typer.Exit: With code 1 if an error occurs during content extraction.
    """
    try:
        # Interpret 0 as None for unlimited characters
        chars_limit = max_chars if max_chars != 0 else None
        content = open_url(url=url, max_chars=chars_limit)
        typer.echo(content)
    except PySearchHelperError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
