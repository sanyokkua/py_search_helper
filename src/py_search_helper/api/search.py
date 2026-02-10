"""Search API."""

from py_search_helper.exceptions import EngineNotFoundError, SearchProviderError
from py_search_helper.models import SearchResult
from py_search_helper.registry import get_registry
from py_search_helper.types import MarkdownContent


def search(
    engine: str,
    query: str,
    max_results: int = 10,
    *,
    site: str | None = None,
) -> MarkdownContent:
    """Search using specified engine.

    Args:
        engine: Engine ID (e.g., "ddgs", "pyside")
        query: Search query string
        max_results: Maximum number of results to return (default: 10)
        site: Optional domain to restrict search (e.g., "python.org")

    Returns:
        Markdown-formatted search results

    Raises:
        ValueError: If query/engine is empty, max_results invalid, or site is empty
        EngineNotFoundError: If engine is not registered
        SearchProviderError: If search provider fails

    Example:
        >>> results = search(engine="ddgs", query="python asyncio", max_results=5)
        >>> print(results)
        # Search Results

        ## Result 1
        **Title:** Python asyncio documentation
        **URL:** https://docs.python.org/3/library/asyncio.html
        **Description:** asyncio is a library to write concurrent code...

        >>> results = search(engine="ddgs", query="asyncio", site="docs.python.org")
    """
    # Validation
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not engine:
        raise ValueError("Engine ID cannot be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    if site is not None and not site.strip():
        raise ValueError("Site parameter cannot be empty or whitespace-only")

    # Build effective query with site filter if provided
    effective_query = _build_query(query, site)

    # Get provider from registry
    registry = get_registry()
    provider = registry.get_provider(engine)
    if provider is None:
        raise EngineNotFoundError(f"Engine '{engine}' not found")

    # Execute search
    try:
        results = provider.search(effective_query, max_results=max_results)
        return _format_results_as_markdown(results, query, engine, site)
    except SearchProviderError:
        raise
    except Exception as e:
        raise SearchProviderError(f"Search failed for engine '{engine}': {e}") from e


def _build_query(query: str, site: str | None) -> str:
    """Build effective query string with site filter if provided.

    Args:
        query: Base search query
        site: Optional domain filter

    Returns:
        Query string with site filter appended if site provided

    Example:
        >>> _build_query("python", None)
        'python'
        >>> _build_query("python", "python.org")
        'python site:python.org'
    """
    if site is None:
        return query
    return f"{query} site:{site}"


def _format_results_as_markdown(
    results: list[SearchResult],
    query: str,
    engine: str,
    site: str | None = None,
) -> MarkdownContent:
    """Format search results as Markdown.

    Args:
        results: List of SearchResult objects
        query: Original search query
        engine: Engine ID used for search
        site: Optional site filter used

    Returns:
        Markdown-formatted search results
    """
    site_info = f" (site: {site})" if site else ""

    if not results:
        return f'# Search Results for "{query}"{site_info} (engine: {engine})\n\nNo results found.'

    lines = [f'# Search Results for "{query}"{site_info} (engine: {engine})\n']

    for i, result in enumerate(results, 1):
        lines.append(f"## Result {i}")
        lines.append(f"**Title:** {result.title}")
        lines.append(f"**URL:** {result.url}")
        lines.append(f"**Description:** {result.description}")
        lines.append("")

    return "\n".join(lines)
