"""py-search-helper: Python library for web search and content extraction.

This library provides a simple interface for:
- Discovering available search engines
- Searching across multiple providers (web, documentation, wikis)
- Extracting clean content from URLs

Example:
    >>> from py_search_helper import get_search_engines, search, open_url
    >>>
    >>> # Discover engines
    >>> engines = get_search_engines()
    >>> print(engines)
    [('ddgs', 'General web search'), ...]
    >>>
    >>> # Search
    >>> results = search(engine="ddgs", query="python asyncio", max_results=5)
    >>> print(results)
    # Search Results...
    >>>
    >>> # Open URL
    >>> content = open_url("https://example.com", max_chars=1000)
    >>> print(content)
    # Content from https://example.com...
"""

# Bootstrap providers (must be imported before API functions are used)
from py_search_helper import _bootstrap  # noqa: F401
from py_search_helper.api import get_search_engines, open_url, search
from py_search_helper.exceptions import (
    EngineError,
    EngineNotFoundError,
    PySearchHelperError,
    SearchError,
    SearchProviderError,
    URLError,
    URLNotFoundError,
    URLTimeoutError,
)
from py_search_helper.models import EngineInfo, SearchResult

__version__ = "0.1.0"

__all__ = [
    "EngineError",
    # Models
    "EngineInfo",
    "EngineNotFoundError",
    # Exceptions
    "PySearchHelperError",
    "SearchError",
    "SearchProviderError",
    "SearchResult",
    "URLError",
    "URLNotFoundError",
    "URLTimeoutError",
    # Core API
    "get_search_engines",
    "open_url",
    "search",
]
