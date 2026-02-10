"""Abstract search provider protocol."""

from typing import Protocol

from py_search_helper.models import EngineInfo, SearchResult


class SearchProvider(Protocol):
    """Protocol for search provider implementations.

    All providers must implement get_info() and search().

    Note: The search() method receives query strings that may include
    search operators (e.g., "site:example.com"). Providers should pass
    these queries directly to their underlying search engines.
    """

    def get_info(self) -> EngineInfo:
        """Return provider metadata.

        Returns:
            EngineInfo with provider details
        """
        ...

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Perform search and return results.

        Args:
            query: Search query string (may include operators like "site:")
            max_results: Maximum number of results to return (default: 10)

        Returns:
            List of SearchResult objects

        Raises:
            SearchProviderError: If search fails

        Note:
            Queries may contain search operators added by the API layer.
            Pass the query string unchanged to the underlying search engine.
        """
        ...
