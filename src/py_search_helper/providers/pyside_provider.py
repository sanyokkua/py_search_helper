"""PySide documentation provider implementation."""

from py_search_helper.models import EngineInfo, SearchResult
from py_search_helper.providers.ddgs_provider import DDGSProvider


class PySideProvider:
    """PySide documentation search provider (URL-based).

    Searches Qt for Python official documentation.
    """

    def get_info(self) -> EngineInfo:
        """Return provider metadata.

        Returns:
            EngineInfo with PySide provider details
        """
        return EngineInfo(
            id="pyside",
            name="PySide Documentation",
            description="Qt for Python official documentation",
            scope="PySide6/Qt for Python API documentation and guides",
            base_url="https://doc.qt.io/qtforpython-6",
        )

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search PySide documentation using DuckDuckGo site filter.

        Args:
            query: Search query string (may already include "site:" operator)
            max_results: Maximum number of results to return (default: 10)

        Returns:
            List of SearchResult objects from PySide6 documentation

        Raises:
            SearchProviderError: If search fails

        Note:
            If the query doesn't contain a "site:" operator, this method
            automatically adds "site:doc.qt.io/qtforpython-6" to restrict
            results to PySide6 official documentation. If a site filter is
            already present (e.g., from the API's site parameter), it is
            preserved, allowing users to search other Qt documentation sites.
        """
        # Only add default site filter if query doesn't already have one
        if "site:" not in query:
            query = f"{query} site:doc.qt.io/qtforpython-6"

        # Use DDGS provider to search
        ddgs = DDGSProvider()
        return ddgs.search(query, max_results=max_results)
