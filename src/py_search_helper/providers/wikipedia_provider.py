"""Wikipedia search provider implementation."""

from py_search_helper.models import EngineInfo, SearchResult
from py_search_helper.providers.ddgs_provider import DDGSProvider


class WikipediaProvider:
    """Wikipedia search provider (URL-based).

    Searches Wikipedia encyclopedia.
    """

    def get_info(self) -> EngineInfo:
        """Return provider metadata.

        Returns:
            EngineInfo with Wikipedia provider details
        """
        return EngineInfo(
            id="wikipedia",
            name="Wikipedia",
            description="Wikipedia encyclopedia",
            scope="Wikipedia articles and encyclopedia content",
            base_url="https://en.wikipedia.org",
        )

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search Wikipedia using DuckDuckGo site filter.

        Args:
            query: Search query string (may already include "site:" operator)
            max_results: Maximum number of results to return (default: 10)

        Returns:
            List of SearchResult objects from Wikipedia

        Raises:
            SearchProviderError: If search fails

        Note:
            If the query doesn't contain a "site:" operator, this method
            automatically adds "site:wikipedia.org" to restrict results to
            Wikipedia content across all language editions. If a site filter is
            already present, it is preserved, allowing users to search specific
            language editions.
        """
        # Only add default site filter if query doesn't already have one
        if "site:" not in query:
            query = f"{query} site:wikipedia.org"

        # Use DDGS provider to search
        ddgs = DDGSProvider()
        return ddgs.search(query, max_results=max_results)
