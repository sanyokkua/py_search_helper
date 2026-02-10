"""DDGS (DuckDuckGo Search) provider implementation."""

import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from py_search_helper.exceptions import SearchProviderError
from py_search_helper.models import EngineInfo, SearchResult


class DDGSProvider:
    """DDGS search provider (API-based).

    Uses the ddgs library to perform web searches with automatic rate limiting.
    Implements retry logic for rate limit errors.
    """

    def __init__(self, requests_per_minute: int = 6) -> None:
        """Initialize DDGS provider.

        Args:
            requests_per_minute: Rate limit for requests (default: 6)
        """
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

    def get_info(self) -> EngineInfo:
        """Return provider metadata.

        Returns:
            EngineInfo with DDGS provider details
        """
        return EngineInfo(
            id="ddgs",
            name="DuckDuckGo Search",
            description="General web search (DuckDuckGo)",
            scope="General web content from multiple sources",
            base_url="https://duckduckgo.com",
        )

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Perform search using DDGS.

        Args:
            query: Search query string (may include operators like "site:example.com")
            max_results: Maximum results to return (max 30 recommended)

        Returns:
            List of SearchResult objects

        Raises:
            SearchProviderError: If search fails after retry

        Note:
            Query may include DDGS search operators such as:
            - site:domain.com - Restrict to specific domain
            - filetype:pdf - Restrict to file type
            - intitle:keyword - Search in title only
        """
        # Enforce rate limiting
        self._rate_limit()

        try:
            # Perform search (limit to 30 to reduce rate limit risk)
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=min(max_results, 30))

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("href", ""),
                        description=result.get("body", ""),
                    )
                )

            return search_results

        except RatelimitException:
            # Wait and retry once
            time.sleep(60)
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(query, max_results=min(max_results, 30))
                return [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        description=r.get("body", ""),
                    )
                    for r in results
                ]
            except Exception as retry_error:
                raise SearchProviderError(f"DDGS rate limit retry failed: {retry_error}") from retry_error

        except TimeoutException as e:
            raise SearchProviderError(f"DDGS search timeout: {e}") from e

        except DDGSException as e:
            raise SearchProviderError(f"DDGS search failed: {e}") from e

        except Exception as e:
            raise SearchProviderError(f"Unexpected error during DDGS search: {e}") from e

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
