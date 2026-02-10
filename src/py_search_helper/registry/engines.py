"""Engine registry for managing search providers."""

from typing import Protocol

from py_search_helper.exceptions import EngineError
from py_search_helper.models import EngineInfo, SearchResult


class SearchProvider(Protocol):
    """Search provider protocol.

    Note: The search() method receives query strings that may include
    search operators (e.g., "site:example.com"). Providers should pass
    these queries directly to their underlying search engines.
    """

    def get_info(self) -> EngineInfo:
        """Return provider metadata."""
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


class EngineRegistry:
    """Registry for search engine providers."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._providers: dict[str, SearchProvider] = {}

    def register(self, provider: SearchProvider) -> None:
        """Register a search provider.

        Args:
            provider: SearchProvider instance to register

        Raises:
            EngineError: If provider with same ID already registered
        """
        info = provider.get_info()
        if info.id in self._providers:
            raise EngineError(f"Provider with ID '{info.id}' already registered")
        self._providers[info.id] = provider

    def get_provider(self, engine_id: str) -> SearchProvider | None:
        """Get provider by engine ID.

        Args:
            engine_id: Engine identifier

        Returns:
            SearchProvider instance or None if not found
        """
        return self._providers.get(engine_id)

    def get_all_engines(self) -> list[tuple[str, str]]:
        """Get all registered engines.

        Returns:
            List of (engine_id, description) tuples
        """
        result = []
        for provider in self._providers.values():
            info = provider.get_info()
            result.append((info.id, info.description))
        return result

    def get_engine_info(self, engine_id: str) -> EngineInfo | None:
        """Get full engine information.

        Args:
            engine_id: Engine identifier

        Returns:
            EngineInfo or None if not found
        """
        provider = self.get_provider(engine_id)
        if provider is None:
            return None
        return provider.get_info()


# Global registry instance
_registry = EngineRegistry()


def get_registry() -> EngineRegistry:
    """Get global registry instance.

    Returns:
        The global EngineRegistry instance
    """
    return _registry
