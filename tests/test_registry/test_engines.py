"""Tests for engine registry."""

import pytest

from py_search_helper.exceptions import EngineError
from py_search_helper.models import EngineInfo, SearchResult
from py_search_helper.registry import EngineRegistry, get_registry


class MockProvider:
    """Mock search provider for testing."""

    def __init__(self, provider_id: str = "mock") -> None:
        """Initialize mock provider with custom ID."""
        self.provider_id = provider_id

    def get_info(self) -> EngineInfo:
        """Return mock engine info."""
        return EngineInfo(
            id=self.provider_id,
            name=f"Mock Provider {self.provider_id}",
            description=f"Mock search provider ({self.provider_id})",
            scope="Test scope",
            base_url=f"https://{self.provider_id}.test.com",
        )

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Return mock search results."""
        return [
            SearchResult(
                title=f"Result {i} for {query}",
                url=f"https://{self.provider_id}.test.com/result{i}",
                description=f"Mock result {i}",
            )
            for i in range(1, min(max_results, 5) + 1)
        ]


@pytest.fixture
def registry() -> EngineRegistry:
    """Provide a fresh registry instance for each test."""
    return EngineRegistry()


@pytest.fixture
def mock_provider() -> MockProvider:
    """Provide a mock provider for testing."""
    return MockProvider("test")


def test_registry_initialization(registry: EngineRegistry) -> None:
    """Test that registry initializes with empty provider dict."""
    assert isinstance(registry, EngineRegistry)
    assert registry.get_all_engines() == []


def test_get_global_registry() -> None:
    """Test that get_registry returns a registry instance."""
    registry = get_registry()
    assert isinstance(registry, EngineRegistry)


def test_register_provider(registry: EngineRegistry, mock_provider: MockProvider) -> None:
    """Test registering a provider."""
    registry.register(mock_provider)
    engines = registry.get_all_engines()

    assert len(engines) == 1
    assert engines[0] == ("test", "Mock search provider (test)")


def test_register_duplicate_provider_raises_error(registry: EngineRegistry, mock_provider: MockProvider) -> None:
    """Test that registering duplicate provider raises EngineError."""
    registry.register(mock_provider)

    with pytest.raises(EngineError, match="Provider with ID 'test' already registered"):
        registry.register(MockProvider("test"))


def test_get_provider_by_id(registry: EngineRegistry, mock_provider: MockProvider) -> None:
    """Test retrieving provider by engine ID."""
    registry.register(mock_provider)
    provider = registry.get_provider("test")

    assert provider is not None
    assert provider.get_info().id == "test"


def test_get_provider_nonexistent_returns_none(registry: EngineRegistry) -> None:
    """Test that getting non-existent provider returns None."""
    provider = registry.get_provider("nonexistent")
    assert provider is None


def test_get_all_engines(registry: EngineRegistry) -> None:
    """Test getting all registered engines."""
    registry.register(MockProvider("engine1"))
    registry.register(MockProvider("engine2"))
    registry.register(MockProvider("engine3"))

    engines = registry.get_all_engines()

    assert len(engines) == 3
    engine_ids = [e[0] for e in engines]
    assert "engine1" in engine_ids
    assert "engine2" in engine_ids
    assert "engine3" in engine_ids


def test_get_engine_info(registry: EngineRegistry, mock_provider: MockProvider) -> None:
    """Test getting full engine information."""
    registry.register(mock_provider)
    info = registry.get_engine_info("test")

    assert info is not None
    assert info.id == "test"
    assert info.name == "Mock Provider test"
    assert info.description == "Mock search provider (test)"
    assert info.scope == "Test scope"
    assert info.base_url == "https://test.test.com"


def test_get_engine_info_nonexistent_returns_none(registry: EngineRegistry) -> None:
    """Test that getting info for non-existent engine returns None."""
    info = registry.get_engine_info("nonexistent")
    assert info is None


def test_provider_search_works(registry: EngineRegistry, mock_provider: MockProvider) -> None:
    """Test that registered provider's search method works."""
    registry.register(mock_provider)
    provider = registry.get_provider("test")

    assert provider is not None
    results = provider.search("test query", max_results=3)

    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].title == "Result 1 for test query"
    assert results[0].url == "https://test.test.com/result1"


def test_multiple_providers_independent(registry: EngineRegistry) -> None:
    """Test that multiple providers work independently."""
    registry.register(MockProvider("provider1"))
    registry.register(MockProvider("provider2"))

    provider1 = registry.get_provider("provider1")
    provider2 = registry.get_provider("provider2")

    assert provider1 is not None
    assert provider2 is not None

    info1 = provider1.get_info()
    info2 = provider2.get_info()

    assert info1.id == "provider1"
    assert info2.id == "provider2"
    assert info1.base_url == "https://provider1.test.com"
    assert info2.base_url == "https://provider2.test.com"
