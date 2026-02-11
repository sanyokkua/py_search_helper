# Provider Implementation Guide

## Overview

Providers implement the SearchProvider protocol to enable searching through different sources. Each provider must implement two methods: `get_info()` and `search()`.

## SearchProvider Protocol

```python
from py_search_helper.models import EngineInfo, SearchResult

class SearchProvider(Protocol):
    """Protocol for search provider implementations."""

    def get_info(self) -> EngineInfo:
        """Return provider metadata."""
        ...

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Perform search and return results."""
        ...
```

## Implementation Steps

### Step 1: Create Provider File

Create `src/py_search_helper/providers/custom_provider.py`

### Step 2: Implement Protocol

```python
from py_search_helper.models import EngineInfo, SearchResult
from py_search_helper.exceptions import SearchProviderError

class CustomProvider:
    """Custom search provider."""

    def get_info(self) -> EngineInfo:
        """Return provider metadata."""
        return EngineInfo(
            id="custom",
            name="Custom Search",
            description="Search custom website",
            scope="Specific domain content",
            base_url="https://example.com"
        )

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Perform search and return results.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results

        Raises:
            SearchProviderError: If search operation fails
        """
        if not query.strip():
            raise SearchProviderError("Query cannot be empty")

        # Perform search logic here
        results = self._perform_search(query, max_results)

        return [
            SearchResult(
                title=result["title"],
                url=result["url"],
                description=result["description"]
            )
            for result in results
        ]

    def _perform_search(self, query: str, max_results: int) -> list[dict]:
        """Internal search implementation."""
        # Implementation details
        pass
```

### Step 3: Register Provider

Edit `src/py_search_helper/_bootstrap.py`:

```python
from py_search_helper.providers.custom_provider import CustomProvider
from py_search_helper.providers.ddgs_provider import DDGSProvider
from py_search_helper.providers.pyside_provider import PySideProvider
from py_search_helper.providers.wikipedia_provider import WikipediaProvider
from py_search_helper.registry.engines import get_registry

def bootstrap_providers() -> None:
    """Register all available providers."""
    registry = get_registry()

    # Register existing providers
    registry.register(DDGSProvider())
    registry.register(PySideProvider())
    registry.register(WikipediaProvider())

    # Register your new provider
    registry.register(CustomProvider())
```

### Step 4: Write Tests

Create `tests/test_providers/test_custom_provider.py`:

```python
import pytest
from py_search_helper.providers.custom_provider import CustomProvider
from py_search_helper.models import EngineInfo, SearchResult
from py_search_helper.exceptions import SearchProviderError

def test_custom_provider_get_info():
    """Test that provider returns correct metadata."""
    provider = CustomProvider()
    info = provider.get_info()

    assert isinstance(info, EngineInfo)
    assert info.id == "custom"
    assert info.name == "Custom Search"
    assert info.base_url == "https://example.com"

def test_custom_provider_search_returns_results():
    """Test that search returns list of results."""
    provider = CustomProvider()
    results = provider.search("test query", max_results=5)

    assert isinstance(results, list)
    assert len(results) <= 5
    assert all(isinstance(r, SearchResult) for r in results)

def test_custom_provider_search_validates_empty_query():
    """Test that empty query raises error."""
    provider = CustomProvider()

    with pytest.raises(SearchProviderError, match="Query cannot be empty"):
        provider.search("", max_results=5)

def test_custom_provider_search_validates_whitespace_query():
    """Test that whitespace-only query raises error."""
    provider = CustomProvider()

    with pytest.raises(SearchProviderError, match="Query cannot be empty"):
        provider.search("   ", max_results=5)
```

## Provider Types

### API-Based Provider

Uses external library or API. Returns structured results directly.

**Example: DDGS Provider**

```python
from ddgs import DDGS
from ddgs.exceptions import RatelimitException, TimeoutException
import time

class DDGSProvider:
    """DuckDuckGo search provider using DDGS library."""

    def __init__(self) -> None:
        """Initialize provider with rate limiting."""
        self._last_request_time = 0.0
        self._min_interval = 3.0  # 3 seconds between requests

    def get_info(self) -> EngineInfo:
        """Return DDGS provider metadata."""
        return EngineInfo(
            id="ddgs",
            name="DuckDuckGo",
            description="DuckDuckGo web search",
            scope="General web content",
            base_url="https://duckduckgo.com"
        )

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Perform search using DDGS library."""
        if not query.strip():
            raise SearchProviderError("Query cannot be empty")

        # Rate limiting
        self._rate_limit()

        # Perform search with retry on rate limit
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
        except RatelimitException:
            time.sleep(60)  # Wait 1 minute
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
        except TimeoutException as e:
            raise SearchProviderError(f"Search timeout: {e}") from e

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                description=r.get("body", "")
            )
            for r in results
        ]

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
```

### DDGS-Delegated Provider

Some providers leverage the DDGS provider as a backend with automatic site filtering. This pattern is simple, maintainable, and inherits DDGS reliability.

**Implementation:**
```python
from py_search_helper.models import EngineInfo, SearchResult
from py_search_helper.providers.ddgs_provider import DDGSProvider

class PySideProvider:
    """PySide6 documentation search provider using DDGS delegation."""

    def __init__(self) -> None:
        """Initialize provider with DDGS backend."""
        self._ddgs_provider = DDGSProvider()

    def get_info(self) -> EngineInfo:
        """Return PySide provider metadata."""
        return EngineInfo(
            id="pyside",
            name="PySide6 Documentation",
            description="PySide6 documentation search",
            scope="PySide6 official documentation",
            base_url="https://doc.qt.io/qtforpython-6"
        )

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search PySide6 documentation using DDGS with site filter.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results from PySide6 documentation
        """
        site_filter = "site:doc.qt.io/qtforpython-6"
        if "site:" not in query:
            query = f"{query} {site_filter}"
        return self._ddgs_provider.search(query, max_results)
```

**Benefits:**
- Inherits DDGS rate limiting and error handling
- No additional dependencies required
- Simple, maintainable implementation (~15 lines)
- Domain-scoped search without custom HTML parsing
- Production-ready reliability

**Current Implementations:** PySide, Wikipedia

### URL-Based Provider with HTML Parsing

Constructs search URL, fetches HTML page, parses results. This pattern is more complex but offers direct control.

**Example: URL-Based Provider Template**

```python
import httpx
from bs4 import BeautifulSoup

class URLBasedProvider:
    """URL-based search provider template."""

    def get_info(self) -> EngineInfo:
        """Return provider metadata."""
        return EngineInfo(
            id="urlbased",
            name="URL-Based Search",
            description="Search via URL construction",
            scope="Website content",
            base_url="https://example.com"
        )

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Perform search by fetching and parsing HTML."""
        if not query.strip():
            raise SearchProviderError("Query cannot be empty")

        # 1. Construct search URL
        url = self._build_search_url(query)

        # 2. Fetch page
        try:
            response = httpx.get(url, timeout=10, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SearchProviderError(f"HTTP error: {e}") from e

        # 3. Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        result_elements = soup.select(".search-result")

        # 4. Extract and return results
        results = []
        for elem in result_elements[:max_results]:
            try:
                title = elem.select_one(".title").get_text(strip=True)
                url = elem.select_one("a")["href"]
                description = elem.select_one(".description").get_text(strip=True)

                results.append(SearchResult(
                    title=title,
                    url=url,
                    description=description
                ))
            except (AttributeError, KeyError):
                continue  # Skip malformed results

        return results

    def _build_search_url(self, query: str) -> str:
        """Construct search URL with query parameter."""
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        return f"https://example.com/search?q={encoded_query}"
```

## Rate Limiting

For providers with rate limits, implement rate limiting logic:

```python
import time

class RateLimitedProvider:
    """Provider with rate limiting."""

    def __init__(self, requests_per_minute: int = 6) -> None:
        """Initialize with rate limit."""
        self._min_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Perform rate-limited search."""
        self._rate_limit()
        # ... perform search
```

## Error Handling

Providers should handle errors consistently:

```python
from py_search_helper.exceptions import SearchProviderError

def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
    """Perform search with error handling."""
    # Validate input
    if not query.strip():
        raise SearchProviderError("Query cannot be empty")

    # Handle external errors
    try:
        results = self._external_search(query, max_results)
    except ExternalAPIError as e:
        raise SearchProviderError(f"External API error: {e}") from e
    except TimeoutError as e:
        raise SearchProviderError(f"Search timeout: {e}") from e

    # Validate output
    if not results:
        return []

    return results
```

## Connection Flow

1. **Provider Creation**: Provider class instantiated (e.g., `DDGSProvider()`)
2. **Registration**: Provider registered in `_bootstrap.py` via `registry.register()`
3. **Bootstrap Execution**: Bootstrap runs automatically on module import
4. **Registry Storage**: Registry stores provider instance by engine ID
5. **API Lookup**: API retrieves provider via `registry.get_provider(engine_id)`
6. **Search Execution**: Provider's `search()` method called with query
7. **Result Return**: Formatted results returned to user

## Testing Strategy

### Unit Tests

Test provider in isolation:

```python
def test_provider_get_info():
    """Test metadata retrieval."""
    provider = CustomProvider()
    info = provider.get_info()
    assert info.id == "custom"

def test_provider_search_validation():
    """Test input validation."""
    provider = CustomProvider()
    with pytest.raises(SearchProviderError):
        provider.search("")
```

### Integration Tests

Test provider registration:

```python
def test_provider_registration():
    """Test provider can be registered and retrieved."""
    from py_search_helper.registry.engines import get_registry

    provider = CustomProvider()
    registry = get_registry()
    registry.register(provider)

    retrieved = registry.get_provider("custom")
    assert retrieved is provider
```

### Mock Tests

Mock external dependencies:

```python
from unittest.mock import Mock, patch

@patch("custom_provider.external_api")
def test_provider_search_with_mock(mock_api):
    """Test search with mocked external API."""
    mock_api.search.return_value = [
        {"title": "Result 1", "url": "https://example.com/1", "description": "Desc 1"}
    ]

    provider = CustomProvider()
    results = provider.search("test", max_results=1)

    assert len(results) == 1
    assert results[0].title == "Result 1"
    mock_api.search.assert_called_once()
```

## Best Practices

1. **Input Validation**: Always validate query is non-empty
2. **Error Handling**: Wrap external calls in try-except blocks
3. **Rate Limiting**: Implement delays for rate-limited APIs
4. **Timeouts**: Set reasonable timeouts for HTTP requests
5. **Logging**: Log errors and important events
6. **Documentation**: Document provider metadata, parameters, and exceptions
7. **Testing**: Write unit, integration, and mock tests
8. **Type Hints**: Use complete type hints for all methods

## Provider Status

| Provider  | Type           | Status     |
| --------- | -------------- | ---------- |
| DDGS      | API-based      | Production |
| PySide    | DDGS-delegated | Production |
| Wikipedia | DDGS-delegated | Production |

## Examples

See existing provider implementations:
- `src/py_search_helper/providers/ddgs_provider.py` - API-based with rate limiting
- `src/py_search_helper/providers/pyside_provider.py` - DDGS-delegated (production)
- `src/py_search_helper/providers/wikipedia_provider.py` - DDGS-delegated (production)
