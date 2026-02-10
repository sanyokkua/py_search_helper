"""Exception hierarchy for py-search-helper."""


class PySearchHelperError(Exception):
    """Base exception for all py-search-helper errors."""


class EngineError(PySearchHelperError):
    """Base exception for engine operations."""


class EngineNotFoundError(EngineError):
    """Raised when specified engine is not registered."""


class SearchError(PySearchHelperError):
    """Base exception for search operations."""


class SearchProviderError(SearchError):
    """Raised when search provider fails."""


class URLError(PySearchHelperError):
    """Base exception for URL operations."""


class URLNotFoundError(URLError):
    """Raised when URL returns 404."""


class URLTimeoutError(URLError):
    """Raised when URL request times out."""
