"""Bootstrap registry with default providers."""

from py_search_helper.providers import DDGSProvider, PySideProvider, WikipediaProvider
from py_search_helper.registry import get_registry


def bootstrap_providers() -> None:
    """Register default search providers.

    Registers DDGS, PySide, and Wikipedia providers (all working implementations).
    """
    registry = get_registry()

    # Register DDGS provider (working implementation)
    registry.register(DDGSProvider())

    # Register PySide provider (working implementation)
    registry.register(PySideProvider())

    # Register Wikipedia provider (working implementation)
    registry.register(WikipediaProvider())


# Auto-bootstrap on module import
bootstrap_providers()
