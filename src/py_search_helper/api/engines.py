"""Get available search engines API."""

from py_search_helper.exceptions import EngineError
from py_search_helper.registry import get_registry
from py_search_helper.types import EngineTuple


def get_search_engines() -> list[EngineTuple]:
    """Get list of available search engines.

    Returns:
        List of (engine_id, description) tuples

    Raises:
        EngineError: If engine registry fails

    Example:
        >>> engines = get_search_engines()
        >>> for engine_id, desc in engines:
        ...     print(f"{engine_id}: {desc}")
        ddgs: General web search (DuckDuckGo)
        pyside: Qt for Python official documentation
    """
    try:
        registry = get_registry()
        return registry.get_all_engines()
    except Exception as e:
        raise EngineError(f"Failed to retrieve engines: {e}") from e
