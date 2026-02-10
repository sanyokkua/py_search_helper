"""API module for py-search-helper."""

from py_search_helper.api.engines import get_search_engines
from py_search_helper.api.open import open_url
from py_search_helper.api.search import search

__all__ = ["get_search_engines", "open_url", "search"]
