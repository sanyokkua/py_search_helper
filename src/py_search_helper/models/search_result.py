"""Search result model."""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    description: str
