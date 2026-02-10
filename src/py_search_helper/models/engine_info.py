"""Engine information model."""

from dataclasses import dataclass


@dataclass
class EngineInfo:
    """Metadata about a search engine/provider."""

    id: str
    name: str
    description: str
    scope: str
    base_url: str
