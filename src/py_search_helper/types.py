"""Type definitions and type aliases."""

from typing import Protocol

# Type aliases
EngineID = str
EngineTuple = tuple[EngineID, str]  # (id, description)
MarkdownContent = str

# Re-export Protocol for provider implementations
__all__ = [
    "EngineID",
    "EngineTuple",
    "MarkdownContent",
    "Protocol",
]
