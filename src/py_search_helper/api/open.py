"""Open URL API."""

from py_search_helper.exceptions import URLError, URLNotFoundError, URLTimeoutError
from py_search_helper.extractors import extract_content
from py_search_helper.types import MarkdownContent


def open_url(url: str, max_chars: int | None = 500) -> MarkdownContent:
    """Open a URL and return its content as Markdown.

    Args:
        url: URL to open (must start with http:// or https://)
        max_chars: Maximum characters to return (default: 500, None for unlimited)

    Returns:
        Markdown-formatted page content (truncated if max_chars specified)

    Raises:
        ValueError: If URL is empty or invalid format, or max_chars is invalid
        URLNotFoundError: If URL returns 404
        URLTimeoutError: If request times out
        URLError: If extraction fails

    Example:
        >>> content = open_url("https://example.com", max_chars=200)
        >>> print(content)
        # Example Domain

        This domain is for use in illustrative examples...
    """
    # Validation
    if not url:
        raise ValueError("URL cannot be empty")
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    if max_chars is not None and max_chars < 1:
        raise ValueError("max_chars must be at least 1 or None")

    # Extract content from URL
    try:
        content = extract_content(url)

        # Truncate if max_chars specified
        if max_chars is not None and len(content) > max_chars:
            return content[:max_chars] + "..."

        return content
    except URLNotFoundError:
        raise
    except URLTimeoutError:
        raise
    except URLError:
        raise
    except Exception as e:
        raise URLError(f"Failed to open URL {url}: {e}") from e
