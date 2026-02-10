"""Content extraction from URLs."""

from typing import Any

from py_web_text_extractor import create_extractor_service
from py_web_text_extractor.exception.exceptions import TextExtractionError, UrlIsNotValidException

from py_search_helper.exceptions import URLError, URLNotFoundError, URLTimeoutError

# Module-level service instance (reuse across calls)
_extractor_service: Any = None


def get_extractor_service() -> Any:
    """Get or create extractor service instance.

    Returns:
        The py-web-text-extractor service instance (singleton)
    """
    global _extractor_service  # noqa: PLW0603
    if _extractor_service is None:
        _extractor_service = create_extractor_service()
    return _extractor_service


def extract_content(url: str) -> str:
    """Extract content from URL as Markdown.

    Uses py-web-text-extractor with dual-extractor fallback strategy
    (markitdown → trafilatura).

    Args:
        url: URL to extract content from (must start with http:// or https://)

    Returns:
        Markdown-formatted content

    Raises:
        URLError: If URL is invalid or extraction fails
        URLNotFoundError: If URL returns 404
        URLTimeoutError: If request times out
    """
    service = get_extractor_service()

    try:
        # Use strict mode - raises exceptions on failure
        content: str = service.extract_text_from_page(url)
        return content

    except UrlIsNotValidException as e:
        raise URLError(f"Invalid URL: {e}") from e

    except TextExtractionError as e:
        # Check if it's a 404 or timeout
        error_msg = str(e).lower()

        if "404" in error_msg or "not found" in error_msg:
            raise URLNotFoundError(f"URL not found: {url}") from e

        if "timeout" in error_msg or "timed out" in error_msg:
            raise URLTimeoutError(f"Request timed out for {url}") from e

        raise URLError(f"Content extraction failed for {url}: {e}") from e

    except Exception as e:
        # Catch any unexpected errors
        raise URLError(f"Unexpected error extracting content from {url}: {e}") from e


def extract_content_safe(url: str) -> str:
    """Extract content from URL, returning empty string on failure.

    Safe mode version of extract_content() that never raises exceptions.
    Useful for batch processing where you want to continue on errors.

    Args:
        url: URL to extract content from

    Returns:
        Markdown-formatted content, or empty string if extraction fails
    """
    service = get_extractor_service()

    try:
        # Use safe mode - returns empty string on failure
        content: str = service.extract_text_from_page_safe(url)
        return content
    except Exception:
        # Fallback in case safe mode still raises
        return ""
