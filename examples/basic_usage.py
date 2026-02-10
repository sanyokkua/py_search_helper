"""Basic usage examples for py-search-helper.

This module demonstrates the three core API functions:
1. get_search_engines() - Discover available search engines
2. search() - Search using a specific engine
3. open_url() - Extract content from a URL
"""

from py_search_helper import get_search_engines, open_url, search
from py_search_helper.exceptions import EngineNotFoundError, URLError


def example_1_discover_engines() -> None:
    """Example 1: Discover available search engines."""
    print("=" * 60)
    print("Example 1: Discover Available Search Engines")
    print("=" * 60)

    engines = get_search_engines()
    print(f"\nFound {len(engines)} search engine(s):\n")

    for engine_id, description in engines:
        print(f"  • {engine_id:15} - {description}")

    print()


def example_2_basic_search() -> None:
    """Example 2: Perform a basic search with default settings."""
    print("=" * 60)
    print("Example 2: Basic Search (Default Settings)")
    print("=" * 60)

    # Search with default max_results=10
    results = search(engine="ddgs", query="python asyncio")

    print("\nSearch query: 'python asyncio'")
    print(f"Results length: {len(results)} characters\n")
    print("First 500 characters of results:")
    print(results[:500])
    print("...\n")


def example_3_custom_max_results() -> None:
    """Example 3: Search with custom max_results parameter."""
    print("=" * 60)
    print("Example 3: Search with Custom max_results")
    print("=" * 60)

    # Search for only 3 results
    results = search(engine="ddgs", query="python", max_results=3)

    print("\nSearch query: 'python'")
    print("max_results: 3")
    print(f"\nResults:\n{results}\n")


def example_4_open_url_with_limit() -> None:
    """Example 4: Extract content from URL with character limit."""
    print("=" * 60)
    print("Example 4: Open URL with Character Limit")
    print("=" * 60)

    # Extract content with default max_chars=500
    content = open_url("https://example.com")

    print("\nURL: https://example.com")
    print("max_chars: 500 (default)")
    print(f"\nExtracted content ({len(content)} chars):\n")
    print(content)
    print()


def example_5_open_url_unlimited() -> None:
    """Example 5: Extract full content from URL."""
    print("=" * 60)
    print("Example 5: Open URL with Unlimited Characters")
    print("=" * 60)

    # Extract full content (no limit)
    content = open_url("https://example.com", max_chars=None)

    print("\nURL: https://example.com")
    print("max_chars: None (unlimited)")
    print(f"\nExtracted full content ({len(content)} chars):\n")
    print(content)
    print()


def example_6_open_url_custom_limit() -> None:
    """Example 6: Extract content with custom character limit."""
    print("=" * 60)
    print("Example 6: Open URL with Custom Character Limit")
    print("=" * 60)

    # Extract only first 200 characters
    content = open_url("https://example.com", max_chars=200)

    print("\nURL: https://example.com")
    print("max_chars: 200")
    print(f"\nExtracted content ({len(content)} chars):\n")
    print(content)
    print()


def example_7_error_handling() -> None:
    """Example 7: Handle errors gracefully."""
    print("=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)

    # Handle engine not found
    try:
        search(engine="nonexistent", query="test")
    except EngineNotFoundError as e:
        print(f"\n✓ Caught EngineNotFoundError: {e}")

    # Handle empty query
    try:
        search(engine="ddgs", query="")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")

    # Handle invalid URL
    try:
        open_url("not-a-valid-url")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")

    print()


def example_8_complete_workflow() -> None:
    """Example 8: Complete workflow - search and extract."""
    print("=" * 60)
    print("Example 8: Complete Workflow (Search → Extract)")
    print("=" * 60)

    # Step 1: Search
    print("\nStep 1: Searching for 'python typing'...")
    results = search(engine="ddgs", query="python typing", max_results=2)
    print("✓ Search complete")

    # Step 2: Parse first URL from results (simple parsing)
    lines = results.split("\n")
    first_url = None
    for line in lines:
        if line.startswith("**URL:**"):
            first_url = line.replace("**URL:**", "").strip()
            break

    if first_url:
        print(f"\nStep 2: Extracting content from: {first_url[:50]}...")

        # Step 3: Extract content
        try:
            content = open_url(first_url, max_chars=300)
            print(f"✓ Extracted {len(content)} characters\n")
            print("First 200 characters:")
            print(content[:200])
            print("...\n")
        except URLError as e:
            print(f"✗ Failed to extract content: {e}\n")
    else:
        print("✗ Could not find URL in search results\n")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PY-SEARCH-HELPER - BASIC USAGE EXAMPLES")
    print("=" * 60 + "\n")

    try:
        example_1_discover_engines()
        example_2_basic_search()
        example_3_custom_max_results()
        example_4_open_url_with_limit()
        example_5_open_url_unlimited()
        example_6_open_url_custom_limit()
        example_7_error_handling()
        example_8_complete_workflow()

        print("=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}\n")
        raise


if __name__ == "__main__":
    main()
