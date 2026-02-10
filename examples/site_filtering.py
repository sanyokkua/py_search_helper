"""Example: Using site filtering to search specific domains.

This example demonstrates how to use the site parameter to restrict
search results to specific domains.
"""

from py_search_helper import search


def main() -> None:
    """Demonstrate site filtering functionality."""
    print("=" * 80)
    print("Site Filtering Examples")
    print("=" * 80)
    print()

    # Example 1: Search Python official documentation
    print("1. Searching Python Official Documentation")
    print("-" * 80)
    results = search(
        engine="ddgs",
        query="asyncio tutorial",
        site="docs.python.org",
        max_results=3,
    )
    print(results)
    print()

    # Example 2: Search Stack Overflow
    print("2. Searching Stack Overflow")
    print("-" * 80)
    results = search(
        engine="ddgs",
        query="python threading vs multiprocessing",
        site="stackoverflow.com",
        max_results=3,
    )
    print(results)
    print()

    # Example 3: Search GitHub repositories
    print("3. Searching GitHub Repositories")
    print("-" * 80)
    results = search(
        engine="ddgs",
        query="python web framework",
        site="github.com",
        max_results=3,
    )
    print(results)
    print()

    # Example 4: Search arXiv for research papers
    print("4. Searching arXiv (Research Papers)")
    print("-" * 80)
    results = search(
        engine="ddgs",
        query="machine learning transformers",
        site="arxiv.org",
        max_results=3,
    )
    print(results)
    print()

    # Example 5: Search MDN Web Docs
    print("5. Searching MDN Web Docs")
    print("-" * 80)
    results = search(
        engine="ddgs",
        query="javascript promises",
        site="developer.mozilla.org",
        max_results=3,
    )
    print(results)
    print()

    # Example 6: Search without site filter (general web search)
    print("6. General Web Search (No Site Filter)")
    print("-" * 80)
    results = search(
        engine="ddgs",
        query="python best practices",
        max_results=3,
    )
    print(results)
    print()

    # Example 7: Combining with PySide provider
    print("7. PySide Provider with Custom Site Override")
    print("-" * 80)
    # PySide provider normally defaults to doc.qt.io/qtforpython-6
    # But we can override it to search other Qt documentation sites
    results = search(
        engine="pyside",
        query="QPushButton",
        site="doc.qt.io",  # Override default to search broader Qt docs
        max_results=3,
    )
    print(results)
    print()

    print("=" * 80)
    print("Site Filtering Examples Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
