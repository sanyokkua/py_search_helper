# CLI Usage and Development

This document provides a comprehensive guide to using and developing the Command Line Interface (CLI) for `py-search-helper`. The CLI exposes the library's core functionalities directly through the terminal, making it easy to perform web searches and content extraction without writing Python code.

## CLI Overview

The `py-search-helper` CLI is built using the `Typer` framework, providing a user-friendly and robust command-line experience. It wraps the three main functions of the `py-search-helper` library: `get_search_engines`, `search`, and `open_url`.

### Installation

The CLI is automatically installed when you install the `py-search-helper` package.

```bash
# If installing from source (development)
git clone https://github.com/sanyokkua/py-search-helper.git
cd py-search-helper
uv sync --all-extras

# If installing from PyPI (when published)
pip install py-search-helper
```

After installation, the `py-search-helper` command will be available in your terminal.

## Available Commands

### `py-search-helper get-engines`

Lists all available search engines that the library can use, along with their descriptions.

**Usage:**

```bash
py-search-helper get-engines
```

**Example Output:**

```
Available Search Engines:
  - ddgs: General web search (DuckDuckGo)
  - pyside: Qt for Python official documentation
  - wikipedia: Wikipedia encyclopedia
```

### `py-search-helper search [ENGINE] [QUERY]`

Performs a web search using a specified engine and query.

**Usage:**

```bash
py-search-helper search <ENGINE> <QUERY> [OPTIONS]
```

**Arguments:**

-   `<ENGINE>`: **(Required)** The ID of the search engine to use (e.g., `ddgs`, `wikipedia`).
-   `<QUERY>`: **(Required)** The search query string.

**Options:**

-   `--max-results`, `-m`: Maximum number of results to return (default: `10`).
-   `--site`, `-s`: Optional domain to restrict the search (e.g., `docs.python.org`).

**Examples:**

```bash
# Basic search with DuckDuckGo
py-search-helper search ddgs "python typer framework"

# Search with a custom result limit
py-search-helper search ddgs "python requests library" -m 5

# Search only within a specific domain
py-search-helper search ddgs "asyncio tutorial" -s docs.python.org

# Search Wikipedia for a specific topic
py-search-helper search wikipedia "Large Language Models" -m 3
```

### `py-search-helper open-page [URL]`

Extracts and displays the clean text content from a given URL.

**Usage:**

```bash
py-search-helper open-page <URL> [OPTIONS]
```

**Arguments:**

-   `<URL>`: **(Required)** The URL from which to extract content. Must start with `http://` or `https://`.

**Options:**

-   `--max-chars`, `-c`: Maximum number of characters to return (default: `500`). Use `0` for unlimited content.

**Examples:**

```bash
# Extract content with the default character limit
py-search-helper open-page https://www.example.com

# Extract full content (unlimited characters)
py-search-helper open-page https://docs.python.org/3/library/asyncio.html -c 0
```

## Error Handling

The CLI prints user-friendly error messages if an operation fails (e.g., invalid engine ID, network issues). The process will exit with a non-zero status code (`1`) in case of an error, which is useful for scripting.

## Development

The CLI is implemented in `src/py_search_helper/cli.py`. When contributing to the CLI:

-   Ensure all new commands are added to the `typer.Typer` app instance.
-   Use `typer.Argument` and `typer.Option` for defining command-line parameters.
-   Wrap calls to core library functions (`get_search_engines`, `search`, `open_url`) in `try...except PySearchHelperError` blocks to provide consistent error reporting.
-   Add corresponding tests in `tests/test_cli.py`.
