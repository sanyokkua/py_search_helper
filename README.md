# py-search-helper

[![CI](https://github.com/sanyokkua/py-search-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/sanyokkua/py-search-helper/actions)
[![PyPI version](https://badge.fury.io/py/py-search-helper.svg)](https://pypi.org/project/py-search-helper/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`py-search-helper` is a Python library that offers a unified interface for searching the internet and extracting web content. It integrates various search engines and provides functionality for clean text extraction from web pages. Additionally, it features an MCP (Model Context Protocol) server, enabling seamless integration with AI agents for research and information gathering tasks.

> Note: Single Page Applications (SPAs) cannot be parsed correctly from the initial HTML response because substantive content is dynamically rendered by JavaScript after load.

## Features

- **Search provider**: DuckDuckGo (wraps several search engines)
- **Domain-specific search filtering** (restrict results to specific websites)
- Extract clean text content from web pages using dual-extractor fallback (markitdown → trafilatura)
- Discover available search engines programmatically
- MCP (Model Context Protocol) server for AI agent integration
- Configurable result limits and character truncation
- Type-safe API with comprehensive exception handling

## Prerequisites

- Python 3.14+ ([Download Python](https://www.python.org/downloads/))
- `uv` (recommended for dependency management and environment setup)
- Internet connection for search and content extraction

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/sanyokkua/py-search-helper.git
cd py-search-helper
uv sync --all-extras
```

### From PyPI (for library/mcp users)

```bash
pip install py-search-helper
```

### Verify Installation

```bash
uv run python -c "from py_search_helper import get_search_engines; print(get_search_engines())"
# Expected output (or similar): [('ddgs', 'General web search (DuckDuckGo)')]
```

## Configuration

### MCP

**Configuration Example:**
```json
{
  "mcpServers": {
    "py-search-helper": {
      "command": "uv",
      "args": ["run", "python", "-m", "py_search_helper.mcp"]
    }
  }
}
```

## Quick Start

### Library Usage

```python
from py_search_helper import get_search_engines, search, open_url

# Discover available search engines
engines = get_search_engines()
print(engines)
# Output: [('ddgs', 'General web search (DuckDuckGo)'), ('pyside', 'Qt for Python official documentation'), ('wikipedia', 'Wikipedia encyclopedia')]

# Search with default settings (max_results=10)
results = search(engine="ddgs", query="python asyncio")
print(results)  # Markdown-formatted search results

# Search with custom limit
results = search(engine="ddgs", query="python", max_results=5)

# Search specific domain
docs_results = search(engine="ddgs", query="asyncio tutorial", site="docs.python.org")

# Search PySide documentation
pyside_results = search(engine="pyside", query="QPushButton", max_results=5)
print(pyside_results)  # PySide6 documentation results

# Search Wikipedia
wiki_results = search(engine="wikipedia", query="Python programming language", max_results=5)
print(wiki_results)  # Wikipedia articles (all languages)

# Extract content with default limit (max_chars=500)
content = open_url("https://example.com")
print(content)  # First 500 characters

# Extract unlimited content
full_content = open_url("https://example.com", max_chars=None)
```

### CLI Usage

Once installed, the `py-search-helper` command provides access to the library's core functionalities directly from your terminal.

**List available search engines:**

```bash
py-search-helper get-engines
# Example Output:
# Available Search Engines:
#   - ddgs: General web search (DuckDuckGo)
#   - pyside: Qt for Python official documentation
#   - wikipedia: Wikipedia encyclopedia
```

**Perform a web search:**

```bash
# Basic search
py-search-helper search ddgs "python typer"

# Search with custom result limit
py-search-helper search ddgs "python requests" -m 5

# Search a specific domain
py-search-helper search ddgs "asyncio tutorial" -s docs.python.org
```

**Extract content from a URL:**

```bash
# Extract content with default character limit (500 chars)
py-search-helper open-page https://www.example.com

# Extract full content (unlimited characters)
py-search-helper open-page https://www.example.com -c 0
```

### MCP Server Usage

Start the MCP server for AI agent integration:

> Use [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) for testing MCPs


```bash
# STDIO mode
uv run py-search-helper-mcp

For MCP Inspector use command uv and as parameters:
`--directory ~path_to_th_project/py-search-helper run py-search-helper-mcp`

# HTTP mode (requires fastmcp[http])
fastmcp run py_search_helper.mcp.server:mcp --transport http --port 8000
```

## API Reference

### get_search_engines()

Returns list of available search engines.

**Returns:** `list[tuple[str, str]]` - List of (engine_id, description) tuples

**Raises:** `EngineError` - If engine registry fails

**Example:**
```python
engines = get_search_engines()
for engine_id, description in engines:
    print(f"{engine_id}: {description}")
```

### search(engine, query, max_results=10, *, site=None)

Search using specified engine with optional domain filtering.

**Parameters:**
- `engine` (str): Engine ID (e.g., "ddgs", "pyside", "wikipedia")
- `query` (str): Search query string
- `max_results` (int): Maximum results to return (default: 10, max recommended: 30)
- `site` (str | None): Optional domain to restrict search (default: None)

**Returns:** `str` - Markdown-formatted search results

**Raises:**
- `ValueError` - If query is empty, max_results < 1, or site is empty
- `EngineNotFoundError` - If engine is not registered
- `SearchProviderError` - If search operation fails

**Examples:**
```python
# General search
results = search(engine="ddgs", query="python typing", max_results=5)

# Search specific domain
results = search(
    engine="ddgs",
    query="machine learning",
    site="arxiv.org",
    max_results=10
)

# Search Stack Overflow
results = search(engine="ddgs", query="async await", site="stackoverflow.com")
```

### open_url(url, max_chars=500)

Extract content from a URL.

**Parameters:**
- `url` (str): URL to open (must start with http:// or https://)
- `max_chars` (int | None): Maximum characters to return (default: 500, None for unlimited)

**Returns:** `str` - Markdown-formatted page content

**Raises:**
- `ValueError` - If URL is invalid or max_chars < 1
- `URLNotFoundError` - If URL returns 404
- `URLTimeoutError` - If request times out
- `URLError` - If extraction fails

**Example:**
```python
# With character limit
content = open_url("https://example.com", max_chars=1000)

# Unlimited
full_content = open_url("https://example.com", max_chars=None)
```

## MCP Server Tools

The MCP server exposes three tools for AI agents:

### get_engines

List available search engines.

**Parameters:** None

**Returns:** List of (engine_id, description) tuples

### search_web

Search using specified engine with optional domain filtering.

**Parameters:**
- `engine` (str): Engine ID
- `query` (str): Search query
- `max_results` (int): Maximum results (default: 10)
- `site` (str | None): Optional domain to restrict search (default: None)

**Returns:** Markdown-formatted search results

**Examples:**
```python
# General search
search_web(engine="ddgs", query="python asyncio", max_results=5)

# Search specific domain
search_web(engine="ddgs", query="asyncio", site="docs.python.org", max_results=5)
```

### search_web_ddg

Search using DuckDuckGo engine with optional domain filtering.

**Parameters:**
- `query` (str): Search query
- `max_results` (int): Maximum results (default: 10)
- `site` (str | None): Optional domain to restrict search (default: None)

**Returns:** Markdown-formatted search results

**Examples:**
```python
# General search
search_web_ddg(query="python asyncio", max_results=5)

# Search specific domain
search_web_ddg(query="asyncio", site="docs.python.org", max_results=5)
```

### open_page

Extract content from a URL.

**Parameters:**
- `url` (str): URL to open
- `max_chars` (int | None): Maximum characters (default: 500)

**Returns:** Markdown-formatted page content

## Error Handling

The library uses a hierarchical exception system:

```
PySearchHelperError (base)
├── EngineError
│   └── EngineNotFoundError
├── SearchError
│   └── SearchProviderError
└── URLError
    ├── URLNotFoundError
    └── URLTimeoutError
```

**Example:**
```python
from py_search_helper import search, open_url
from py_search_helper.exceptions import (
    EngineNotFoundError,
    SearchProviderError,
    URLError,
)

try:
    results = search(engine="ddgs", query="python")
except EngineNotFoundError:
    print("Engine not found")
except SearchProviderError:
    print("Search failed")

try:
    content = open_url("https://example.com")
except URLError as e:
    print(f"URL error: {e}")
```

## Available Search Engines

### DDGS (DuckDuckGo Search)

- **Engine ID:** `ddgs`
- **Description:** General web search using DuckDuckGo
- **Features:** Rate limiting, retry logic, configurable result limits
- **Max Results:** 30 (recommended limit)

### PySide Documentation

- **Engine ID:** `pyside`
- **Description:** Search Qt for Python documentation
- **Implementation:** Uses DuckDuckGo with `site:doc.qt.io/qtforpython-6` filter
- **Scope:** PySide6 API documentation, tutorials, examples, and guides

### Wikipedia

- **Engine ID:** `wikipedia`
- **Description:** Search Wikipedia encyclopedia
- **Implementation:** Uses DuckDuckGo with `site:wikipedia.org` filter
- **Scope:** Wikipedia articles and encyclopedia content (all language editions)

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py` - Core API demonstrations (8 examples)
- `site_filtering.py` - Examples with a search with site filter

Run examples:

```bash
uv run python examples/basic_usage.py
uv run python examples/site_filtering.py
```

## Limitations

- DDGS provider subject to DuckDuckGo rate limits (6 requests/minute default)
- Content extraction may fail for JavaScript-heavy sites
- Some sites block automated scraping

## Documentation

**Architecture & Design:**
- [Architecture Overview](docs/architecture/ARCHITECTURE.md) - System design, component relationships, data flow

**Development:**
- [Development Guide](docs/development/DEVELOPMENT.md) - Setup, building, testing, code quality
- [Provider Implementation Guide](docs/development/PROVIDERS.md) - How to add new search providers

**MCP Server:**
- [MCP Server Guide](docs/mcp/MCP_SERVER.md) - MCP server setup, configuration, and usage

See [Development Guide](docs/development/DEVELOPMENT.md) for detailed setup instructions.

## License

MIT License - See LICENSE file for details
