# py-search-helper

Python library for searching on the internet and extracting web content with MCP server support.

## What is py-search-helper?

py-search-helper provides a consistent interface for searching information on the internet and retrieving web content. It supports multiple search engines, clean content extraction from web pages, and exposes functionality through an MCP (Model Context Protocol) server for AI agent integration.

## Why Use This?

**For Python Developers:**
- Simple API with three functions: discover engines, search, and extract content
- No need to learn different search APIs or content extraction libraries
- Type-safe interface with comprehensive error handling
- Ready for integration into automation scripts, data pipelines, or applications

**For AI Agents:**
- MCP server enables seamless integration with Claude and other AI agents
- Agents can discover available search sources, perform targeted searches, and extract content
- Useful for research tasks, documentation lookup, and information gathering

**Use Cases:**
- Documentation search across multiple sources
- Content aggregation and analysis
- AI-powered search assistants
- Web scraping with clean text extraction

## Features

- **Search provider**: DuckDuckGo (wraps several search engines)
- **Domain-specific search filtering** (restrict results to specific websites)
- Extract clean text content from web pages using dual-extractor fallback (markitdown → trafilatura)
- Discover available search engines programmatically
- MCP (Model Context Protocol) server for AI agent integration
- Configurable result limits and character truncation
- Type-safe API with comprehensive exception handling

## Requirements

- Python 3.14+
- Internet connection for search and content extraction

## Installation

```bash
# Development installation with uv
git clone <repository-url>
cd py-search-helper
uv sync --all-extras

# Or install from PyPI (when published)
pip install py-search-helper
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

# Search Stack Overflow
so_results = search(engine="ddgs", query="python threading", site="stackoverflow.com")

# Search GitHub repositories
gh_results = search(engine="ddgs", query="python web framework", site="github.com")

# Search PySide documentation
pyside_results = search(engine="pyside", query="QPushButton", max_results=5)
print(pyside_results)  # PySide6 documentation results

# Search Wikipedia
wiki_results = search(engine="wikipedia", query="Python programming language", max_results=5)
print(wiki_results)  # Wikipedia articles (all languages)

# Search specific language edition
wiki_fr = search(engine="wikipedia", query="Python site:fr.wikipedia.org", max_results=5)
print(wiki_fr)  # French Wikipedia

# Extract content with default limit (max_chars=500)
content = open_url("https://example.com")
print(content)  # First 500 characters

# Extract unlimited content
full_content = open_url("https://example.com", max_chars=None)
```

### MCP Server Usage

Start the MCP server for AI agent integration:

```bash
# STDIO mode (for Claude Desktop)
uv run python -m py_search_helper.mcp

# HTTP mode (requires fastmcp[http])
fastmcp run py_search_helper.mcp.server:mcp --transport http --port 8000
```

Configure Claude Desktop (`claude_desktop_config.json`):

**File Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Configuration:**
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

Restart Claude Desktop after editing the configuration file.

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

**Guidelines:**
- Follow PEP 8 style (enforced by Ruff)
- Add type hints to all functions
- Write tests for new features
- Update documentation as needed
- Keep commits focused and atomic

See [Development Guide](docs/development/DEVELOPMENT.md) for detailed setup instructions.

## License

MIT License - See LICENSE file for details
