# MCP Server

## What is MCP?

Model Context Protocol (MCP) enables AI agents to interact with external tools and data sources. The py-search-helper MCP server exposes the library's search and content extraction functionality to AI agents like Claude.

## Available Tools

The MCP server provides three tools that map directly to the library's public API:

### 1. get_engines

List available search engines.

**Parameters:** None

**Returns:** `list[tuple[str, str]]` - List of (engine_id, description) pairs

**Example:**
```python
[
    ("ddgs", "DuckDuckGo web search"),
    ("pyside", "PySide6 documentation search"),
    ("wikipedia", "Wikipedia encyclopedia search")
]
```

### 2. search_web

Search using specified engine.

**Parameters:**
- `engine` (str): Engine ID (e.g., "ddgs", "pyside", "wikipedia")
- `query` (str): Search query string
- `max_results` (int, optional): Maximum number of results (default: 10, range: 1-30)

**Returns:** Markdown-formatted search results with titles, URLs, and descriptions

**Example:**
```markdown
# Search Results for "python asyncio"

## 1. Python Asyncio Tutorial
https://docs.python.org/3/library/asyncio.html
Official Python documentation for asyncio...

## 2. Real Python: Async IO in Python
https://realpython.com/async-io-python/
Complete guide to async programming...
```

### 3. open_page

Extract content from a URL.

**Parameters:**
- `url` (str): URL to open and extract content from
- `max_chars` (int | None, optional): Maximum characters to return (default: 500)

**Returns:** Markdown-formatted content extracted from the page

**Example:**
```markdown
# Page Title

Main content of the page extracted as Markdown...
```

## Usage

### STDIO Mode (Claude Desktop)

STDIO mode is used for local integration with Claude Desktop.

**Start Server:**
```bash
uv run python -m py_search_helper.mcp
```

**Configure Claude Desktop:**

Edit the Claude Desktop configuration file:

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

**Restart Claude Desktop** after editing the configuration file.

### HTTP Mode (Network Access)

HTTP mode enables network-based access to the MCP server.

**Start Server:**
```bash
fastmcp run py_search_helper.mcp.server:mcp --transport http --port 8000
```

**Server URL:**
```
http://localhost:8000
```

**Access from Network:**
Replace `localhost` with your server's IP address for network access.

## Typical AI Agent Workflow

1. **Discovery**: Agent calls `get_engines()` to discover available search sources
2. **Search**: Agent calls `search_web(engine, query)` to find relevant information
3. **Content Extraction**: Agent calls `open_page(url)` to read specific pages
4. **Processing**: Agent analyzes content and responds to user

**Example Interaction:**
```python
# 1. Discover engines
engines = get_engines()
# → [("ddgs", "DuckDuckGo web search"), ...]

# 2. Search for information
results = search_web(engine="ddgs", query="python asyncio tutorial", max_results=5)
# → Markdown with 5 search results

# 3. Open first result URL
content = open_page(url="https://docs.python.org/3/library/asyncio.html", max_chars=1000)
# → First 1000 characters of page content as Markdown

# 4. Agent processes content and responds
```

## Testing

### Test Import

Verify MCP server can be imported:

```bash
uv run python -c "from py_search_helper.mcp.server import mcp; print(mcp)"
```

Expected output:
```
<FastMCP server instance>
```

### Test Tools Directly

Test tools in Python:

```python
from py_search_helper.mcp.server import get_engines, search_web, open_page

# Test get_engines
engines = get_engines()
print(f"Available engines: {engines}")

# Test search_web
results = search_web(engine="ddgs", query="python", max_results=3)
print(f"Search results:\n{results}")

# Test open_page
content = open_page(url="https://example.com", max_chars=200)
print(f"Page content:\n{content}")
```

### Test with MCP Client

Use FastMCP's testing utilities:

```bash
# Run MCP server in test mode
fastmcp test py_search_helper.mcp.server:mcp
```

## Error Handling

The MCP server translates library exceptions to MCP error responses:

**ValueError** → MCP validation error
```python
search_web(engine="ddgs", query="")  # Empty query
# → Error: "Query cannot be empty"
```

**EngineNotFoundError** → MCP error
```python
search_web(engine="unknown", query="test")
# → Error: "Engine 'unknown' not found"
```

**SearchProviderError** → MCP error
```python
search_web(engine="ddgs", query="test")  # If search fails
# → Error: "Search failed: <details>"
```

## Configuration Options

### Rate Limiting

Rate limiting is handled internally by providers (e.g., DDGS enforces 3-second delays).

### Max Results

Limit results to reduce response size and processing time:

```python
search_web(engine="ddgs", query="python", max_results=5)  # Return max 5 results
```

### Content Truncation

Limit content length for large pages:

```python
open_page(url="https://example.com", max_chars=500)  # Return max 500 characters
```

## Advanced Usage

### Custom Transport

FastMCP supports multiple transport types:

```bash
# STDIO (default, for Claude Desktop)
fastmcp run server.py --transport stdio

# HTTP (for network access)
fastmcp run server.py --transport http --port 8000

# SSE (Server-Sent Events)
fastmcp run server.py --transport sse --port 8000
```

### Logging

Enable logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# MCP server will log all tool calls and results
```

### Running in Production

For production deployments:

```bash
# Use production ASGI server
uvicorn py_search_helper.mcp.server:mcp --host 0.0.0.0 --port 8000
```

## Troubleshooting

**Issue: Claude Desktop doesn't see tools**
- Verify configuration file path is correct
- Restart Claude Desktop after editing config
- Check config JSON is valid (no trailing commas)

**Issue: MCP server fails to start**
- Ensure all dependencies installed: `uv sync --all-extras`
- Check Python version: `python --version` (should be 3.14+)
- Test import: `python -c "from py_search_helper.mcp.server import mcp"`

**Issue: Search returns no results**
- Check internet connection
- Verify engine ID is correct (use `get_engines()`)
- Try with different query

**Issue: Rate limit errors**
- DDGS provider enforces 3-second delays automatically
- For frequent searches, consider adding custom rate limiting
- Use proxy if rate limits persist

## Further Reading

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Claude Desktop Integration Guide](https://docs.anthropic.com/claude/docs)
