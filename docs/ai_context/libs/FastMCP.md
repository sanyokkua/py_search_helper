# FastMCP — AI Reference Guide

> **Version:** 2.14.5 (stable) / 3.0.0b1 (beta)
> **Language/Runtime:** Python 3.10+
> **Last Updated:** February 2026
> **Source:** https://gofastmcp.com

---

## 📋 Overview

FastMCP is the standard Python framework for building MCP (Model Context Protocol) servers and clients. It provides a clean, decorator-based API that handles protocol complexity—serialization, validation, error handling—so developers can focus on business logic. FastMCP powers 70% of MCP servers across all languages and was incorporated into the official MCP SDK in 2024.

---

## 📦 Installation

### Standard Installation
```bash
uv pip install fastmcp
```

### Beta Version (FastMCP 3.0)
```bash
uv pip install fastmcp==3.0.0b1
```

### Production Pin (Stable v2)
```bash
uv pip install 'fastmcp<3'
```

### Peer Dependencies
| Package          | Purpose                               | Required           |
| ---------------- | ------------------------------------- | ------------------ |
| `uv`             | Dependency management, CLI operations | Highly recommended |
| `pytest-asyncio` | Testing async servers                 | For testing        |
| `starlette`      | HTTP transport, custom routes         | Included           |

### Version Compatibility Matrix
| FastMCP Version | Python Version | MCP SDK | Notes                   |
| --------------- | -------------- | ------- | ----------------------- |
| 2.14.x          | ≥3.10          | 1.x     | Current stable          |
| 3.0.0b1         | ≥3.10          | 1.x     | Beta, breaking changes  |
| <2.0            | ≥3.10          | 1.x     | Legacy, not recommended |

---

## 🚀 Quick Start

```python
# server.py
from fastmcp import FastMCP

# Create server instance
mcp = FastMCP("Demo 🚀")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool
def greet(name: str) -> str:
    """Greet someone by name"""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()  # STDIO transport (default)
```

**Run with CLI:**
```bash
# Using CLI (recommended)
fastmcp run server.py

# Or directly
python server.py
```

**Run with HTTP transport:**
```bash
fastmcp run server.py --transport http --port 8000
```

---

## 🏗️ Core Concepts

### Components
Components are what you expose to AI agents: **tools**, **resources**, and **prompts**. Wrap a Python function with a decorator, and FastMCP handles schema generation, validation, and documentation.

```python
@mcp.tool
def search(query: str, limit: int = 10) -> list[dict]:
    """Search the database"""
    return db.search(query, limit)
```

### Providers
Providers determine where components come from: decorated functions, files on disk, OpenAPI specs, or remote servers. Your logic can live anywhere.

### Transforms
Transforms shape what clients see: namespacing, filtering, authorization, versioning. The same server can present differently to different users.

### Transport Protocols
| Transport | Use Case                  | Multi-Client | Network |
| --------- | ------------------------- | ------------ | ------- |
| `stdio`   | Local dev, Claude Desktop | No           | No      |
| `http`    | Production, remote access | Yes          | Yes     |
| `sse`     | Legacy compatibility only | Yes          | Yes     |

---

## 📖 API Reference

### FastMCP Class

**Signature:**
```python
class FastMCP:
    def __init__(
        self,
        name: str = "FastMCP",
        instructions: str | None = None,
        stateless_http: bool = False,
        **kwargs
    ) -> None
```

**Parameters:**
| Name             | Type          | Required | Default     | Description                                  |
| ---------------- | ------------- | -------- | ----------- | -------------------------------------------- |
| `name`           | `str`         | No       | `"FastMCP"` | Server name shown to clients                 |
| `instructions`   | `str \| None` | No       | `None`      | Instructions for LLM clients                 |
| `stateless_http` | `bool`        | No       | `False`     | Enable stateless mode for horizontal scaling |

**Example:**
```python
from fastmcp import FastMCP

mcp = FastMCP(
    name="My Analysis Server",
    instructions="Use this server for data analysis tasks",
    stateless_http=True  # For load-balanced deployments
)
```

---

### @mcp.tool Decorator

**Signature:**
```python
@mcp.tool
def function_name(param: type, ...) -> return_type:
    """Docstring becomes tool description"""
    ...
```

**Parameters:**
| Name            | Type                  | Required | Default | Description                            |
| --------------- | --------------------- | -------- | ------- | -------------------------------------- |
| Function params | Any JSON-serializable | Varies   | —       | Automatically converted to JSON schema |
| Return type     | Any JSON-serializable | Yes      | —       | Serialized and returned to client      |

**Example:**
```python
@mcp.tool
def analyze_data(
    data: list[dict],
    column: str,
    operation: str = "mean"
) -> dict:
    """
    Analyze a column in the dataset.
    
    Args:
        data: List of records to analyze
        column: Column name to analyze
        operation: Statistical operation (mean, sum, count)
    
    Returns:
        Analysis result with value and metadata
    """
    import statistics
    values = [row[column] for row in data if column in row]
    
    if operation == "mean":
        result = statistics.mean(values)
    elif operation == "sum":
        result = sum(values)
    elif operation == "count":
        result = len(values)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return {"operation": operation, "column": column, "result": result}
```

**⚠️ Common Mistakes:**
- Forgetting type hints (FastMCP needs them for schema generation)
- Missing docstring (becomes empty tool description)
- Returning non-JSON-serializable objects (use `dict`, `list`, primitives)

---

### mcp.run() Method

**Signature:**
```python
def run(
    self,
    transport: Literal["stdio", "http", "sse"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    **kwargs
) -> None
```

**Parameters:**
| Name        | Type  | Required | Default       | Description                         |
| ----------- | ----- | -------- | ------------- | ----------------------------------- |
| `transport` | `str` | No       | `"stdio"`     | Protocol: `stdio`, `http`, or `sse` |
| `host`      | `str` | No       | `"127.0.0.1"` | Bind address (HTTP only)            |
| `port`      | `int` | No       | `8000`        | Port number (HTTP only)             |
| `path`      | `str` | No       | `"/mcp/"`     | URL path (HTTP only)                |

**Example:**
```python
if __name__ == "__main__":
    # STDIO for local/desktop use
    mcp.run()
    
    # HTTP for network access
    mcp.run(transport="http", host="0.0.0.0", port=8000)
```

**⚠️ Common Mistakes:**
- Calling `run()` inside an async function (use `run_async()` instead)
- Using `host="127.0.0.1"` when you need external access (use `"0.0.0.0"`)

---

### mcp.run_async() Method

**Signature:**
```python
async def run_async(
    self,
    transport: Literal["stdio", "http", "sse"] = "stdio",
    **kwargs
) -> None
```

**Example:**
```python
import asyncio
from fastmcp import FastMCP

mcp = FastMCP("Async Server")

@mcp.tool
async def fetch_data(url: str) -> dict:
    """Fetch data from URL"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

async def main():
    await mcp.run_async(transport="http", port=8000)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### mcp.http_app() Method

**Signature:**
```python
def http_app(
    self,
    path: str = "/mcp/",
    middleware: list[Middleware] | None = None
) -> Starlette
```

**Returns:** `Starlette` — ASGI application for use with Uvicorn or other servers

**Example:**
```python
from fastmcp import FastMCP

mcp = FastMCP("ASGI Server")

@mcp.tool
def process(data: str) -> str:
    return f"Processed: {data}"

# Create ASGI app
app = mcp.http_app()

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

---

### @mcp.custom_route Decorator

**Signature:**
```python
@mcp.custom_route(path: str, methods: list[str] = ["GET"])
async def handler(request: Request) -> Response
```

**Example:**
```python
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

@mcp.custom_route("/status", methods=["GET"])
async def status(request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy", "tools": 5})
```

---

## ✅ Best Practices

### DO ✓

- **Always use type hints** for all function parameters and return values
  ```python
  @mcp.tool
  def good(name: str, count: int = 10) -> list[str]:  # ✓ Full types
      ...
  ```

- **Write descriptive docstrings** — they become tool descriptions for LLMs
  ```python
  @mcp.tool
  def search(query: str) -> list[dict]:
      """
      Search the knowledge base for relevant documents.
      
      Args:
          query: Natural language search query
      
      Returns:
          List of matching documents with title and content
      """
      ...
  ```

- **Use `if __name__ == "__main__":`** for the run() call
  ```python
  if __name__ == "__main__":
      mcp.run()
  ```

- **Use factory functions** for complex setup that must run with `fastmcp run`
  ```python
  async def create_server() -> FastMCP:
      mcp = FastMCP("MyServer")
      await initialize_database()
      
      @mcp.tool
      def query(sql: str) -> list[dict]:
          return db.execute(sql)
      
      return mcp
  ```

- **Enable stateless mode** for load-balanced deployments
  ```python
  mcp = FastMCP("Scalable", stateless_http=True)
  ```

### DON'T ✗

- **Don't call `run()` inside async functions** — it creates its own event loop
  ```python
  async def main():
      mcp.run()  # ✗ ERROR: event loop already running
      await mcp.run_async()  # ✓ Use this instead
  ```

- **Don't rely on `if __name__ == "__main__":`** for setup with CLI
  ```python
  # This code WON'T run when using `fastmcp run server.py`
  if __name__ == "__main__":
      setup_database()  # ✗ Skipped by CLI
      mcp.run()
  ```

- **Don't use SSE transport** for new projects
  ```python
  mcp.run(transport="sse")  # ✗ Legacy, use "http" instead
  ```

- **Don't return complex objects** without serialization
  ```python
  @mcp.tool
  def bad() -> pd.DataFrame:  # ✗ Not JSON serializable
      return df
  
  @mcp.tool  
  def good() -> list[dict]:  # ✓ JSON serializable
      return df.to_dict(orient="records")
  ```

---

## 🔧 Configuration

### fastmcp.json Configuration File

```json
{
  "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
  "source": {
    "path": "server.py",
    "entrypoint": "mcp"
  },
  "environment": {
    "type": "uv",
    "python": ">=3.10",
    "dependencies": ["pandas", "httpx"]
  },
  "deployment": {
    "transport": "http",
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO",
    "env": {
      "API_KEY": "${API_KEY}",
      "DATABASE_URL": "postgres://${DB_HOST}/mydb"
    }
  }
}
```

| Section       | Field          | Type        | Default       | Description                                             |
| ------------- | -------------- | ----------- | ------------- | ------------------------------------------------------- |
| `source`      | `path`         | `string`    | **Required**  | Path to Python server file                              |
| `source`      | `entrypoint`   | `string`    | Auto-detect   | Server instance name (`mcp`, `server`, `app`)           |
| `environment` | `python`       | `string`    | System        | Python version constraint                               |
| `environment` | `dependencies` | `list[str]` | `[]`          | pip packages to install                                 |
| `environment` | `requirements` | `string`    | —             | Path to requirements.txt                                |
| `environment` | `editable`     | `list[str]` | —             | Paths for editable installs                             |
| `deployment`  | `transport`    | `string`    | `"stdio"`     | `stdio`, `http`, or `sse`                               |
| `deployment`  | `host`         | `string`    | `"127.0.0.1"` | Bind address                                            |
| `deployment`  | `port`         | `int`       | `3000`        | Port number                                             |
| `deployment`  | `env`          | `object`    | `{}`          | Environment variables (supports `${VAR}` interpolation) |

**Run with config:**
```bash
# Auto-detect fastmcp.json in current directory
fastmcp run

# Explicit config file
fastmcp run fastmcp.json

# Override config values
fastmcp run fastmcp.json --port 9000 --transport http
```

---

### Environment Variables

| Variable                 | Description                                 |
| ------------------------ | ------------------------------------------- |
| `FASTMCP_STATELESS_HTTP` | Set to `true` to enable stateless HTTP mode |

---

## ⚠️ Common Pitfalls & Errors

### "Event loop is already running"

**Symptom:**
```
RuntimeError: This event loop is already running
```

**Cause:** Calling `mcp.run()` inside an async function

**Solution:**
```python
# Wrong
async def main():
    mcp.run()  # Creates new event loop, conflicts with existing

# Correct
async def main():
    await mcp.run_async()  # Uses existing event loop
```

---

### "Cannot find server instance"

**Symptom:**
```
Error: Could not find a FastMCP server instance in server.py
```

**Cause:** Server variable not named `mcp`, `server`, or `app`

**Solution:**
```bash
# Option 1: Rename your variable
mcp = FastMCP("MyServer")  # ✓ Use standard name

# Option 2: Specify entrypoint explicitly
fastmcp run server.py:my_custom_name
```

---

### Setup code not running with CLI

**Symptom:** Database connections, API clients, or other setup code doesn't execute when using `fastmcp run`

**Cause:** `fastmcp run` ignores the `if __name__ == "__main__":` block

**Solution:** Use a factory function:
```python
async def create_server() -> FastMCP:
    # Setup code runs here
    await db.connect()
    
    mcp = FastMCP("MyServer")
    
    @mcp.tool
    def query(sql: str) -> list[dict]:
        return db.execute(sql)
    
    return mcp

# Run with: fastmcp run server.py:create_server
```

---

### Sessions fail in load-balanced environment

**Symptom:**
```
Session not found
```

**Cause:** Requests routed to different server instances

**Solution:** Enable stateless mode:
```python
mcp = FastMCP("MyServer", stateless_http=True)
```

Or via environment variable:
```bash
FASTMCP_STATELESS_HTTP=true uvicorn app:app --workers 4
```

---

### Dependencies not found when installed to Claude Desktop

**Symptom:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Cause:** MCP clients run servers in isolated environments

**Solution:** Specify dependencies explicitly:
```bash
fastmcp install claude-desktop server.py --with pandas --with numpy
```

Or use fastmcp.json:
```json
{
  "source": {"path": "server.py"},
  "environment": {"dependencies": ["pandas", "numpy"]}
}
```

---

## 🔄 Version Migration Guide

### From v2.x to v3.0

**Breaking Changes:**
- 3.0 is currently in **beta** — not recommended for production
- Some API changes may occur before final release

**Installation:**
```bash
# Install beta
uv pip install fastmcp==3.0.0b1

# Pin to stable v2 for production
uv pip install 'fastmcp<3'
```

**Documentation:**
- v3.0 docs: https://gofastmcp.com/getting-started/welcome
- v2.x docs: https://gofastmcp.com/v2/getting-started/welcome

---

## 📝 Type Definitions

```python
from typing import Literal, Any
from starlette.applications import Starlette
from starlette.middleware import Middleware

TransportType = Literal["stdio", "http", "sse"]

class FastMCP:
    name: str
    instructions: str | None
    
    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]: ...
    def resource(self, uri: str) -> Callable[[Callable], Callable]: ...
    def prompt(self, fn: Callable[..., Any]) -> Callable[..., Any]: ...
    
    def run(
        self,
        transport: TransportType = "stdio",
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/mcp/",
    ) -> None: ...
    
    async def run_async(
        self,
        transport: TransportType = "stdio",
        **kwargs,
    ) -> None: ...
    
    def http_app(
        self,
        path: str = "/mcp/",
        middleware: list[Middleware] | None = None,
    ) -> Starlette: ...
    
    def custom_route(
        self,
        path: str,
        methods: list[str] = ["GET"],
    ) -> Callable[[Callable], Callable]: ...
```

---

## 🧪 Testing Patterns

### Setup pytest for async testing

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

```bash
uv pip install pytest pytest-asyncio
```

### Basic test fixture

```python
import pytest
from fastmcp import FastMCP
from fastmcp.client import Client

# Import your server
from my_project.server import mcp

@pytest.fixture
async def client():
    """Create a test client connected to the server"""
    async with Client(transport=mcp) as client:
        yield client

async def test_list_tools(client: Client):
    """Test that expected tools are available"""
    tools = await client.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "add" in tool_names
    assert "greet" in tool_names

async def test_call_tool(client: Client):
    """Test calling a tool"""
    result = await client.call_tool(
        name="add",
        arguments={"a": 2, "b": 3}
    )
    assert result.data == 5
```

### Parametrized testing

```python
import pytest

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
async def test_add_tool(client: Client, a: int, b: int, expected: int):
    result = await client.call_tool(
        name="add",
        arguments={"a": a, "b": b}
    )
    assert result.data == expected
```

---

## 🔗 Related Libraries & Ecosystem

| Library     | Purpose           | Compatibility                              |
| ----------- | ----------------- | ------------------------------------------ |
| `mcp`       | Official MCP SDK  | FastMCP builds on this                     |
| `httpx`     | Async HTTP client | Use in async tools                         |
| `uvicorn`   | ASGI server       | Production HTTP deployment                 |
| `starlette` | Web framework     | Used internally, extend with custom routes |
| `pydantic`  | Data validation   | Automatic schema generation                |

---

## 📚 Additional Resources

- [Official Documentation](https://gofastmcp.com)
- [GitHub Repository](https://github.com/jlowin/fastmcp)
- [PyPI Package](https://pypi.org/project/fastmcp/)
- [Discord Community](https://discord.gg/uu8dJCgttd)
- [LLM-friendly docs (sitemap)](https://gofastmcp.com/llms.txt)
- [LLM-friendly docs (full)](https://gofastmcp.com/llms-full.txt)
- [MCP Protocol Specification](https://modelcontextprotocol.io)

---

## 💡 AI Agent Instructions

When writing code using FastMCP:

1. **Always** add type hints to all tool function parameters and return values
2. **Always** write docstrings for tools — they become the description LLMs see
3. **Always** use `if __name__ == "__main__":` around `mcp.run()` calls
4. **Never** call `mcp.run()` inside an async function — use `mcp.run_async()`
5. **Never** use `transport="sse"` for new projects — use `"http"` instead
6. **Prefer** factory functions over `__main__` blocks when setup code must run with CLI
7. **Prefer** `fastmcp.json` configuration over CLI arguments for reproducibility
8. **Check** that return values are JSON-serializable (dict, list, str, int, float, bool, None)
9. **Remember** that `fastmcp run` ignores the `if __name__ == "__main__":` block entirely
10. **Remember** to specify `--with` dependencies when installing to Claude Desktop/Cursor

### Code Generation Checklist
- [ ] All tool functions have complete type hints
- [ ] All tool functions have descriptive docstrings
- [ ] Return values are JSON-serializable
- [ ] `mcp.run()` is inside `if __name__ == "__main__":` block
- [ ] Using `run_async()` if inside async context
- [ ] Dependencies listed in fastmcp.json or `--with` flags for installation
- [ ] Using `"http"` transport (not `"sse"`) for network deployments
- [ ] Factory function used if setup code must run with CLI

### CLI Quick Reference
```bash
# Run server (STDIO)
fastmcp run server.py

# Run server (HTTP)
fastmcp run server.py --transport http --port 8000

# Run with auto-reload for development
fastmcp run server.py --reload

# Run with MCP Inspector
fastmcp dev server.py

# Install to Claude Desktop
fastmcp install claude-desktop server.py --with pandas

# Install to Cursor
fastmcp install cursor server.py --with pandas

# List tools on a server
fastmcp list server.py

# Call a tool
fastmcp call server.py add a=1 b=2

# Inspect server configuration
fastmcp inspect server.py --format fastmcp
```