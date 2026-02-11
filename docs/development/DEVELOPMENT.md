# Development Guide

## Setup

### Requirements
- Python 3.14+
- uv package manager
- Internet connection (for search operations)

### Installation

```bash
# Install Python 3.14
uv python install 3.14

# Clone and install
git clone <repository-url>
cd py-search-helper
uv sync --all-extras
```

### Verify Installation

```bash
uv run python -c "from py_search_helper import get_search_engines; print(get_search_engines())"
```

Expected output:
```
[('ddgs', 'DuckDuckGo web search'), ('pyside', 'PySide6 documentation search'), ('wikipedia', 'Wikipedia encyclopedia search')]
```

## Project Structure

```
py-search-helper/
├── src/
│   └── py_search_helper/        # Main library package
│       ├── __init__.py          # Public API exports
│       ├── _bootstrap.py        # Provider auto-registration
│       ├── api/                 # Public API implementation
│       │   ├── engines.py       # get_search_engines()
│       │   ├── search.py        # search()
│       │   └── open.py          # open_url()
│       ├── providers/           # Search provider implementations
│       │   ├── base.py          # SearchProvider protocol
│       │   ├── ddgs_provider.py # DuckDuckGo Search (API-based)
│       │   ├── pyside_provider.py # PySide documentation search (DDGS-delegated)
│       │   └── wikipedia_provider.py # Wikipedia search (DDGS-delegated)
│       ├── registry/            # Engine registry
│       │   └── engines.py       # Registration and lookup
│       ├── extractors/          # Content extraction
│       │   └── content.py       # URL content extraction
│       ├── models/              # Data models
│       │   ├── engine_info.py   # EngineInfo model
│       │   └── search_result.py # SearchResult model
│       ├── mcp/                 # MCP server
│       │   └── server.py        # MCP server implementation
│       ├── exceptions.py        # Custom exceptions
│       └── types.py             # Type definitions
├── tests/                       # Test suite
├── docs/                        # Documentation
├── examples/                    # Usage examples
├── pyproject.toml              # Project configuration
└── uv.lock                     # Dependency lockfile
```

## Building

### Build Package

```bash
uv build
```

**Output:**
- `dist/py_search_helper-0.1.0-py3-none-any.whl` (wheel)
- `dist/py_search_helper-0.1.0.tar.gz` (source distribution)

### Install Locally

```bash
uv pip install dist/py_search_helper-0.1.0-py3-none-any.whl
```

## Code Quality

### Linting

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .
```

**Configuration:**
- Target: Python 3.14
- Line length: 120 characters
- Rules: E, W, F, I, B, C4, UP, ARG, SIM, TCH, PTH, ERA, PL, RUF

### Formatting

```bash
uv run ruff format .
```

**Configuration:**
- Quote style: double
- Indent: 4 spaces
- Line length: 120

### Type Checking

```bash
uv run mypy src/
```

**Configuration:**
- Strict mode enabled
- Python version: 3.14
- Tests excluded from type checking

## Testing

### Run Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# Specific file
uv run pytest tests/test_api/test_search.py

# Specific test
uv run pytest tests/test_api/test_search.py::test_search_validates_empty_query

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x
```

### Test Categories

**Unit Tests:**
- API function validation
- Provider implementation
- Registry operations
- Model validation

**Integration Tests:**
- Provider registration
- End-to-end search operations
- Content extraction
- MCP server tools

**Mock Tests:**
- External API calls (DDGS)
- HTTP requests
- Content extraction

## Dependencies

### Runtime Dependencies

```toml
[project.dependencies]
ddgs = ">=9.10.0"              # Search backend
fastmcp = ">=2.14.5"           # MCP server framework
py-web-text-extractor = ">=0.1.0"  # Content extraction
```

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "ruff>=0.8.0",             # Linting and formatting
    "mypy>=1.13.0",            # Type checking
    "pytest>=8.0.0",           # Testing framework
    "pytest-cov>=6.0.0",       # Coverage reporting
]
```

### Add Dependency

```bash
# Runtime dependency
uv add <package-name>

# Development dependency
uv add --dev <package-name>
```

### Update Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package <package-name>
```

## Pre-Commit Checklist

Run before committing:

```bash
# 1. Format code
uv run ruff format .

# 2. Fix linting issues
uv run ruff check --fix .

# 3. Type check
uv run mypy src/

# 4. Run tests
uv run pytest --cov=src
```

All checks should pass before committing.

## Code Style

### Docstrings

Google-style format required for all public functions:

```python
def search(engine: str, query: str, max_results: int = 10) -> str:
    """Search using specified engine.

    Args:
        engine: Engine ID (e.g., "ddgs", "pyside")
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Markdown-formatted search results

    Raises:
        ValueError: If query is empty or engine is invalid
        EngineNotFoundError: If engine is not registered
        SearchProviderError: If search operation fails
    """
```

### Type Hints

All public functions must have complete type hints:

```python
from collections.abc import Sequence

def process_results(results: Sequence[SearchResult], limit: int | None = None) -> list[str]:
    """Process search results."""
    return [r.title for r in results[:limit]]
```

### Import Organization

```python
# Standard library imports
import logging
from pathlib import Path

# Third-party imports
from ddgs import DDGS
from fastmcp import FastMCP

# Local imports
from py_search_helper.models import SearchResult
from py_search_helper.exceptions import SearchProviderError
```

## Troubleshooting

### Common Issues

**Issue: Import errors after installation**
```bash
# Solution: Reinstall in editable mode
uv pip install -e .
```

**Issue: Type checking fails**
```bash
# Solution: Install type stubs
uv add --dev types-requests
```

**Issue: Tests fail with import errors**
```bash
# Solution: Install test dependencies
uv sync --all-extras
```

### Getting Help

- Check documentation in `docs/`
- Review examples in `examples/`
- Run tests with `-v` for verbose output
- Check library documentation (DDGS, FastMCP, py-web-text-extractor)
