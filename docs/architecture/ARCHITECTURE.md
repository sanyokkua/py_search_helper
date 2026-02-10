# Architecture

## Project Structure

py-search-helper consists of 5 main components:

1. **Public API** (`api/`) - Three functions: get_search_engines(), search(), open_url()
2. **Provider System** (`providers/`) - Search engine implementations
3. **Registry** (`registry/`) - Provider management and lookup
4. **Content Extraction** (`extractors/`) - URL content retrieval
5. **MCP Server** (`mcp/`) - Model Context Protocol integration

## System Diagram

```
┌─────────────────────────────────────────────────┐
│         Public API (3 functions)                │
│  • get_search_engines()                         │
│  • search(engine, query, max_results)           │
│  • open_url(url, max_chars)                     │
└─────────────────┬───────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼──────┐  ┌───▼─────┐  ┌────▼────────┐
│ Registry │  │Provider │  │  Extractor  │
│(Engines) │  │(Search) │  │  (Content)  │
└────┬─────┘  └────┬────┘  └───┬─────────┘
     │             │           │
     │        ┌────▼────┐      │
     │        │  DDGS   │◄─────┼──────┐
     │        │Provider │      │      │
     │        └─────────┘      │      │
     │                         │  ┌───▼──────────┐
     │                         │  │ PySide/Wiki  │
     │                         │  │ Providers    │
     │                         │  └──────────────┘
     │                    ┌────▼──────────┐
     │                    │ py-web-text   │
     │                    │  extractor    │
     │                    └───────────────┘
     │
┌────▼────────────────────┐
│   MCP Server (Optional) │
│  • get_engines          │
│  • search_web           │
│  • open_page            │
└─────────────────────────┘
```

## Component Relationships

### API Layer → Registry
- API calls `registry.get_provider(engine)` to lookup search providers
- API calls `registry.get_all_engines()` to list available engines

### API Layer → Extractor
- API calls `extract_content(url)` to fetch web page content
- Extractor uses py-web-text-extractor library (markitdown → trafilatura fallback)

### Registry → Providers
- Registry stores provider instances
- Providers implement SearchProvider protocol (get_info, search)
- Bootstrap module auto-registers providers on import

### MCP Server → API
- MCP server wraps all three API functions as MCP tools
- Uses FastMCP framework for tool definition
- STDIO transport for Claude Desktop integration

## Data Flow

### Search Operation
1. User → API: `search(engine="ddgs", query="python")`
2. API → Registry: `get_provider("ddgs")`
3. Registry → Provider: DDGSProvider instance
4. Provider: performs search via DDGS library
5. Provider → API: `list[SearchResult]`
6. API: formats as Markdown
7. API → User: Markdown string

### Content Extraction
1. User → API: `open_url(url, max_chars=500)`
2. API → Extractor: `extract_content(url)`
3. Extractor: fetches and processes URL
4. Extractor → API: Markdown content
5. API: truncates to 500 characters if needed
6. API → User: Markdown string

## Provider Types

### API-Based (Example: DDGS)
Uses external library/API. Returns structured results directly.

**Characteristics:**
- Uses external library (ddgs)
- Returns structured results
- No HTML parsing required
- Direct API integration

### URL-Based with DDGS Backend (Example: PySide)
Uses DDGS provider with site-specific filters.

**Characteristics:**
- Leverages existing DDGS provider
- Adds site: filter (e.g., `site:doc.qt.io/qtforpython-6`)
- No new dependencies
- Inherits rate limiting from DDGS
- Simple implementation (~10 lines)

### DDGS-Delegated Provider (Example: PySide, Wikipedia)

**Status:** Production

**How it works:**
1. Creates instance of DDGSProvider as backend
2. Adds automatic site-specific filter (e.g., site:doc.qt.io)
3. Delegates search to DDGS with modified query
4. Returns results directly from DDGS
5. Inherits rate limiting and error handling from DDGS

**Example:**
- PySide provider: Searches `site:doc.qt.io/qtforpython-6`
- Wikipedia provider: Searches `site:wikipedia.org`

**Advantages:**
- Simple implementation (no HTML parsing)
- Inherits DDGS reliability and rate limiting
- No additional dependencies
- Maintainable and testable

**Characteristics:**
- Leverages existing DDGS provider
- Adds site-specific filters
- Production-ready with minimal code
- Domain-scoped search capability

## Technology Stack

**Core:**
- Python: 3.14+
- Search: ddgs>=9.10.0
- Content: py-web-text-extractor>=0.1.0
- MCP: FastMCP>=2.14.5

**Development:**
- Type Checking: MyPy (strict)
- Linting/Formatting: Ruff
- Testing: pytest, pytest-cov
- Build: uv

## Design Patterns

### Protocol-Based Polymorphism
Providers implement `SearchProvider` protocol, enabling duck typing and loose coupling.

### Registry Pattern
Central registry manages provider instances and lookups, enabling runtime engine discovery.

### Strategy Pattern
Different search strategies (API-based vs URL-based) encapsulated in provider implementations.

### Facade Pattern
Public API provides simple interface hiding complex provider, registry, and extraction logic.
