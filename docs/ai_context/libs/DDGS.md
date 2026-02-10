# DDGS (Dux Distributed Global Search) — AI Reference Guide

> **Version:** 9.10.0
> **Language/Runtime:** Python >= 3.10
> **Last Updated:** 2025-01-13
> **Source:** https://github.com/deedy5/ddgs

---

## 📋 Overview

DDGS (Dux Distributed Global Search) is a Python metasearch library that aggregates results from diverse web search services including Google, Bing, DuckDuckGo, Brave, Wikipedia, and others. It provides a unified interface for text, image, video, news, and book searches with built-in proxy support, result deduplication, and automatic backend fallback mechanisms.

> ⚠️ **CRITICAL**: The package was renamed from `duckduckgo-search` to `ddgs`. The old package names (`duckduckgo-search`, `duckduckgo_search`) are **deprecated**. Always use `ddgs`.

> ⚠️ **RATE LIMITING**: Since DDGS scrapes public search engines, you will encounter rate limits. Use proxies and implement delays between requests. See [Rate Limiting & Throttling](#-rate-limiting--throttling) section.

---

## 📦 Installation

### Using UV (Recommended)
```bash
# Install UV first (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to project
uv add ddgs

# Or install globally
uv pip install ddgs

# Install with specific Python version
uv pip install ddgs --python 3.12

# Create new project with ddgs
uv init my-search-project
cd my-search-project
uv add ddgs
```

### Using pip
```bash
pip install -U ddgs
```

### Using pipx (for CLI usage)
```bash
pipx install ddgs
```

### Using Poetry
```bash
poetry add ddgs
```

### ❌ Deprecated Installation (Do NOT use)
```bash
# WRONG - These are deprecated:
pip install duckduckgo-search  # ❌ Deprecated
pip install duckduckgo_search  # ❌ Deprecated
```

### Dependencies
| Dependency | Purpose          | Version   |
| ---------- | ---------------- | --------- |
| `click`    | CLI framework    | >= 8.1.8  |
| `primp`    | HTTP client      | >= 0.15.0 |
| `lxml`     | XML/HTML parsing | >= 5.3.0  |

### Version Compatibility Matrix
| DDGS Version | Python Version | Notes                                |
| ------------ | -------------- | ------------------------------------ |
| 9.x          | >= 3.10        | Current stable                       |
| 8.x          | >= 3.9         | Deprecated (was `duckduckgo-search`) |
| 3.x          | >= 3.8         | Legacy, do not use                   |

---

## 🚀 Quick Start

```python
from ddgs import DDGS

# Simple text search
results = DDGS().text("python programming", max_results=5)
print(results)

# With proxy and custom timeout
ddgs = DDGS(proxy="http://user:pass@proxy.example.com:8080", timeout=10)
results = ddgs.text("machine learning")

# Using context manager (recommended for multiple searches)
with DDGS() as ddgs:
    text_results = ddgs.text("AI news", max_results=10)
    image_results = ddgs.images("sunset photos", max_results=5)
```

---

## 🏗️ Core Concepts

### Metasearch Architecture
DDGS doesn't query a single search engine—it aggregates results from multiple backends simultaneously. When you set `backend="auto"`, it intelligently selects and queries multiple search engines, deduplicates results, and ranks them by frequency.

### Backend Selection
- `"auto"` — Intelligent selection with automatic fallback (default)
- `"all"` — Query all available backends for maximum coverage
- Specific backends — Comma-separated list like `"google,brave,wikipedia"`

### Result Deduplication
Results from multiple backends are deduplicated using URL-based keys. When duplicates are found, the result with the longer body text is preserved, and results are ranked by frequency (how many backends returned it).

### Lazy Loading
The DDGS class is lazy-loaded, meaning search engine instances are only created when first needed and cached for subsequent use.

---

## 🚦 Rate Limiting & Throttling

> ⚠️ **IMPORTANT**: DDGS scrapes public search engines (DuckDuckGo, Google, Bing, etc.) that have strict anti-scraping mechanisms. Rate limiting is **expected behavior**, not a bug.

### Common Rate Limit Symptoms

| Symptom                   | Cause                                    | Solution                                |
| ------------------------- | ---------------------------------------- | --------------------------------------- |
| `RatelimitException`      | API limits exhausted                     | Use proxies, add delays                 |
| HTTP 202 Ratelimit        | DuckDuckGo blocking non-browser requests | Rotate proxies, reduce frequency        |
| Empty results             | Temporary IP ban                         | Wait 10-30 minutes, use different proxy |
| `TimeoutException` spikes | Backend throttling                       | Increase timeout, reduce concurrency    |

### Rate Limit Mitigation Strategies

#### 1. Use Proxies (Official Recommendation)
```python
from ddgs import DDGS

# HTTP proxy
ddgs = DDGS(proxy="http://user:pass@proxy.example.com:8080")

# SOCKS5 proxy (better for anonymity)
ddgs = DDGS(proxy="socks5://127.0.0.1:9050")

# Tor Browser shortcut
ddgs = DDGS(proxy="tb")

# Rotating proxies example
import random

proxies = [
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080",
    "http://proxy3.example.com:8080",
]

def search_with_proxy_rotation(query: str):
    proxy = random.choice(proxies)
    return DDGS(proxy=proxy).text(query, max_results=10)
```

#### 2. Add Delays Between Requests
```python
import time
from ddgs import DDGS

def search_with_delay(queries: list[str], delay: float = 2.0):
    """Search multiple queries with delay between each."""
    results = []
    with DDGS() as ddgs:
        for query in queries:
            result = ddgs.text(query, max_results=10)
            results.append(result)
            time.sleep(delay)  # Wait between requests
    return results

# For heavy usage, use longer delays (5-10 seconds)
def safe_search(query: str):
    time.sleep(random.uniform(3.0, 7.0))  # Random delay
    return DDGS().text(query, max_results=10)
```

#### 3. Limit Results Per Request
```python
from ddgs import DDGS

# ❌ Aggressive - likely to trigger rate limits
results = DDGS().text("query", max_results=100)

# ✅ Conservative - less likely to be blocked
results = DDGS().text("query", max_results=20)

# ✅ Even safer for frequent searches
results = DDGS().text("query", max_results=10)
```

#### 4. Implement Request Queue with Rate Limiting
```python
import asyncio
import time
from collections import deque
from ddgs import DDGS
from ddgs.exceptions import RatelimitException

class RateLimitedSearcher:
    def __init__(self, requests_per_minute: int = 10, proxy: str | None = None):
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.proxy = proxy
    
    def search(self, query: str, max_results: int = 10) -> list[dict]:
        # Enforce minimum interval between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_request_time = time.time()
        return DDGS(proxy=self.proxy).text(query, max_results=max_results)

# Usage
searcher = RateLimitedSearcher(requests_per_minute=6)  # 1 request per 10 seconds
results = searcher.search("python tutorials")
```

#### 5. Implement Caching
```python
import hashlib
import json
from pathlib import Path
from ddgs import DDGS

class CachedSearcher:
    def __init__(self, cache_dir: str = ".search_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
    
    def _cache_key(self, query: str, **kwargs) -> str:
        key_data = json.dumps({"query": query, **kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def search(self, query: str, max_results: int = 10, **kwargs) -> list[dict]:
        cache_key = self._cache_key(query, max_results=max_results, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Check cache
        if cache_file.exists():
            import time
            if time.time() - cache_file.stat().st_mtime < self.ttl_seconds:
                return json.loads(cache_file.read_text())
        
        # Fetch fresh results
        results = DDGS().text(query, max_results=max_results, **kwargs)
        
        # Save to cache
        cache_file.write_text(json.dumps(results))
        return results

# Usage - same query won't hit the API twice within TTL
searcher = CachedSearcher(ttl_hours=6)
results = searcher.search("python async programming")
```

#### 6. Exponential Backoff with Retry
```python
import time
import random
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

def search_with_backoff(
    query: str,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_results: int = 10
) -> list[dict]:
    """Search with exponential backoff on rate limit errors."""
    
    for attempt in range(max_retries):
        try:
            return DDGS().text(query, max_results=max_results)
        
        except RatelimitException:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limited. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
            time.sleep(delay)
        
        except TimeoutException:
            if attempt == max_retries - 1:
                raise
            
            delay = base_delay * (attempt + 1)
            print(f"Timeout. Waiting {delay:.1f}s before retry")
            time.sleep(delay)
    
    return []

# Usage
results = search_with_backoff("machine learning tutorials")
```

#### 7. Async Rate Limiting (with aiolimiter)
```python
# First: uv add aiolimiter
import asyncio
from aiolimiter import AsyncLimiter
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor

# Allow 5 requests per minute
rate_limiter = AsyncLimiter(5, 60)

async def async_search(query: str, max_results: int = 10) -> list[dict]:
    async with rate_limiter:
        # DDGS is sync, so run in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: DDGS().text(query, max_results=max_results)
            )

async def search_many(queries: list[str]) -> list[list[dict]]:
    tasks = [async_search(q) for q in queries]
    return await asyncio.gather(*tasks)

# Usage
queries = ["python", "javascript", "rust", "go"]
results = asyncio.run(search_many(queries))
```

### Rate Limit Quick Reference

| Usage Pattern                | Recommended Settings                                   |
| ---------------------------- | ------------------------------------------------------ |
| Single query, occasional     | Default settings                                       |
| Multiple queries, sequential | 2-5 second delay between requests                      |
| Bulk queries (10+)           | Use proxy, 5-10 second delays, `max_results <= 20`     |
| Production/continuous        | Rotating proxies, caching, rate limiter (5-10 req/min) |
| High volume                  | Self-hosted API server with MCP integration            |

---

## 📖 API Reference

### DDGS Class

**Signature:**
```python
class DDGS:
    def __init__(
        self,
        proxy: str | None = None,
        timeout: int | None = 5,
        verify: bool | str = True
    )
```

**Parameters:**
| Name      | Type          | Required | Default | Description                                                                                                 |
| --------- | ------------- | -------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| `proxy`   | `str \| None` | No       | `None`  | Proxy URL (http/https/socks5). Also reads `DDGS_PROXY` env var. Use `"tb"` for Tor Browser (127.0.0.1:9150) |
| `timeout` | `int \| None` | No       | `5`     | HTTP request timeout in seconds                                                                             |
| `verify`  | `bool \| str` | No       | `True`  | SSL verification. `True` to verify, `False` to skip, or path to PEM file                                    |

**Example:**
```python
from ddgs import DDGS

# Basic initialization
ddgs = DDGS()

# With SOCKS5 proxy
ddgs = DDGS(proxy="socks5://127.0.0.1:9050")

# With Tor Browser shortcut
ddgs = DDGS(proxy="tb")  # Expands to socks5://127.0.0.1:9150

# With custom timeout and no SSL verification
ddgs = DDGS(timeout=30, verify=False)

# Using context manager
with DDGS(proxy="http://proxy:8080") as ddgs:
    results = ddgs.text("query")
```

---

### 1. text() — Web Search

**Signature:**
```python
def text(
    query: str,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    max_results: int | None = 10,
    page: int = 1,
    backend: str = "auto",
) -> list[dict[str, str]]
```

**Parameters:**
| Name          | Type          | Required | Default      | Description                                                                                            |
| ------------- | ------------- | -------- | ------------ | ------------------------------------------------------------------------------------------------------ |
| `query`       | `str`         | Yes      | —            | Search query. Supports operators: `filetype:pdf`, `site:example.com`, `intitle:term`, `"exact phrase"` |
| `region`      | `str`         | No       | `"us-en"`    | Region code (e.g., `us-en`, `uk-en`, `de-de`, `ru-ru`)                                                 |
| `safesearch`  | `str`         | No       | `"moderate"` | Content filter: `"on"`, `"moderate"`, `"off"`                                                          |
| `timelimit`   | `str \| None` | No       | `None`       | Time filter: `"d"` (day), `"w"` (week), `"m"` (month), `"y"` (year)                                    |
| `max_results` | `int \| None` | No       | `10`         | Max results. `None` returns all from first page                                                        |
| `page`        | `int`         | No       | `1`          | Results page number                                                                                    |
| `backend`     | `str`         | No       | `"auto"`     | Backend(s): `"auto"`, `"all"`, or comma-separated list                                                 |

**Available Backends:** `bing`, `brave`, `duckduckgo`, `google`, `grokipedia`, `mojeek`, `yandex`, `yahoo`, `wikipedia`

**Returns:** `list[dict[str, str]]`
```python
[
    {
        "title": "Page Title",
        "href": "https://example.com/page",
        "body": "Description or snippet text..."
    },
    # ...
]
```

**Example:**
```python
from ddgs import DDGS

# Basic search
results = DDGS().text("python programming", max_results=5)

# Search with filters
results = DDGS().text(
    query="climate change",
    region="us-en",
    safesearch="off",
    timelimit="y",  # Last year
    max_results=20,
    backend="google,brave,wikipedia"
)

# Search for PDF files
results = DDGS().text("machine learning filetype:pdf", max_results=10)

# Site-specific search
results = DDGS().text("site:github.com python async", max_results=10)
```

---

### 2. images() — Image Search

**Signature:**
```python
def images(
    query: str,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    max_results: int | None = 10,
    page: int = 1,
    backend: str = "auto",
    size: str | None = None,
    color: str | None = None,
    type_image: str | None = None,
    layout: str | None = None,
    license_image: str | None = None,
) -> list[dict[str, str]]
```

**Additional Parameters:**
| Name            | Type          | Required | Default | Description                                                                                                                                                |
| --------------- | ------------- | -------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `size`          | `str \| None` | No       | `None`  | `"Small"`, `"Medium"`, `"Large"`, `"Wallpaper"`                                                                                                            |
| `color`         | `str \| None` | No       | `None`  | `"color"`, `"Monochrome"`, `"Red"`, `"Orange"`, `"Yellow"`, `"Green"`, `"Blue"`, `"Purple"`, `"Pink"`, `"Brown"`, `"Black"`, `"Gray"`, `"Teal"`, `"White"` |
| `type_image`    | `str \| None` | No       | `None`  | `"photo"`, `"clipart"`, `"gif"`, `"transparent"`, `"line"`                                                                                                 |
| `layout`        | `str \| None` | No       | `None`  | `"Square"`, `"Tall"`, `"Wide"`                                                                                                                             |
| `license_image` | `str \| None` | No       | `None`  | `"any"` (All CC), `"Public"`, `"Share"`, `"ShareCommercially"`, `"Modify"`, `"ModifyCommercially"`                                                         |

**Available Backends:** `duckduckgo` (only)

**Returns:** `list[dict[str, str]]`
```python
[
    {
        "title": "Image title",
        "image": "https://example.com/full-image.jpg",
        "thumbnail": "https://example.com/thumb.jpg",
        "url": "https://example.com/page-with-image",
        "height": 1080,
        "width": 1920,
        "source": "Bing"
    },
    # ...
]
```

**Example:**
```python
from ddgs import DDGS

results = DDGS().images(
    query="butterfly",
    region="us-en",
    safesearch="off",
    size="Large",
    color="Monochrome",
    type_image="photo",
    layout="Wide",
    license_image="any",
    max_results=20
)
```

---

### 3. videos() — Video Search

**Signature:**
```python
def videos(
    query: str,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    max_results: int | None = 10,
    page: int = 1,
    backend: str = "auto",
    resolution: str | None = None,
    duration: str | None = None,
    license_videos: str | None = None,
) -> list[dict[str, str]]
```

**Additional Parameters:**
| Name             | Type          | Required | Default | Description                                              |
| ---------------- | ------------- | -------- | ------- | -------------------------------------------------------- |
| `resolution`     | `str \| None` | No       | `None`  | `"high"`, `"standart"` ⚠️ Note: "standart" not "standard" |
| `duration`       | `str \| None` | No       | `None`  | `"short"`, `"medium"`, `"long"`                          |
| `license_videos` | `str \| None` | No       | `None`  | `"creativeCommon"`, `"youtube"`                          |

**Available Backends:** `duckduckgo` (only)

**Returns:** `list[dict[str, str]]`
```python
[
    {
        "content": "https://www.youtube.com/watch?v=xxxxx",
        "description": "Video description...",
        "duration": "8:22",
        "embed_html": "<iframe ...></iframe>",
        "embed_url": "https://www.youtube.com/embed/xxxxx?autoplay=1",
        "image_token": "token_string",
        "images": {
            "large": "https://...",
            "medium": "https://...",
            "motion": "",
            "small": "https://..."
        },
        "provider": "Bing",
        "published": "2024-07-03T05:30:03.0000000",
        "publisher": "YouTube",
        "statistics": {"viewCount": 29059},
        "title": "Video Title",
        "uploader": "Channel Name"
    },
    # ...
]
```

**Example:**
```python
from ddgs import DDGS

results = DDGS().videos(
    query="python tutorial",
    region="us-en",
    safesearch="moderate",
    timelimit="m",  # Last month
    resolution="high",
    duration="medium",
    max_results=10
)
```

**⚠️ Common Mistakes:**
- Using `resolution="standard"` — the correct value is `"standart"` (typo in API)

---

### 4. news() — News Search

**Signature:**
```python
def news(
    query: str,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    max_results: int | None = 10,
    page: int = 1,
    backend: str = "auto",
) -> list[dict[str, str]]
```

**Parameters:**
| Name        | Type          | Required | Default | Description                                                    |
| ----------- | ------------- | -------- | ------- | -------------------------------------------------------------- |
| `timelimit` | `str \| None` | No       | `None`  | `"d"` (day), `"w"` (week), `"m"` (month) — ⚠️ No `"y"` for news |

**Available Backends:** `bing`, `duckduckgo`, `yahoo`

**Returns:** `list[dict[str, str]]`
```python
[
    {
        "date": "2024-07-03T16:25:22+00:00",
        "title": "Article headline",
        "body": "Article summary...",
        "url": "https://news-site.com/article",
        "image": "https://news-site.com/image.jpg",
        "source": "News Source Name"
    },
    # ...
]
```

**Example:**
```python
from ddgs import DDGS

results = DDGS().news(
    query="artificial intelligence",
    region="us-en",
    safesearch="off",
    timelimit="w",  # Last week
    max_results=15,
    backend="duckduckgo,yahoo"
)
```

---

### 5. books() — Book Search

**Signature:**
```python
def books(
    query: str,
    max_results: int | None = 10,
    page: int = 1,
    backend: str = "auto",
) -> list[dict[str, str]]
```

**Available Backends:** `annasarchive` (only)

**Returns:** `list[dict[str, str]]`
```python
[
    {
        "title": "The Sea-Wolf",
        "author": "Jack London",
        "publisher": "DigiCat, 2022",
        "info": "English [en], .epub, 🚀/zlib, 0.5MB, 📗 Book (unknown)",
        "url": "https://annas-archive.li/md5/574f6556f1df6717de4044e36c7c2782",
        "thumbnail": "https://s3proxy.cdn-zlib.sk/covers299/..."
    },
    # ...
]
```

**Example:**
```python
from ddgs import DDGS

results = DDGS().books(
    query="sea wolf jack london",
    max_results=10,
    page=1
)
```

---

## ✅ Best Practices

### DO ✓
- **Use context manager for multiple searches:**
  ```python
  with DDGS() as ddgs:
      results1 = ddgs.text("query1")
      results2 = ddgs.text("query2")
  ```
- **Implement rate limiting for production:**
  ```python
  import time
  
  def safe_search(queries: list[str]) -> list:
      results = []
      for query in queries:
          results.append(DDGS().text(query, max_results=10))
          time.sleep(3)  # 3 second delay between requests
      return results
  ```
- **Use proxies for heavy usage:**
  ```python
  ddgs = DDGS(proxy="socks5://127.0.0.1:9050")
  ```
- **Cache results to avoid redundant requests:**
  ```python
  # Don't search the same query twice in a session
  cache = {}
  def cached_search(query):
      if query not in cache:
          cache[query] = DDGS().text(query)
      return cache[query]
  ```
- **Handle exceptions properly:**
  ```python
  from ddgs import DDGS
  from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
  
  try:
      results = DDGS().text("query")
  except RatelimitException:
      time.sleep(60)  # Wait before retrying
  except TimeoutException:
      # Retry with longer timeout
      results = DDGS(timeout=30).text("query")
  except DDGSException as e:
      print(f"Search failed: {e}")
  ```
- **Keep `max_results` reasonable (≤30) to reduce rate limit risk**

### DON'T ✗
- **Don't use the old package name:**
  ```python
  # ❌ WRONG
  from duckduckgo_search import DDGS
  
  # ✅ CORRECT
  from ddgs import DDGS
  ```
- **Don't make rapid sequential requests without delays:**
  ```python
  # ❌ WRONG - Will likely trigger rate limits
  for query in queries:
      results = DDGS().text(query)
  
  # ✅ CORRECT
  for query in queries:
      results = DDGS().text(query)
      time.sleep(5)
  ```
- **Don't request excessive results:**
  ```python
  # ❌ Risky - more likely to be blocked
  results = DDGS().text("query", max_results=100)
  
  # ✅ Safer
  results = DDGS().text("query", max_results=20)
  ```
- **Don't use `timelimit="y"` for news searches** — only `"d"`, `"w"`, `"m"` are valid
- **Don't spell resolution as `"standard"`** — it's `"standart"` (API quirk)

---

## 🔧 Configuration

### Proxy Configuration
```python
from ddgs import DDGS

# HTTP proxy
ddgs = DDGS(proxy="http://user:pass@proxy.example.com:8080")

# SOCKS5 proxy
ddgs = DDGS(proxy="socks5://127.0.0.1:9050")

# Tor Browser shortcut
ddgs = DDGS(proxy="tb")  # Automatically uses socks5://127.0.0.1:9150

# Via environment variable
import os
os.environ["DDGS_PROXY"] = "http://proxy:8080"
ddgs = DDGS()  # Will use DDGS_PROXY
```

### Environment Variables
| Variable      | Purpose                | Example               |
| ------------- | ---------------------- | --------------------- |
| `DDGS_PROXY`  | Default proxy for DDGS | `http://proxy:8080`   |
| `HTTP_PROXY`  | System HTTP proxy      | `http://proxy:8080`   |
| `HTTPS_PROXY` | System HTTPS proxy     | `https://proxy:8080`  |
| `NO_PROXY`    | Hosts to bypass        | `localhost,127.0.0.1` |

---

## ⚠️ Common Pitfalls & Errors

### Wrong Package Name

**Symptom:**
```
ModuleNotFoundError: No module named 'duckduckgo_search'
```

**Cause:** Using the deprecated package name

**Solution:**
```bash
pip uninstall duckduckgo-search
pip install ddgs
# Or with UV:
uv pip uninstall duckduckgo-search
uv add ddgs
```
```python
# Change imports from:
from duckduckgo_search import DDGS  # ❌ Old

# To:
from ddgs import DDGS  # ✅ New
```

---

### RatelimitException / HTTP 202 Ratelimit

**Symptom:**
```
ddgs.exceptions.RatelimitException: Rate limit exceeded
```
Or HTTP status 202 Ratelimit from DuckDuckGo

**Cause:** Too many requests in short time; request doesn't look like real browser behavior

**Solution:**
```python
import time
import random
from ddgs import DDGS
from ddgs.exceptions import RatelimitException

def search_with_retry(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return DDGS().text(query, max_results=20)  # Keep max_results low
        except RatelimitException:
            wait_time = (2 ** attempt) * 5 + random.uniform(0, 3)  # 5-8s, 13-16s, 23-26s
            print(f"Rate limited. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")

# Better: Use proxy
results = DDGS(proxy="socks5://127.0.0.1:9050").text("query")
```

---

### TimeoutException

**Symptom:**
```
ddgs.exceptions.TimeoutException: Request timeout
```

**Cause:** Network issues or slow backend response

**Solution:**
```python
from ddgs import DDGS

# Increase timeout
ddgs = DDGS(timeout=30)
results = ddgs.text("query")
```

---

### Empty Results / IP Ban

**Symptom:**
```python
results = DDGS().text("query")
print(results)  # []
```

**Cause:** Temporary IP ban from frequent requests, backend issues, or overly restrictive filters

**Solution:**
```python
# 1. Wait before retrying (10-30 minutes for IP ban)
import time
time.sleep(600)  # Wait 10 minutes

# 2. Use a proxy
results = DDGS(proxy="http://proxy:8080").text("query")

# 3. Try different backends
results = DDGS().text("query", backend="google")

# 4. Reduce filters
results = DDGS().text("query", safesearch="off", timelimit=None)
```

---

### Invalid timelimit for News

**Symptom:**
Unexpected results or errors when using `timelimit="y"` with news

**Cause:** News search only supports `"d"`, `"w"`, `"m"` — not `"y"`

**Solution:**
```python
# ❌ WRONG
results = DDGS().news("topic", timelimit="y")

# ✅ CORRECT
results = DDGS().news("topic", timelimit="m")  # Max is month
```

---

### Misspelled Video Resolution

**Symptom:**
Resolution filter not working

**Cause:** The API expects `"standart"` not `"standard"`

**Solution:**
```python
# ❌ WRONG
results = DDGS().videos("query", resolution="standard")

# ✅ CORRECT
results = DDGS().videos("query", resolution="standart")
```

---

## 🔄 Version Migration Guide

### From duckduckgo-search (any version) to ddgs 9.x

**Breaking Changes:**
1. **Package renamed**: `duckduckgo-search` → `ddgs`
   ```bash
   pip uninstall duckduckgo-search
   pip install ddgs
   # Or with UV:
   uv pip uninstall duckduckgo-search
   uv add ddgs
   ```
   
2. **Import changed**: 
   ```python
   # Old
   from duckduckgo_search import DDGS
   
   # New
   from ddgs import DDGS
   ```

3. **Expanded backends**: New backends added (`grokipedia`, `mullvad_brave`, `mullvad_google`)

4. **Books search**: Now uses `annasarchive` backend

**Deprecated:**
- Package name `duckduckgo-search` → Use `ddgs` instead
- Package name `duckduckgo_search` → Use `ddgs` instead

---

## 📝 Type Definitions

```python
from typing import TypedDict

class TextResult(TypedDict):
    title: str
    href: str
    body: str

class ImageResult(TypedDict):
    title: str
    image: str
    thumbnail: str
    url: str
    height: int
    width: int
    source: str

class VideoResult(TypedDict):
    content: str
    description: str
    duration: str
    embed_html: str
    embed_url: str
    image_token: str
    images: dict[str, str]
    provider: str
    published: str
    publisher: str
    statistics: dict[str, int]
    title: str
    uploader: str

class NewsResult(TypedDict):
    date: str
    title: str
    body: str
    url: str
    image: str
    source: str

class BookResult(TypedDict):
    title: str
    author: str
    publisher: str
    info: str
    url: str
    thumbnail: str
```

---

## 🧪 Testing Patterns

```python
import pytest
from unittest.mock import Mock, patch
from ddgs import DDGS

# Test with mocked results
def test_text_search():
    mock_results = [
        {"title": "Test", "href": "https://test.com", "body": "Test body"}
    ]
    
    with patch.object(DDGS, 'text', return_value=mock_results):
        ddgs = DDGS()
        results = ddgs.text("test query")
        assert len(results) == 1
        assert results[0]["title"] == "Test"

# Integration test (requires network)
@pytest.mark.integration
def test_real_search():
    results = DDGS(timeout=30).text("python", max_results=5)
    assert len(results) > 0
    assert "title" in results[0]
    assert "href" in results[0]
    assert "body" in results[0]

# Test exception handling
def test_timeout_handling():
    from ddgs.exceptions import TimeoutException
    
    with pytest.raises(TimeoutException):
        # This would need actual network conditions to trigger
        DDGS(timeout=0.001).text("query")
```

---

## 🔗 Related Libraries & Ecosystem

| Library      | Purpose                        | Compatibility |
| ------------ | ------------------------------ | ------------- |
| `primp`      | HTTP client used internally    | >= 0.15.0     |
| `click`      | CLI framework                  | >= 8.1.8      |
| `lxml`       | HTML/XML parsing               | >= 5.3.0      |
| `aiolimiter` | Async rate limiting (optional) | Any           |

---

## 📚 Additional Resources

- [Official Documentation (README)](https://github.com/deedy5/ddgs)
- [GitHub Repository](https://github.com/deedy5/ddgs)
- [PyPI Package](https://pypi.org/project/ddgs/)
- [CLI Help](Run `ddgs --help` after installation)

---

## 💡 AI Agent Instructions

When writing code using DDGS:

1. **Always** import from `ddgs`, never from `duckduckgo_search`
2. **Always** handle `DDGSException`, `RatelimitException`, and `TimeoutException`
3. **Always** add delays (2-5 seconds) between multiple sequential searches
4. **Never** use `timelimit="y"` with `news()` — only `"d"`, `"w"`, `"m"` are valid
5. **Never** use `resolution="standard"` — the correct spelling is `"standart"`
6. **Never** make rapid sequential requests without delays (will trigger rate limits)
7. **Prefer** using context manager (`with DDGS() as ddgs:`) for multiple searches
8. **Prefer** proxies for any production or high-volume usage
9. **Prefer** `max_results <= 30` to reduce rate limit risk
10. **Check** that Python version is >= 3.10 before suggesting this library
11. **Remember** that `images()`, `videos()` only support `duckduckgo` backend
12. **Remember** that `books()` only supports `annasarchive` backend
13. **Remember** result structure varies by search type — always check the return format
14. **Remember** rate limiting is expected — always implement retry logic with backoff

### Code Generation Checklist
- [ ] Import is `from ddgs import DDGS`
- [ ] Exception handling is implemented (especially `RatelimitException`)
- [ ] Delays added between multiple requests (`time.sleep(3)` minimum)
- [ ] Appropriate `timeout` is set for production use
- [ ] `max_results` is kept reasonable (≤ 30)
- [ ] `timelimit` values match the search type (no `"y"` for news/videos)
- [ ] `resolution="standart"` not `"standard"` for videos
- [ ] Backend is appropriate for search type
- [ ] Proxy configured for production/heavy usage

### Quick Reference — Backends by Search Type
```
text()   → bing, brave, duckduckgo, google, grokipedia, mojeek, yandex, yahoo, wikipedia
images() → duckduckgo
videos() → duckduckgo  
news()   → bing, duckduckgo, yahoo
books()  → annasarchive
```

### Quick Reference — Time Limits
```
text()   → d, w, m, y
images() → d, w, m, y
videos() → d, w, m
news()   → d, w, m      ⚠️ NO "y"!
```

### Quick Reference — Rate Limit Prevention
```python
# Minimum safe pattern for multiple queries
import time
from ddgs import DDGS
from ddgs.exceptions import RatelimitException

with DDGS(timeout=15) as ddgs:
    for query in queries:
        try:
            results = ddgs.text(query, max_results=20)
            time.sleep(5)  # Wait 5 seconds between requests
        except RatelimitException:
            time.sleep(60)  # Wait 1 minute on rate limit
```