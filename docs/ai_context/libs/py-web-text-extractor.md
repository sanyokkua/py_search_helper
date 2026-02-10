# py-web-text-extractor — AI Reference Guide

> **Version:** 0.1.0
> **Language/Runtime:** Python 3.14+
> **Last Updated:** 2025-01-27
> **Source:** https://pypi.org/project/py-web-text-extractor/

---

## 📋 Overview

`py-web-text-extractor` is a Python library and CLI tool for extracting clean text content from web pages. It employs a dual-extractor fallback strategy, using `markitdown` as the primary extractor and automatically falling back to `trafilatura` if the primary method fails. The library provides both strict (exception-raising) and safe (error-suppressing) modes for flexible error handling in different use cases.

---

## 📦 Installation

### Standard Installation
```bash
pip install py-web-text-extractor
```

### Using uv (recommended for Python 3.14+)
```bash
uv add py-web-text-extractor
```

### Dependencies
| Dependency    | Version Constraint | Purpose                         |
| ------------- | ------------------ | ------------------------------- |
| `markitdown`  | `>=0.0.2`          | Primary text extraction engine  |
| `trafilatura` | `>=2.0.0`          | Fallback text extraction engine |
| `typer`       | `>=0.12.0`         | CLI framework                   |

### Version Compatibility Matrix
| Library Version | Python Version | Notes                          |
| --------------- | -------------- | ------------------------------ |
| 0.1.0           | 3.14+          | Requires Python 3.14 or higher |

### ⚠️ Python 3.14+ Compatibility Issue

The `pydub` library (a dependency of `markitdown`) contains invalid escape sequences that cause `SyntaxError` in Python 3.13+:

```
SyntaxError: "\(" is an invalid escape sequence
```

**Fix:** Run this patch script after installation:

```bash
#!/bin/bash
# fix_pydub.sh - Patch pydub for Python 3.13+ compatibility

PYDUB_UTILS=$(find .venv -name "utils.py" -path "*/pydub/utils.py" | head -n1)

if [ -z "$PYDUB_UTILS" ]; then
  echo "⚠️  pydub/utils.py not found, skipping patch"
  exit 0
fi

echo "🔧 Patching: $PYDUB_UTILS"
cp "$PYDUB_UTILS" "${PYDUB_UTILS}.bak"
sed -i.bak2 's/\\(/\\\\(/g; s/\\)/\\\\)/g' "$PYDUB_UTILS"
echo "✓ Successfully patched pydub/utils.py"
```

---

## 🚀 Quick Start

### CLI Quick Start
```bash
# Basic extraction - prints extracted text to stdout
py-web-text-extractor https://example.com

# Safe mode - returns empty output on failure instead of error
py-web-text-extractor https://example.com --safe

# Verbose mode - shows debug information
py-web-text-extractor https://example.com --verbose
```

### Python Library Quick Start
```python
from py_web_text_extractor import create_extractor_service

# Create service instance
service = create_extractor_service()

# Extract text (raises exception on failure)
text = service.extract_text_from_page("https://example.com")
print(text)

# Safe extraction (returns empty string on failure)
text = service.extract_text_from_page_safe("https://example.com")
```

---

## 🏗️ Core Concepts

### Dual Extractor Strategy
The library uses a fallback architecture for maximum reliability:
1. **Primary:** Attempts extraction using `markitdown`
2. **Fallback:** If `markitdown` fails or returns blank content, automatically retries with `trafilatura`
3. **Result:** Returns the first successful extraction; raises error or returns empty string if both fail

### Error Handling Modes

| Mode       | Method                          | Behavior on Failure       |
| ---------- | ------------------------------- | ------------------------- |
| **Strict** | `extract_text_from_page()`      | Raises specific exception |
| **Safe**   | `extract_text_from_page_safe()` | Returns empty string `""` |

### URL Validation
- URLs must be non-empty strings
- URLs must start with `http://` or `https://`
- Invalid URLs raise `UrlIsNotValidException` in strict mode

---

## 📖 API Reference

### `ExtractorService`

The main class for handling text extraction from web pages.

**Import:**
```python
from py_web_text_extractor.service.extractor_service import ExtractorService
# OR
from py_web_text_extractor import ExtractorService
# OR
from py_web_text_extractor import Extractor  # Alias
```

**Instantiation:**
```python
# Direct instantiation
service = ExtractorService()

# Using factory function
from py_web_text_extractor import create_extractor_service
service = create_extractor_service()
```

---

### `ExtractorService.extract_text_from_page()`

**Signature:**
```python
def extract_text_from_page(self, url: str) -> str
```

**Parameters:**
| Name  | Type  | Required | Default | Description                         |
| ----- | ----- | -------- | ------- | ----------------------------------- |
| `url` | `str` | Yes      | N/A     | HTTP/HTTPS URL to extract text from |

**Returns:** `str` — Cleaned text content extracted from the web page.

**Raises:**
| Exception                | Condition                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| `UrlIsNotValidException` | URL is `None`, empty, blank, not a string, or doesn't start with `http://` or `https://` |
| `TextExtractionFailure`  | Both `markitdown` and `trafilatura` extraction methods failed                            |

**Example:**
```python
from py_web_text_extractor import ExtractorService
from py_web_text_extractor.exception.exceptions import (
    TextExtractionError,
    UrlIsNotValidException,
)

service = ExtractorService()

try:
    text = service.extract_text_from_page("https://example.com")
    print(f"Extracted {len(text)} characters")
except UrlIsNotValidException as e:
    print(f"Invalid URL: {e}")
except TextExtractionError as e:
    print(f"Extraction failed: {e}")
```

**⚠️ Common Mistakes:**
- Passing a URL without the protocol prefix (`example.com` instead of `https://example.com`)
- Passing `None` or non-string values as the URL
- Not handling exceptions when the target page is unreachable or has no extractable content

---

### `ExtractorService.extract_text_from_page_safe()`

**Signature:**
```python
def extract_text_from_page_safe(self, url: str) -> str
```

**Parameters:**
| Name  | Type  | Required | Default | Description                                   |
| ----- | ----- | -------- | ------- | --------------------------------------------- |
| `url` | `str` | Yes      | N/A     | URL to extract text from (any value accepted) |

**Returns:** `str` — Extracted text if successful, empty string `""` on any failure.

**Raises:** None — All exceptions are caught and logged internally.

**Example:**
```python
from py_web_text_extractor import ExtractorService

service = ExtractorService()

# Safe for batch processing - never raises exceptions
urls = [
    "https://example.com",
    "https://invalid-domain-xyz.com",
    "not-a-url",
]

for url in urls:
    text = service.extract_text_from_page_safe(url)
    if text:
        print(f"✓ {url}: {len(text)} characters")
    else:
        print(f"✗ {url}: extraction failed or empty")
```

**⚠️ Common Mistakes:**
- Assuming an empty string means the page has no content (it could also indicate an error)
- Not checking if the result is empty before processing

---

### `create_extractor_service()`

Factory function for creating `ExtractorService` instances.

**Signature:**
```python
def create_extractor_service() -> ExtractorService
```

**Parameters:** None

**Returns:** `ExtractorService` — A new, configured service instance.

**Example:**
```python
from py_web_text_extractor import create_extractor_service

service = create_extractor_service()
text = service.extract_text_from_page("https://example.com")
```

---

### CLI Interface

**Command:**
```bash
py-web-text-extractor <URL> [OPTIONS]
```

**Arguments:**
| Argument | Type     | Required | Description                           |
| -------- | -------- | -------- | ------------------------------------- |
| `URL`    | `string` | Yes      | The web page URL to extract text from |

**Options:**
| Option      | Type   | Default | Description                                |
| ----------- | ------ | ------- | ------------------------------------------ |
| `--safe`    | `flag` | `False` | Enable safe mode (return empty on failure) |
| `--verbose` | `flag` | `False` | Enable debug output                        |

**Exit Codes:**
| Code | Constant          | Meaning                       |
| ---- | ----------------- | ----------------------------- |
| `0`  | Success           | Text extracted successfully   |
| `1`  | No Content        | No text content found on page |
| `2`  | Invalid URL       | The provided URL is malformed |
| `3`  | Extraction Failed | Text extraction failed        |
| `4`  | Unexpected Error  | An unexpected error occurred  |

**Examples:**
```bash
# Basic usage
py-web-text-extractor https://example.com

# Pipe output to file
py-web-text-extractor https://example.com > output.txt

# Safe mode with verbose logging
py-web-text-extractor https://example.com --safe --verbose

# Check exit code in script
py-web-text-extractor https://example.com
if [ $? -eq 0 ]; then
    echo "Success"
fi
```

---

## 📝 Exception Hierarchy

```python
TextExtractionError                    # Base exception for all library errors
├── UrlIsNotValidException             # Invalid or malformed URL
├── TextExtractionFailure              # All extraction methods failed
├── MarkItDownExtractionException      # markitdown-specific failure
└── TrafilaturaExtractionException     # trafilatura-specific failure
```

**Import:**
```python
from py_web_text_extractor.exception.exceptions import (
    TextExtractionError,
    UrlIsNotValidException,
    TextExtractionFailure,
    MarkItDownExtractionException,
    TrafilaturaExtractionException,
)
```

**Usage Pattern:**
```python
from py_web_text_extractor import ExtractorService
from py_web_text_extractor.exception.exceptions import (
    TextExtractionError,
    UrlIsNotValidException,
    TextExtractionFailure,
)

service = ExtractorService()

try:
    text = service.extract_text_from_page(url)
except UrlIsNotValidException:
    # Handle invalid URL - user input error
    print("Please provide a valid HTTP/HTTPS URL")
except TextExtractionFailure:
    # Handle extraction failure - external issue
    print("Could not extract text from this page")
except TextExtractionError:
    # Catch-all for any library error
    print("An error occurred during extraction")
```

---

## ✅ Best Practices

### DO ✓

- **Use the factory function** for creating service instances:
  ```python
  from py_web_text_extractor import create_extractor_service
  service = create_extractor_service()
  ```

- **Use safe mode for batch processing** where individual failures shouldn't stop the pipeline:
  ```python
  results = {url: service.extract_text_from_page_safe(url) for url in urls}
  ```

- **Use strict mode when you need to know about failures**:
  ```python
  try:
      text = service.extract_text_from_page(url)
  except TextExtractionError as e:
      log_error(url, e)
      raise
  ```

- **Check for empty strings** when using safe mode:
  ```python
  text = service.extract_text_from_page_safe(url)
  if not text:
      handle_empty_result(url)
  ```

- **Reuse the service instance** for multiple extractions:
  ```python
  service = create_extractor_service()
  for url in urls:
      text = service.extract_text_from_page_safe(url)
  ```

### DON'T ✗

- **Don't pass URLs without protocol**:
  ```python
  # ❌ Wrong
  service.extract_text_from_page("example.com")
  
  # ✓ Correct
  service.extract_text_from_page("https://example.com")
  ```

- **Don't catch base `Exception` when specific handling is needed**:
  ```python
  # ❌ Too broad
  try:
      text = service.extract_text_from_page(url)
  except Exception:
      pass
  
  # ✓ Specific handling
  try:
      text = service.extract_text_from_page(url)
  except UrlIsNotValidException:
      handle_bad_url()
  except TextExtractionFailure:
      handle_extraction_failure()
  ```

- **Don't assume empty results mean empty pages**:
  ```python
  # ❌ Misleading assumption
  text = service.extract_text_from_page_safe(url)
  if not text:
      print("Page is empty")  # Could be an error!
  
  # ✓ Better approach
  text = service.extract_text_from_page_safe(url)
  if not text:
      print("No content extracted (page may be empty or extraction failed)")
  ```

- **Don't create new instances for every extraction**:
  ```python
  # ❌ Inefficient
  for url in urls:
      service = ExtractorService()  # Creates new instance each time
      text = service.extract_text_from_page_safe(url)
  
  # ✓ Efficient
  service = ExtractorService()
  for url in urls:
      text = service.extract_text_from_page_safe(url)
  ```

---

## ⚠️ Common Pitfalls & Errors

### Invalid URL Format

**Symptom:**
```python
UrlIsNotValidException: Invalid URL: example.com
```

**Cause:** URL missing `http://` or `https://` protocol prefix.

**Solution:**
```python
# ❌ Wrong
url = "example.com"

# ✓ Correct
url = "https://example.com"

# ✓ Or validate/fix programmatically
if not url.startswith(("http://", "https://")):
    url = f"https://{url}"
```

---

### Non-String URL Type

**Symptom:**
```python
UrlIsNotValidException: URL must be a string, got NoneType
```

**Cause:** Passing `None` or non-string value as URL.

**Solution:**
```python
# ❌ Wrong
url = None
text = service.extract_text_from_page(url)

# ✓ Correct - validate before calling
url = get_url_from_user()
if url and isinstance(url, str):
    text = service.extract_text_from_page(url)
```

---

### Both Extractors Failed

**Symptom:**
```python
TextExtractionFailure: Failed to extract text from https://example.com using both MarkItDown and Trafilatura
```

**Cause:** The target page may be:
- Unreachable or returning errors
- Heavily JavaScript-dependent (content not in initial HTML)
- Blocking automated requests
- Containing only non-text content (images, videos)

**Solution:**
```python
# Use safe mode for resilience
text = service.extract_text_from_page_safe(url)

# Or handle the specific exception
try:
    text = service.extract_text_from_page(url)
except TextExtractionFailure:
    # Log and continue, or try alternative approach
    logger.warning(f"Could not extract content from {url}")
    text = ""
```

---

### pydub SyntaxError (Python 3.14+)

**Symptom:**
```
SyntaxError: "\(" is an invalid escape sequence
```

**Cause:** `pydub` library uses invalid escape sequences incompatible with Python 3.13+.

**Solution:** Run the patch script from the Installation section before using the library.

---

## 🔧 Configuration

### Logging Configuration

The library uses Python's standard `logging` module. Configure logging to see debug information:

```python
import logging

# Enable debug logging for the library
logging.basicConfig(level=logging.DEBUG)

# Or configure specific logger
logger = logging.getLogger("py_web_text_extractor")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# Now use the service
from py_web_text_extractor import create_extractor_service
service = create_extractor_service()
text = service.extract_text_from_page("https://example.com")
```

**Log Levels:**
| Level     | Information Provided                                       |
| --------- | ---------------------------------------------------------- |
| `DEBUG`   | Extraction attempts, URL validation details                |
| `INFO`    | Fallback events (when markitdown fails, using trafilatura) |
| `WARNING` | Non-fatal errors, safe mode failures                       |
| `ERROR`   | Fatal errors before exception is raised                    |

---

## 🧪 Testing Patterns

### Unit Testing with the Library

```python
import pytest
from py_web_text_extractor import ExtractorService
from py_web_text_extractor.exception.exceptions import (
    UrlIsNotValidException,
    TextExtractionFailure,
)


class TestExtractorService:
    """Test suite for ExtractorService."""

    @pytest.fixture
    def service(self):
        """Create a fresh service instance for each test."""
        return ExtractorService()

    def test_extract_text_from_valid_url(self, service):
        """Test extraction from a known working URL."""
        text = service.extract_text_from_page("https://example.com")
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Example Domain" in text

    def test_extract_raises_on_invalid_url(self, service):
        """Test that invalid URLs raise appropriate exception."""
        with pytest.raises(UrlIsNotValidException):
            service.extract_text_from_page("not-a-valid-url")

    def test_extract_raises_on_empty_url(self, service):
        """Test that empty URLs raise appropriate exception."""
        with pytest.raises(UrlIsNotValidException):
            service.extract_text_from_page("")

    def test_extract_raises_on_none_url(self, service):
        """Test that None URL raises appropriate exception."""
        with pytest.raises(UrlIsNotValidException):
            service.extract_text_from_page(None)

    def test_safe_mode_returns_empty_on_invalid_url(self, service):
        """Test that safe mode returns empty string on invalid URL."""
        result = service.extract_text_from_page_safe("not-a-valid-url")
        assert result == ""

    def test_safe_mode_returns_string(self, service):
        """Test that safe mode always returns a string."""
        result = service.extract_text_from_page_safe("https://example.com")
        assert isinstance(result, str)


@pytest.mark.integration
class TestExtractorServiceIntegration:
    """Integration tests requiring network access."""

    def test_real_website_extraction(self):
        """Test extraction from a real website."""
        service = ExtractorService()
        text = service.extract_text_from_page("https://httpbin.org/html")
        assert "Herman Melville" in text  # httpbin.org/html contains Moby Dick text
```

### Mocking for Isolated Tests

```python
from unittest.mock import patch, MagicMock
import pytest
from py_web_text_extractor import ExtractorService
from py_web_text_extractor.exception.exceptions import (
    MarkItDownExtractionException,
    TrafilaturaExtractionException,
    TextExtractionFailure,
)


class TestExtractorServiceMocked:
    """Tests with mocked extractors."""

    @patch("py_web_text_extractor.service.extractor_service.mk_extractor")
    def test_markitdown_success(self, mock_mk):
        """Test successful markitdown extraction."""
        mock_mk.extract_text.return_value = "Extracted content"
        
        service = ExtractorService()
        result = service.extract_text_from_page("https://example.com")
        
        assert result == "Extracted content"
        mock_mk.extract_text.assert_called_once_with("https://example.com")

    @patch("py_web_text_extractor.service.extractor_service.tr_extractor")
    @patch("py_web_text_extractor.service.extractor_service.mk_extractor")
    def test_fallback_to_trafilatura(self, mock_mk, mock_tr):
        """Test fallback when markitdown fails."""
        mock_mk.extract_text.side_effect = MarkItDownExtractionException("Failed")
        mock_tr.extract_text.return_value = "Trafilatura content"
        
        service = ExtractorService()
        result = service.extract_text_from_page("https://example.com")
        
        assert result == "Trafilatura content"
        mock_mk.extract_text.assert_called_once()
        mock_tr.extract_text.assert_called_once()

    @patch("py_web_text_extractor.service.extractor_service.tr_extractor")
    @patch("py_web_text_extractor.service.extractor_service.mk_extractor")
    def test_both_extractors_fail(self, mock_mk, mock_tr):
        """Test exception when both extractors fail."""
        mock_mk.extract_text.side_effect = MarkItDownExtractionException("MK Failed")
        mock_tr.extract_text.side_effect = TrafilaturaExtractionException("TR Failed")
        
        service = ExtractorService()
        
        with pytest.raises(TextExtractionFailure):
            service.extract_text_from_page("https://example.com")
```

---

## 🔗 Related Libraries & Ecosystem

| Library            | Purpose                          | Notes                                   |
| ------------------ | -------------------------------- | --------------------------------------- |
| `markitdown`       | HTML to Markdown/text conversion | Primary extractor used by this library  |
| `trafilatura`      | Web scraping & text extraction   | Fallback extractor used by this library |
| `typer`            | CLI framework                    | Powers the command-line interface       |
| `beautifulsoup4`   | HTML parsing                     | Alternative for custom extraction needs |
| `newspaper3k`      | Article extraction               | Alternative library for news articles   |
| `readability-lxml` | Content extraction               | Mozilla Readability port for Python     |

---

## 📚 Additional Resources

- [PyPI Package](https://pypi.org/project/py-web-text-extractor/)
- [GitHub Repository](https://github.com/sanyokkua/py_web_text_extractor)
- [Issue Tracker](https://github.com/sanyokkua/py_web_text_extractor/issues)
- [markitdown Documentation](https://github.com/microsoft/markitdown)
- [trafilatura Documentation](https://trafilatura.readthedocs.io/)

---

## 💡 AI Agent Instructions

When writing code using `py-web-text-extractor`:

1. **Always** include the full URL protocol (`https://` or `http://`)
2. **Always** import exceptions from `py_web_text_extractor.exception.exceptions`
3. **Never** pass `None` or non-string values to extraction methods
4. **Prefer** `create_extractor_service()` factory function over direct `ExtractorService()` instantiation
5. **Prefer** `extract_text_from_page_safe()` for batch processing pipelines
6. **Prefer** `extract_text_from_page()` when you need to handle specific error cases
7. **Check** that the result is not empty when using safe mode before processing
8. **Remember** that empty string from safe mode could mean error OR empty page
9. **Remember** this library requires Python 3.14+ and may need the pydub patch

### Code Generation Checklist

- [ ] URL includes `http://` or `https://` prefix
- [ ] Using `create_extractor_service()` or `ExtractorService()` for instantiation
- [ ] Appropriate exception handling for strict mode (`UrlIsNotValidException`, `TextExtractionFailure`)
- [ ] Empty string check when using `extract_text_from_page_safe()`
- [ ] Service instance is reused across multiple extractions (not recreated each time)
- [ ] Imports are from correct module paths

### Import Quick Reference

```python
# Main service
from py_web_text_extractor import ExtractorService
from py_web_text_extractor import Extractor  # Alias for ExtractorService
from py_web_text_extractor import create_extractor_service

# Exceptions
from py_web_text_extractor.exception.exceptions import (
    TextExtractionError,          # Base exception
    UrlIsNotValidException,       # Invalid URL
    TextExtractionFailure,        # Both extractors failed
    MarkItDownExtractionException,  # markitdown failed
    TrafilaturaExtractionException, # trafilatura failed
)
```
