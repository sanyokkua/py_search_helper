# Python Documentation Skill

## Purpose

This skill enables AI agents to write high-quality Python documentation that follows PEP 257, PEP 8, and modern community best practices. The focus is on **communicating what code does, why constraints exist, and how to use it** — not on marketing language or implementation details.

---

## Core Principles

### 1. Documentation Philosophy

**Document the WHAT and WHY, not the HOW:**
- ✅ What the code does (its purpose and behavior)
- ✅ Why constraints or design decisions exist
- ✅ How to use the API (parameters, returns, exceptions)
- ❌ Implementation details (unless critical for understanding)
- ❌ Subjective praise ("robust", "fast", "powerful", "elegant")

**Type hints are CRITICAL:**
- Every public function must have complete type hints
- Type hints are as important as docstrings
- Modern Python (3.10+) syntax is required
- Type hints enable static analysis, IDE support, and self-documentation

**Documentation serves:**
- Future maintainers who need to understand intent
- API users who need to know how to call functions
- Tools (Sphinx, IDEs, `help()`) that generate reference docs
- Static analyzers (mypy, pyright, Ruff) that verify correctness

---

## What to Document

### Public API (ALWAYS document)

- ✅ **Modules** — purpose, exports, relationships
- ✅ **Public functions and methods** — all parameters, returns, exceptions
- ✅ **Public classes** — purpose, attributes, key methods
- ✅ **Public constants** — meaning and constraints
- ✅ **Package `__init__.py`** — package-level API surface
- ✅ **Exceptions** — when and why they're raised

**Public = accessible and supported for external use**

Use `__all__` to explicitly declare public API:
```python
__all__ = ['calculate_area', 'Rectangle', 'MAX_SIZE']
```

### Private/Internal Code (minimal or no docstrings)

- Private functions (`_helper`, `__internal`) → inline comments only
- Inner implementation logic → inline comments for *why*, not *what*
- Temporary variables → no documentation needed

---

## Docstring Format & Placement

### Mandatory Rules

1. **Triple double quotes only:** `"""Docstring here."""`
2. **First statement** in module/class/function (no comments or blank lines before)
3. **One style consistently** across the entire codebase

### Recommended Style: Google Style (Default)

**Use Google style unless the user explicitly requests otherwise.**

Google style is the **preferred default** for modern Python projects because:
- Clean, readable syntax without markup overhead
- Well-supported by Sphinx (via Napoleon extension)
- Compatible with Ruff's docstring linting
- Widely adopted across the Python ecosystem

**Alternative styles (only use if requested):**

| Style           | Use When                                                  |
| --------------- | --------------------------------------------------------- |
| **NumPy style** | User explicitly requests it or project already uses it    |
| **Sphinx reST** | User explicitly requests it or legacy project requirement |

**Example (Google style - use this by default):**
```python
def calculate_area(shape: str, dimensions: dict) -> float:
    """Calculate the area of a geometric shape.
    
    Args:
        shape: Type of shape ('circle', 'rectangle', 'triangle').
        dimensions: Dictionary of measurements required for the shape.
            For 'circle': {'radius': float}
            For 'rectangle': {'width': float, 'height': float}
            For 'triangle': {'base': float, 'height': float}
    
    Returns:
        Area in square units.
    
    Raises:
        ValueError: If shape is not recognized or dimensions are invalid.
        TypeError: If dimensions is not a dictionary.
    
    Example:
        >>> calculate_area('circle', {'radius': 5.0})
        78.53981633974483
    """
```

---

## Documentation Requirements by Type

### 1. Module Docstrings

**Place at the top of the file (first statement after any `#!/usr/bin/env python` or encoding declarations).**

Include:
- Purpose of the module
- What it exports (major functions, classes, constants)
- Relationship to other modules (if relevant)
- Brief usage example (optional)

```python
"""Geometry calculation utilities.

This module provides functions for calculating areas and perimeters
of common geometric shapes. It supports circles, rectangles, and triangles.

Exports:
    calculate_area: Compute area for a given shape.
    calculate_perimeter: Compute perimeter for a given shape.
    Shape: Enumeration of supported shape types.
    MAX_DIMENSION: Maximum allowed dimension value.
"""
```

### 2. Function/Method Docstrings

**Minimum required sections:**

1. **One-line summary** (imperative mood: "Calculate...", "Return...", "Process...")
2. **Parameters** (Args) — name, type, constraints, defaults
3. **Return value** — type and description
4. **Raises** — all exceptions the function may raise
5. **Examples** (recommended) — simple, realistic usage

**What to include:**
- Type information (if not already in type hints)
- Valid ranges or constraints (e.g., "must be positive")
- Side effects (e.g., "modifies the input list in-place")
- Default behavior when optional parameters are omitted

**What to avoid:**
- Repeating the function signature verbatim
- Obvious information already clear from parameter names
- Subjective descriptions ("efficiently", "quickly", "robustly")

```python
def process_image(image_path: str, resize: bool = False, max_width: int = 1920) -> Image:
    """Load and optionally resize an image.
    
    Args:
        image_path: Path to the image file.
        resize: Whether to resize the image. Defaults to False.
        max_width: Maximum width in pixels when resizing. Must be positive.
            Ignored if resize is False.
    
    Returns:
        Loaded Image object, resized if requested.
    
    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If max_width is not positive.
        PIL.UnidentifiedImageError: If file is not a valid image format.
    """
```

### 3. Class Docstrings

**Place immediately after the class definition.**

Include:
- What the class represents (its abstraction)
- Main responsibilities and behavior
- Public attributes and their meaning
- Important methods (high-level summary only)
- Usage example

```python
class ImageProcessor:
    """Process and transform image files.
    
    This class handles loading, resizing, filtering, and saving images.
    It maintains an internal cache of processed images to avoid redundant work.
    
    Attributes:
        cache_enabled: Whether to cache processed images.
        max_cache_size: Maximum number of images to keep in cache.
    
    Example:
        >>> processor = ImageProcessor(cache_enabled=True)
        >>> img = processor.load('photo.jpg')
        >>> processor.resize(img, width=800)
    """
```

**Special methods** (`__init__`, `__repr__`, `__str__`) only need docstrings if behavior is non-obvious.

### 4. Constants and Module-Level Variables

Document **public constants** that are part of the API:

```python
#: Maximum file size allowed for upload, in bytes.
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

#: Supported image formats.
SUPPORTED_FORMATS = ['jpg', 'png', 'gif', 'webp']
```

Use `#:` prefix for inline documentation that tools can extract.

### 5. Exceptions

Document:
- When the exception is raised
- What conditions trigger it
- How to avoid or handle it

```python
class InvalidShapeError(ValueError):
    """Raised when an unsupported shape type is provided.
    
    This exception is raised by geometry functions when the shape
    parameter does not match any recognized shape type.
    """
```

---

## Inline Comments

**Use inline comments (`#`) ONLY for:**
- Explaining **why** code does something non-obvious
- Documenting workarounds or trade-offs
- Clarifying complex algorithms or business logic

**DO NOT use inline comments to:**
- Restate what the code obviously does
- Describe function behavior (use docstrings)
- Explain variable names (use better names instead)

```python
# ❌ BAD: Obvious restatement
counter = counter + 1  # increment counter by 1

# ✅ GOOD: Explains non-obvious reasoning
# Use binary search instead of linear scan to handle large datasets
# efficiently while maintaining O(log n) worst-case complexity
result = binary_search(sorted_data, target)

# ✅ GOOD: Documents a workaround
# Work around Python 3.8 limitation where asyncio.run() cannot
# be called from within an existing event loop
if sys.version_info < (3, 9):
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(async_func())
```

---

## Documentation Anti-Patterns

### ❌ Avoid These Mistakes

**1. Marketing language instead of technical description:**
```python
# ❌ BAD
def process_data(data):
    """Powerful and robust data processor that efficiently handles data."""
```
```python
# ✅ GOOD
def process_data(data):
    """Remove null values and normalize numeric columns.
    
    Args:
        data: DataFrame containing raw data.
    
    Returns:
        DataFrame with null values removed and numeric columns
        scaled to [0, 1] range.
    """
```

**2. Documenting HOW instead of WHAT:**
```python
# ❌ BAD
def calculate_hash(text):
    """Uses SHA-256 algorithm to iterate through each byte..."""
```
```python
# ✅ GOOD
def calculate_hash(text):
    """Generate a SHA-256 hash of the input text.
    
    Args:
        text: String to hash.
    
    Returns:
        Hexadecimal hash string (64 characters).
    """
```

**3. Over-documenting trivial code:**
```python
# ❌ BAD
def get_name(self):
    """Return the name attribute.
    
    Returns:
        str: The name attribute of this object.
    """
    return self.name
```
```python
# ✅ GOOD (no docstring needed if obvious from context and type hints)
def get_name(self) -> str:
    return self.name
```

**4. Missing exception documentation:**
```python
# ❌ BAD
def divide(a, b):
    """Divide a by b."""
    return a / b  # Can raise ZeroDivisionError!
```
```python
# ✅ GOOD
def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    Args:
        a: Numerator.
        b: Denominator.
    
    Returns:
        Result of division.
    
    Raises:
        ZeroDivisionError: If b is zero.
    """
    return a / b
```

---

## Examples in Docstrings

Include **realistic, runnable examples** that:
- Show typical usage patterns
- Can be used as doctests
- Cover common use cases
- Are simple and self-contained

```python
def parse_date(date_string: str, format: str = '%Y-%m-%d') -> datetime:
    """Parse a date string into a datetime object.
    
    Args:
        date_string: Date in string format.
        format: strftime format string. Defaults to ISO format.
    
    Returns:
        Parsed datetime object.
    
    Raises:
        ValueError: If date_string does not match format.
    
    Examples:
        >>> parse_date('2026-02-09')
        datetime.datetime(2026, 2, 9, 0, 0)
        
        >>> parse_date('09/02/2026', format='%d/%m/%Y')
        datetime.datetime(2026, 2, 9, 0, 0)
    """
```

---

## Agent Workflow

When documenting Python code, follow this process:

### Step 0: Assess Existing Documentation (if any)

**If documentation already exists:**
- Read through all existing docstrings
- Check for completeness (missing parameters, returns, exceptions)
- Verify accuracy against current code signatures
- Note inconsistencies in style or terminology
- Flag outdated information or missing type hints

**Create a mental or written checklist of what needs updating.**

### Step 1: Verify/Add Type Hints

**Type hints are CRITICAL and must be checked first:**
- Scan all public functions and methods
- Ensure ALL parameters have type hints
- Ensure ALL return values have type hints
- Add missing type hints before writing docstrings
- Use modern syntax (`list[str]`, not `List[str]`)

```python
# Add type hints first
def process_data(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Process data..."""  # Docstring comes after type hints are correct
```

### Step 2: Identify Public API
- Scan for `__all__` declarations
- Identify all non-underscore-prefixed names at module level
- Note classes, functions, constants intended for external use
- Mark private helpers (leading underscore) as not needing docstrings

### Step 3: Document Each Public Element (Google style by default)
- **Module:** Add/update module docstring at file top
- **Classes:** Document purpose, attributes, key behavior
- **Functions/Methods:** Document all parameters, returns, exceptions
- **Constants:** Add inline doc comments (`#:`)

### Step 4: Check for Completeness
- [ ] Every public function has Args, Returns, Raises sections
- [ ] All exceptions are documented (check actual `raise` statements in code)
- [ ] Examples provided for non-trivial APIs
- [ ] Constraints and valid ranges noted (e.g., "must be positive")
- [ ] No marketing language or subjective claims
- [ ] Type hints present and accurate

### Step 5: Add Inline Comments (Sparingly)
- Only for non-obvious *why* explanations
- Never restate what code clearly shows
- Focus on algorithmic choices, workarounds, trade-offs

### Step 6: Verify Consistency
- Same docstring style throughout (Google unless specified otherwise)
- Same terminology for similar concepts
- No outdated information from previous implementations
- Type hints match docstring descriptions

### Step 7: Validate with Ruff

**Before considering documentation complete:**

```bash
# Run Ruff to check docstring compliance
ruff check --select D .

# Fix auto-fixable issues
ruff check --select D --fix .

# Verify formatting
ruff format --check .
```

**Address all Ruff warnings related to:**
- Missing docstrings (D100-D107)
- Docstring formatting (D200-D215)
- Docstring content (D400-D417)
- Type hint coverage (ANN001-ANN206)

---

## Quick Reference Checklist

Before marking documentation complete, verify:

### Type Hints (CRITICAL)
- [ ] All public functions have parameter type hints
- [ ] All public functions have return type hints
- [ ] Class attributes have type hints
- [ ] Modern type syntax used (`list[str]`, not `List[str]`)
- [ ] Complex types use TypeAlias when appropriate

### Docstrings (Google style by default)
- [ ] Module docstring describes purpose and exports
- [ ] All public functions/methods have complete docstrings
- [ ] Class docstrings include purpose and key attributes
- [ ] All function parameters documented with constraints
- [ ] Return values documented
- [ ] All raised exceptions documented
- [ ] Examples included for complex or non-obvious APIs

### Quality
- [ ] No marketing language ("robust", "fast", "efficient", etc.)
- [ ] No implementation details in public API docs
- [ ] Inline comments only for *why*, not *what*
- [ ] Consistent docstring style across all files
- [ ] `__all__` defined for modules with public API
- [ ] No outdated information from previous code versions

### Validation
- [ ] `ruff check --select D .` passes (or issues addressed)
- [ ] `ruff format --check .` passes
- [ ] Type hints verified with mypy/pyright
- [ ] Doctest examples run successfully (if applicable)
- [ ] Sphinx builds without warnings (if applicable)

---

## Type Hints: Critical Requirement

**Type hints are MANDATORY for all public API functions, methods, and class attributes.**

Modern Python projects (3.10+) rely on type hints for:
- Static analysis (mypy, pyright, Ruff)
- IDE autocomplete and inline documentation
- Runtime validation (with libraries like pydantic)
- Self-documenting code

### Type Hint Rules

1. **Always include type hints in function signatures:**
   ```python
   def process_data(input_path: str, max_rows: int = 1000) -> pd.DataFrame:
       """Load and process data from CSV file.
       
       Args:
           input_path: Path to CSV file.
           max_rows: Maximum number of rows to load.
       
       Returns:
           Processed DataFrame.
       """
   ```

2. **Use modern type hint syntax (Python 3.10+):**
   ```python
   # ✅ GOOD: Modern syntax
   def merge_dicts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
       """Merge two dictionaries."""
   
   # ❌ BAD: Old syntax (avoid unless Python < 3.10)
   from typing import Dict
   def merge_dicts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
       """Merge two dictionaries."""
   ```

3. **Document complex types in docstrings:**
   ```python
   def process_config(config: dict[str, Any]) -> None:
       """Process application configuration.
       
       Args:
           config: Configuration dictionary with keys:
               - 'api_key' (str): API authentication key
               - 'timeout' (int): Request timeout in seconds
               - 'retries' (int, optional): Number of retry attempts
       """
   ```

4. **Use TypeAlias for complex types:**
   ```python
   from typing import TypeAlias
   
   # Define at module level
   ConfigDict: TypeAlias = dict[str, str | int | bool]
   
   def load_config(path: str) -> ConfigDict:
       """Load configuration from file."""
   ```

5. **Type hint class attributes:**
   ```python
   class ImageProcessor:
       """Process and transform images.
       
       Attributes:
           cache_enabled: Whether to cache processed images.
           max_cache_size: Maximum number of cached images.
       """
       
       cache_enabled: bool
       max_cache_size: int
       
       def __init__(self, cache_enabled: bool = True, max_cache_size: int = 100):
           self.cache_enabled = cache_enabled
           self.max_cache_size = max_cache_size
   ```

### Type Hints in Docstrings

**Do NOT duplicate type hints in docstrings unless adding context:**

```python
# ❌ BAD: Redundant type information
def calculate(value: int) -> int:
    """Calculate result.
    
    Args:
        value (int): Input value.  # Type already in signature!
    
    Returns:
        int: Result.  # Type already in signature!
    """

# ✅ GOOD: Add context, not redundant types
def calculate(value: int) -> int:
    """Calculate result.
    
    Args:
        value: Input value. Must be positive.
    
    Returns:
        Calculated result, always non-negative.
    """
```

---

## Modern Tooling: Ruff and UV

### Modern Python Project Stack

Modern Python projects (2024-2026) use:
- **Ruff**: Ultra-fast linter and formatter (replaces Black, Flake8, isort, pydocstyle)
- **UV**: Fast package manager and dependency resolver
- **Type checking**: mypy or pyright

### Ruff Configuration for Documentation

**Ruff handles both code formatting AND docstring linting.**

Create or update `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
# Enable docstring linting
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "D",      # pydocstyle (docstring linting)
    "UP",     # pyupgrade
    "ANN",    # flake8-annotations (type hints)
]

# Docstring rules (pydocstyle)
[tool.ruff.lint.pydocstyle]
convention = "google"  # Enforce Google-style docstrings

[tool.ruff.lint.per-file-ignores]
# Don't require docstrings in test files
"tests/**/*.py" = ["D"]
# Allow missing docstrings in __init__.py if it's just imports
"**/__init__.py" = ["D104"]

[tool.ruff.format]
# Use Ruff's formatter
quote-style = "double"
indent-style = "space"
docstring-code-format = true  # Format code examples in docstrings
```

### Running Ruff

```bash
# Lint code and docstrings
ruff check .

# Auto-fix issues where possible
ruff check --fix .

# Format code (including docstring examples)
ruff format .

# Check everything in CI
ruff check . && ruff format --check .
```

### UV Package Management

UV is the modern replacement for pip/pip-tools/poetry:

```bash
# Install dependencies
uv pip install -r requirements.txt

# Add a dependency
uv add sphinx sphinx-rtd-theme

# Sync environment with lock file
uv sync
```

### Documentation Generation

**Sphinx with Napoleon (for Google-style docstrings):**

```bash
# Install with UV
uv add --dev sphinx sphinx-autodoc-typehints sphinx-rtd-theme

# Initialize Sphinx
sphinx-quickstart docs

# Build documentation
cd docs && make html
```

**Minimal `docs/conf.py` for Google-style:**
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy style support
    'sphinx_autodoc_typehints',  # Type hint support
]

# Napoleon settings for Google style
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
```

### Verification in CI

**Integrate all checks in continuous integration:**

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Install dependencies
        run: uv sync
      
      - name: Lint with Ruff
        run: uv run ruff check .
      
      - name: Format check with Ruff
        run: uv run ruff format --check .
      
      - name: Type check
        run: uv run mypy src/
      
      - name: Build documentation
        run: |
          cd docs
          uv run make html
```

---

## Updating Existing Documentation

When working with projects that already have documentation, **always verify and update** rather than blindly trusting existing content.

### Verification Workflow

1. **Check for completeness:**
   - [ ] All public functions have docstrings
   - [ ] All parameters are documented
   - [ ] Return values are documented
   - [ ] All raised exceptions are documented
   - [ ] Type hints are present

2. **Check for accuracy:**
   - [ ] Docstrings match current function signatures
   - [ ] Parameter descriptions match actual parameter names
   - [ ] Return type documentation matches actual return type hint
   - [ ] Exception documentation matches exceptions actually raised

3. **Check for consistency:**
   - [ ] Same docstring style throughout (Google by default)
   - [ ] Consistent terminology for similar concepts
   - [ ] Consistent level of detail

4. **Check for quality:**
   - [ ] No marketing language ("robust", "fast", "powerful")
   - [ ] No implementation details in API documentation
   - [ ] Examples are present and correct
   - [ ] Type hints are comprehensive

### Updating Process

```python
# BEFORE: Outdated documentation
def process_file(path, verbose=False):
    """Process a file."""  # Incomplete!
    with open(path) as f:
        return f.read().upper()

# AFTER: Complete, accurate documentation
def process_file(path: str, verbose: bool = False) -> str:
    """Read file and convert contents to uppercase.
    
    Args:
        path: Path to text file to process.
        verbose: Whether to print progress messages.
    
    Returns:
        File contents converted to uppercase.
    
    Raises:
        FileNotFoundError: If path does not exist.
        PermissionError: If file cannot be read.
        UnicodeDecodeError: If file is not valid UTF-8 text.
    """
    with open(path, encoding='utf-8') as f:
        return f.read().upper()
```

### Migration from Old Styles

If you encounter non-Google style docstrings and user hasn't requested to keep them:

```python
# OLD: Sphinx reST style
def calculate(x, y):
    """
    Calculate sum.
    
    :param x: First number
    :type x: int
    :param y: Second number
    :type y: int
    :return: Sum of x and y
    :rtype: int
    """
    return x + y

# NEW: Google style with type hints
def calculate(x: int, y: int) -> int:
    """Calculate sum of two numbers.
    
    Args:
        x: First number.
        y: Second number.
    
    Returns:
        Sum of x and y.
    """
    return x + y
```

---

## Tools and Integration

---

## Complete Example: Modern Python Documentation

Here's a complete example showing all best practices for a modern Python project (2026):

```python
"""Data processing utilities for CSV and JSON files.

This module provides functions for loading, transforming, and saving
structured data in various formats. It handles common data cleaning
tasks and validation.

Exports:
    load_data: Load data from CSV or JSON files.
    clean_data: Remove invalid entries and normalize values.
    save_data: Save processed data to file.
    DataFormat: Enumeration of supported file formats.
    MAX_FILE_SIZE: Maximum allowed file size in bytes.
"""

from enum import Enum
from pathlib import Path
from typing import TypeAlias

import pandas as pd

__all__ = ['load_data', 'clean_data', 'save_data', 'DataFormat', 'MAX_FILE_SIZE']

#: Maximum file size for data imports, in bytes.
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Type alias for complex type
DataDict: TypeAlias = dict[str, list[str | int | float]]


class DataFormat(Enum):
    """Supported data file formats."""
    
    CSV = "csv"
    JSON = "json"


def load_data(
    file_path: str | Path,
    format: DataFormat | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load data from a file into a DataFrame.
    
    Automatically detects format from file extension if not specified.
    Supports CSV and JSON formats.
    
    Args:
        file_path: Path to the data file.
        format: File format. If None, detected from file extension.
        max_rows: Maximum number of rows to load. If None, loads all rows.
    
    Returns:
        DataFrame containing the loaded data.
    
    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file format is unsupported or cannot be detected.
        pd.errors.ParserError: If file is malformed.
    
    Example:
        >>> df = load_data('data.csv', max_rows=1000)
        >>> len(df)
        1000
        
        >>> df = load_data('data.json', format=DataFormat.JSON)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect format if not specified
    if format is None:
        suffix = path.suffix.lower()
        if suffix == '.csv':
            format = DataFormat.CSV
        elif suffix == '.json':
            format = DataFormat.JSON
        else:
            raise ValueError(f"Cannot detect format for: {suffix}")
    
    # Load based on format
    if format == DataFormat.CSV:
        return pd.read_csv(path, nrows=max_rows)
    else:  # JSON
        return pd.read_json(path, nrows=max_rows)


def clean_data(
    data: pd.DataFrame,
    remove_nulls: bool = True,
    normalize_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Remove invalid entries and normalize specified columns.
    
    Args:
        data: DataFrame to clean.
        remove_nulls: Whether to remove rows with any null values.
        normalize_columns: Column names to normalize to [0, 1] range.
            If None, no normalization is performed.
    
    Returns:
        Cleaned DataFrame. Original DataFrame is not modified.
    
    Example:
        >>> df = pd.DataFrame({'a': [1, 2, None], 'b': [10, 20, 30]})
        >>> clean = clean_data(df, remove_nulls=True)
        >>> len(clean)
        2
    """
    result = data.copy()
    
    if remove_nulls:
        result = result.dropna()
    
    if normalize_columns:
        for col in normalize_columns:
            min_val = result[col].min()
            max_val = result[col].max()
            # Avoid division by zero for constant columns
            if max_val > min_val:
                result[col] = (result[col] - min_val) / (max_val - min_val)
    
    return result


class DataProcessor:
    """Process and transform structured data.
    
    This class provides a stateful interface for processing data with
    configurable cleaning and transformation options. It maintains
    processing history for debugging.
    
    Attributes:
        auto_clean: Whether to automatically clean data on load.
        processing_history: List of operations performed, for debugging.
    
    Example:
        >>> processor = DataProcessor(auto_clean=True)
        >>> df = processor.load('data.csv')
        >>> processor.normalize(['price', 'quantity'])
        >>> processor.save('cleaned.csv')
    """
    
    auto_clean: bool
    processing_history: list[str]
    _data: pd.DataFrame | None
    
    def __init__(self, auto_clean: bool = False):
        """Initialize processor.
        
        Args:
            auto_clean: Whether to automatically remove nulls when loading data.
        """
        self.auto_clean = auto_clean
        self.processing_history = []
        self._data = None
    
    def load(self, file_path: str | Path) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            file_path: Path to data file.
        
        Returns:
            Loaded DataFrame.
        
        Raises:
            FileNotFoundError: If file does not exist.
        """
        self._data = load_data(file_path)
        self.processing_history.append(f"Loaded {file_path}")
        
        if self.auto_clean:
            self._data = clean_data(self._data)
            self.processing_history.append("Auto-cleaned data")
        
        return self._data
    
    @property
    def data(self) -> pd.DataFrame:
        """Get current data.
        
        Returns:
            Current DataFrame being processed.
        
        Raises:
            ValueError: If no data has been loaded yet.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load() first.")
        return self._data
```

### What This Example Demonstrates

✅ **Type hints everywhere** — All parameters, returns, attributes  
✅ **Google-style docstrings** — Clean, readable, Ruff-compliant  
✅ **Modern type syntax** — `list[str]`, not `List[str]`  
✅ **Complete documentation** — Args, Returns, Raises, Examples  
✅ **No marketing language** — Just facts about what code does  
✅ **Module-level docs** — Purpose, exports clearly stated  
✅ **TypeAlias for complex types** — Makes signatures cleaner  
✅ **Property documentation** — When behavior is non-trivial  
✅ **Exception documentation** — All possible errors listed  
✅ **Inline comments** — Only where needed (normalization edge case)

This will pass all Ruff checks with:
```toml
[tool.ruff.lint]
select = ["D", "ANN", "E", "F", "I", "N", "UP"]

[tool.ruff.lint.pydocstyle]
convention = "google"
```

---

## Common Ruff Docstring Errors and Fixes

When running `ruff check --select D`, you may encounter these common issues:

### D100: Missing docstring in public module
```python
# ❌ Missing module docstring
import pandas as pd

# ✅ Add module docstring at top
"""Data processing utilities."""

import pandas as pd
```

### D103: Missing docstring in public function
```python
# ❌ No docstring
def process(data):
    return data.upper()

# ✅ Add docstring
def process(data: str) -> str:
    """Convert data to uppercase."""
    return data.upper()
```

### D417: Missing argument description
```python
# ❌ Parameter not documented
def greet(name: str, title: str = "Mr.") -> str:
    """Greet a person.
    
    Args:
        name: Person's name.
        # Missing: title parameter!
    """

# ✅ All parameters documented
def greet(name: str, title: str = "Mr.") -> str:
    """Greet a person.
    
    Args:
        name: Person's name.
        title: Title to use in greeting.
    """
```

### D401: First line should be in imperative mood
```python
# ❌ Non-imperative summary
def calculate_sum(a: int, b: int) -> int:
    """This function calculates the sum."""
    
# ✅ Imperative mood
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
```



### Package `__init__.py`
Document the package-level API and what users should import:
```python
"""Image processing package.

This package provides tools for loading, transforming, and saving images.

Public API:
    ImageProcessor: Main class for image operations.
    load_image: Load an image from file.
    save_image: Save an image to file.
    
Example:
    >>> from imagetools import ImageProcessor
    >>> processor = ImageProcessor()
"""
```

### Properties and Descriptors
Document properties like methods, explaining what they return:
```python
@property
def area(self) -> float:
    """Calculate and return the area.
    
    Returns:
        Area in square units.
    """
    return self.width * self.height
```

### Abstract Base Classes
Document the contract that subclasses must fulfill:
```python
class Shape(ABC):
    """Abstract base class for geometric shapes.
    
    Subclasses must implement the area() method to calculate
    the shape's area based on its specific dimensions.
    """
    
    @abstractmethod
    def area(self) -> float:
        """Calculate the area of this shape.
        
        Returns:
            Area in square units.
        """
```

### Decorators
Document what the decorator does to the wrapped function:
```python
def retry(max_attempts: int = 3):
    """Retry a function on exception.
    
    Args:
        max_attempts: Maximum number of retry attempts.
    
    Returns:
        Decorator that wraps a function with retry logic.
    
    Example:
        @retry(max_attempts=5)
        def fetch_data():
            ...
    """
```

---

## Summary of Priorities

| Priority     | Action                                                                           |
| ------------ | -------------------------------------------------------------------------------- |
| **Critical** | Add/verify type hints for all public functions and methods                       |
| **Critical** | Verify and update existing documentation for accuracy                            |
| **Highest**  | Document all public API (functions, classes, modules, constants) in Google style |
| **High**     | Include all parameters, returns, and exceptions (check actual code!)             |
| **Medium**   | Add realistic examples for complex APIs                                          |
| **Low**      | Add inline comments only for non-obvious *why* explanations                      |
| **Never**    | Use marketing language or document obvious implementation details                |

---

## Final Note for AI Agents

### Documentation Workflow Priority

1. **Type hints first** — Without proper type hints, documentation is incomplete
2. **Verify existing docs** — Don't trust, verify. Check against actual code
3. **Google style by default** — Unless user explicitly requests otherwise
4. **Validate with Ruff** — Run `ruff check --select D` before finishing
5. **Be precise, not promotional** — Describe exactly what the code does


When in doubt:
1. **Type hints are mandatory** — Add them before writing docstrings
2. **Verify, don't assume** — Check existing docs against actual code
3. **Google style by default** — Unless user explicitly specifies otherwise
4. **Be precise, not promotional** — Describe exactly what the code does
5. **Document the contract** — What users need to know to use the API correctly
6. **Assume readers are competent** — Don't over-explain obvious concepts
7. **Focus on constraints and edge cases** — This is where bugs hide
8. **Validate with Ruff** — Run checks before considering work complete
9. **Keep it factual** — No opinions about code quality in documentation

The best documentation is **accurate, complete, concise, and boring**. If it reads like a reference manual rather than a sales pitch, you're doing it right.

---

## Modern Python Documentation Stack (2026)

For reference, here's the complete modern Python documentation toolchain:

```toml
# pyproject.toml - Modern Python project configuration
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"

[tool.uv]
dev-dependencies = [
    "ruff>=0.6.0",
    "mypy>=1.13.0",
    "sphinx>=8.0.0",
    "sphinx-rtd-theme>=3.0.0",
    "sphinx-autodoc-typehints>=2.0.0",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["D", "E", "F", "I", "N", "UP", "ANN"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.mypy]
strict = true
warn_return_any = true
warn_unused_configs = true
```

**Complete workflow:**
```bash
# Install dependencies
uv sync

# Format code
uv run ruff format .

# Lint code and docs
uv run ruff check .

# Type check
uv run mypy src/

# Build documentation
cd docs && uv run make html

# Run all in CI
uv run ruff format --check . && \
uv run ruff check . && \
uv run mypy src/ && \
cd docs && uv run make html
```

This ensures your Python documentation is modern, validated, and maintainable.

