# The Complete Python Project Guide (2026 Edition)

> A comprehensive reference for starting, structuring, and maintaining Python projects using modern tooling.

---

## Table of Contents

1.  [Philosophy & Core Principles](#1-philosophy--core-principles)
2.  [Toolchain Setup](#2-toolchain-setup)
3.  [VS Code Configuration](#3-vs-code-configuration)
4.  [Creating a New Project (Step-by-Step)](#4-creating-a-new-project-step-by-step)
5.  [Project Structures by Application Type](#5-project-structures-by-application-type)
6.  [Configuration Files Reference](#6-configuration-files-reference)
7.  [Best Practices & Tips](#7-best-practices--tips)
8.  [Common Commands Cheat Sheet](#8-common-commands-cheat-sheet)

---

## 1. Philosophy & Core Principles

### The Guiding Principles of Modern Python (2026)

| Principle           | Description                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| **Consolidation**   | Use fewer, more powerful tools. One tool should do one job exceptionally well.                        |
| **Speed**           | Rust-based tools (uv, Ruff) are the new standard. Slow tooling is a productivity tax.                 |
| **Reproducibility** | Lockfiles are mandatory. `uv.lock` or `poetry.lock` ensures identical environments everywhere.        |
| **Src Layout**      | All code lives inside `src/<package_name>/`. This prevents import confusion and keeps the root clean. |
| **Single Config**   | `pyproject.toml` is the single source of truth for dependencies, tool settings, and project metadata. |

### The 2026 Standard Toolchain

| Category                  | Tool             | Replaces                                   |
| ------------------------- | ---------------- | ------------------------------------------ |
| **Package & Env Manager** | `uv`             | pip, pip-tools, venv, pyenv, pipx          |
| **Linter & Formatter**    | `Ruff`           | Flake8, Black, isort, pyupgrade, autoflake |
| **Type Checker (Editor)** | `Pylance`        | Pyright (standalone)                       |
| **Type Checker (CI/CD)**  | `MyPy`           | —                                          |
| **Testing**               | `Pytest`         | unittest                                   |
| **Task Runner**           | `Just` or `Make` | Shell scripts                              |

---

## 2. Toolchain Setup

### 2.1 Install `uv` (The Foundation)

`uv` manages Python versions, virtual environments, and dependencies. Install it once, globally.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Verify Installation:**
```bash
uv --version
```

### 2.2 Install Python via `uv`

You no longer need to install Python from python.org or use `pyenv`.

```bash
# Install the latest stable Python
uv python install 3.12

# List installed versions
uv python list

# Install a specific version
uv python install 3.11.9
```

### 2.3 Install Global CLI Tools via `uv tool`

For tools you want available everywhere (not per-project), use `uv tool`:

```bash
# Install Ruff globally for use anywhere
uv tool install ruff

# Other useful global tools
uv tool install pre-commit
uv tool install cookiecutter
```

---

## 3. VS Code Configuration

### 3.1 Required Extensions

Install these extensions (Ctrl+Shift+X):

| Extension            | Publisher       | Purpose                                   |
| -------------------- | --------------- | ----------------------------------------- |
| **Python**           | Microsoft       | Core language support, debugging, Pylance |
| **Ruff**             | Astral Software | Linting and formatting                    |
| **Even Better TOML** | tamasfe         | `pyproject.toml` syntax highlighting      |
| **Error Lens**       | usernamehw      | Inline error/warning display              |
| **GitLens**          | GitKraken       | Enhanced Git integration                  |

### 3.2 Recommended User Settings

Open settings: `Ctrl+Shift+P` → `Preferences: Open User Settings (JSON)`

```json
{
    // === Python Core ===
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        },
        "editor.rulers": [88, 100]
    },

    // === Pylance (Type Checking) ===
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "python.analysis.inlayHints.functionReturnTypes": true,
    "python.analysis.inlayHints.variableTypes": true,

    // === Ruff ===
    "ruff.lint.run": "onSave",
    "ruff.organizeImports": true,

    // === Terminal ===
    "python.terminal.activateEnvironment": true,

    // === Files ===
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/.ruff_cache": true,
        "**/*.egg-info": true
    },

    // === Editor General ===
    "editor.bracketPairColorization.enabled": true,
    "editor.guides.bracketPairs": "active"
}
```

### 3.3 Workspace Settings (Per-Project)

Create `.vscode/settings.json` in your project root:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.extraPaths": ["${workspaceFolder}/src"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

### 3.4 Launch Configuration for Debugging

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {"PYTHONPATH": "${workspaceFolder}/src"}
        },
        {
            "name": "Python: Module",
            "type": "debugpy",
            "request": "launch",
            "module": "my_package.main",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {"PYTHONPATH": "${workspaceFolder}/src"}
        },
        {
            "name": "Python: FastAPI",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": ["app.main:app", "--reload", "--port", "8000"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/src"
        }
    ]
}
```

---

## 4. Creating a New Project (Step-by-Step)

### 4.1 Using `uv init` (Recommended)

```bash
# Create a new project directory
mkdir my-project && cd my-project

# Initialize with uv (creates pyproject.toml and basic structure)
uv init

# Pin Python version
uv python pin 3.12

# Create the src layout manually (uv init creates flat layout by default)
mkdir -p src/my_project tests
touch src/my_project/__init__.py
touch src/my_project/main.py
touch tests/__init__.py
touch tests/conftest.py

# Create virtual environment and sync
uv sync
```

### 4.2 Using Poetry (Alternative)

```bash
# Create new project with src layout
poetry new my-project --src
cd my-project

# Set Python version
poetry env use 3.12

# Install dependencies
poetry install
```

### 4.3 Manual Setup (Full Control)

```bash
# Create project structure
mkdir -p my-project/{src/my_project,tests,docs}
cd my-project

# Create essential files
touch src/my_project/__init__.py
touch src/my_project/main.py
touch tests/__init__.py
touch tests/conftest.py
touch README.md
touch .gitignore
touch .python-version

# Create pyproject.toml (see Section 6 for content)
touch pyproject.toml

# Initialize git
git init

# Create virtual environment with uv
uv venv
uv sync
```

---

## 5. Project Structures by Application Type

### 5.1 Python Library

**Use Case:** Reusable packages, PyPI distribution, internal shared libraries.

```
my-library/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Run tests on PR
│       └── publish.yml            # Publish to PyPI on release
│
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── api-reference.md
│   └── mkdocs.yml                 # If using MkDocs
│
├── examples/                      # Usage examples
│   └── basic_usage.py
│
├── src/
│   └── my_library/
│       ├── __init__.py            # Public API exports
│       ├── py.typed               # PEP 561 marker (typed package)
│       ├── core.py                # Core functionality
│       ├── exceptions.py          # Custom exceptions
│       ├── types.py               # Type definitions, protocols
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures
│   ├── test_core.py
│   └── test_utils.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version                # e.g., "3.12"
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

**Key `__init__.py` Pattern:**
```python
"""My Library - A brief description."""

from my_library.core import MainClass, main_function
from my_library.exceptions import MyLibraryError

__version__ = "1.0.0"
__all__ = ["MainClass", "main_function", "MyLibraryError"]
```

---

### 5.2 Web Service / API (FastAPI)

**Use Case:** REST APIs, microservices, backend services.

```
my-service/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
│
├── deployments/                   # Infrastructure configs
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── kubernetes/
│       └── deployment.yaml
│
├── migrations/                    # Database migrations (Alembic)
│   ├── versions/
│   │   └── 001_initial.py
│   ├── env.py
│   └── script.py.mako
│
├── scripts/                       # Utility scripts
│   ├── seed_db.py
│   └── generate_keys.py
│
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py                # FastAPI app factory
│       │
│       ├── api/                   # HTTP Layer (Controllers)
│       │   ├── __init__.py
│       │   ├── deps.py            # Dependency injection
│       │   ├── middleware.py      # Custom middleware
│       │   └── v1/
│       │       ├── __init__.py
│       │       ├── router.py      # Aggregates all v1 routes
│       │       └── endpoints/
│       │           ├── __init__.py
│       │           ├── auth.py
│       │           ├── users.py
│       │           └── health.py
│       │
│       ├── core/                  # Application Core
│       │   ├── __init__.py
│       │   ├── config.py          # Settings (Pydantic BaseSettings)
│       │   ├── security.py        # JWT, hashing, auth
│       │   ├── exceptions.py      # Custom HTTP exceptions
│       │   └── logging.py         # Logging configuration
│       │
│       ├── db/                    # Database Layer
│       │   ├── __init__.py
│       │   ├── session.py         # Engine, SessionLocal
│       │   ├── base.py            # Declarative base
│       │   └── repositories/      # Data access patterns
│       │       ├── __init__.py
│       │       └── user_repo.py
│       │
│       ├── models/                # ORM Models (SQLAlchemy)
│       │   ├── __init__.py
│       │   ├── base.py            # Shared model mixins
│       │   └── user.py
│       │
│       ├── schemas/               # Pydantic Schemas (DTOs)
│       │   ├── __init__.py
│       │   ├── common.py          # Shared schemas
│       │   └── user.py            # UserCreate, UserRead, UserUpdate
│       │
│       └── services/              # Business Logic Layer
│           ├── __init__.py
│           ├── base.py            # Base service class
│           └── user_service.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Test DB, client fixtures
│   ├── factories.py               # Test data factories
│   ├── api/
│   │   └── test_users.py
│   ├── services/
│   │   └── test_user_service.py
│   └── integration/
│       └── test_db.py
│
├── .env.example                   # Template (committed)
├── .env                           # Local secrets (NOT committed)
├── .gitignore
├── .python-version
├── alembic.ini
├── pyproject.toml
├── README.md
└── uv.lock
```

**Layer Responsibilities:**

| Layer              | Responsibility                            | Depends On           |
| ------------------ | ----------------------------------------- | -------------------- |
| `api/endpoints/`   | HTTP request/response handling            | services, schemas    |
| `schemas/`         | Data validation, serialization            | —                    |
| `services/`        | Business logic, orchestration             | repositories, models |
| `db/repositories/` | Database queries, CRUD                    | models               |
| `models/`          | ORM table definitions                     | —                    |
| `core/`            | Configuration, security, shared utilities | —                    |

---

### 5.3 CLI Application

**Use Case:** Command-line tools, DevOps utilities, automation scripts.

```
my-cli/
│
├── .github/
│   └── workflows/
│       └── release.yml            # Build binaries on release
│
├── completions/                   # Shell completions
│   ├── my-cli.bash
│   ├── my-cli.zsh
│   └── my-cli.fish
│
├── src/
│   └── my_cli/
│       ├── __init__.py
│       ├── __main__.py            # python -m my_cli
│       ├── main.py                # Typer app definition
│       │
│       ├── commands/              # Command groups
│       │   ├── __init__.py
│       │   ├── init.py            # my-cli init
│       │   ├── config.py          # my-cli config [get|set]
│       │   └── run.py             # my-cli run
│       │
│       ├── core/                  # Core logic
│       │   ├── __init__.py
│       │   ├── config.py          # Config file loading/saving
│       │   ├── exceptions.py
│       │   └── context.py         # CLI context object
│       │
│       ├── services/              # Business logic
│       │   ├── __init__.py
│       │   └── processor.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── console.py         # Rich console, spinners, tables
│           ├── paths.py           # XDG paths, app directories
│           └── validators.py      # Input validation
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_commands/
│   │   └── test_init.py
│   └── test_cli.py                # Integration tests (CLI runner)
│
├── .gitignore
├── .python-version
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

**Key Files:**

**`src/my_cli/__main__.py`:**
```python
"""Allow running as python -m my_cli."""
from my_cli.main import app

if __name__ == "__main__":
    app()
```

**`src/my_cli/main.py`:**
```python
"""CLI entry point."""
import typer

from my_cli.commands import config, init, run

app = typer.Typer(
    name="my-cli",
    help="My CLI tool description.",
    no_args_is_help=True,
)

app.add_typer(init.app, name="init")
app.add_typer(config.app, name="config")
app.add_typer(run.app, name="run")

if __name__ == "__main__":
    app()
```

**`src/my_cli/utils/paths.py`:**
```python
"""Cross-platform path utilities."""
import os
from pathlib import Path

def get_config_dir() -> Path:
    """Get XDG-compliant config directory."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("APPDATA", "~"))
    else:  # macOS/Linux
        base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config"))
    
    config_dir = base.expanduser() / "my-cli"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def get_config_file() -> Path:
    """Get the main config file path."""
    return get_config_dir() / "config.toml"
```

---

### 5.4 Desktop Application (PyQt6 / PySide6)

**Use Case:** GUI applications, cross-platform desktop tools.

**Architecture:** Model-View-ViewModel (MVVM)

```
my-desktop-app/
│
├── .github/
│   └── workflows/
│       └── build.yml              # Build executables
│
├── build/                         # Build outputs (gitignored)
│
├── build_scripts/
│   ├── build.py                   # Build automation script
│   ├── app.spec                   # PyInstaller spec
│   └── nuitka_config.py           # Nuitka configuration
│
├── installer/                     # Installer configs
│   ├── windows/
│   │   └── installer.nsi          # NSIS script
│   └── macos/
│       └── Info.plist
│
├── src/
│   └── my_app/
│       ├── __init__.py
│       ├── __main__.py            # python -m my_app
│       ├── main.py                # Application bootstrap
│       │
│       ├── assets/                # Static resources
│       │   ├── __init__.py
│       │   ├── resources.py       # Asset loading utilities
│       │   ├── icons/
│       │   │   ├── app_icon.ico
│       │   │   ├── app_icon.icns
│       │   │   └── app_icon.png
│       │   ├── images/
│       │   │   └── splash.png
│       │   ├── fonts/
│       │   │   └── Inter.ttf
│       │   └── styles/
│       │       ├── light.qss
│       │       └── dark.qss
│       │
│       ├── core/                  # Application core
│       │   ├── __init__.py
│       │   ├── config.py          # App settings
│       │   ├── constants.py       # App-wide constants
│       │   ├── exceptions.py
│       │   ├── signals.py         # Global signals/events
│       │   └── state.py           # Application state management
│       │
│       ├── models/                # Data Layer (M in MVVM)
│       │   ├── __init__.py
│       │   ├── database.py        # SQLite/DB logic
│       │   ├── entities.py        # Data classes
│       │   └── repositories/
│       │       ├── __init__.py
│       │       └── project_repo.py
│       │
│       ├── services/              # Business Logic
│       │   ├── __init__.py
│       │   ├── file_service.py
│       │   ├── export_service.py
│       │   └── api_client.py
│       │
│       ├── viewmodels/            # Logic Layer (VM in MVVM)
│       │   ├── __init__.py
│       │   ├── base.py            # Base ViewModel class
│       │   ├── main_vm.py
│       │   └── settings_vm.py
│       │
│       └── views/                 # UI Layer (V in MVVM)
│           ├── __init__.py
│           ├── main_window.py     # Main window
│           ├── dialogs/           # Modal dialogs
│           │   ├── __init__.py
│           │   ├── settings_dialog.py
│           │   └── about_dialog.py
│           ├── widgets/           # Reusable components
│           │   ├── __init__.py
│           │   ├── sidebar.py
│           │   ├── toolbar.py
│           │   └── status_bar.py
│           └── pages/             # Main content pages
│               ├── __init__.py
│               ├── home_page.py
│               └── editor_page.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_viewmodels/
│   └── test_services/
│
├── .gitignore
├── .python-version
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

**Key Implementation Files:**

**`src/my_app/assets/resources.py`:**
```python
"""Asset loading utilities that work in dev and compiled modes."""
import sys
from pathlib import Path
from functools import lru_cache

def _get_assets_dir() -> Path:
    """Get the assets directory, handling PyInstaller bundling."""
    if hasattr(sys, '_MEIPASS'):
        # Running as compiled executable
        return Path(sys._MEIPASS) / "assets"
    else:
        # Running in development
        return Path(__file__).parent


@lru_cache(maxsize=128)
def get_asset_path(relative_path: str) -> Path:
    """Get absolute path to an asset file."""
    return _get_assets_dir() / relative_path


def get_icon(name: str) -> Path:
    """Get path to an icon file."""
    return get_asset_path(f"icons/{name}")


def get_stylesheet(name: str) -> str:
    """Load a QSS stylesheet as string."""
    path = get_asset_path(f"styles/{name}.qss")
    return path.read_text(encoding="utf-8")
```

**`src/my_app/viewmodels/base.py`:**
```python
"""Base ViewModel class with Qt signals integration."""
from PySide6.QtCore import QObject, Signal


class BaseViewModel(QObject):
    """Base class for all ViewModels."""
    
    # Common signals
    loading_changed = Signal(bool)
    error_occurred = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_loading = False
    
    @property
    def is_loading(self) -> bool:
        return self._is_loading
    
    @is_loading.setter
    def is_loading(self, value: bool):
        if self._is_loading != value:
            self._is_loading = value
            self.loading_changed.emit(value)
```

**`src/my_app/main.py`:**
```python
"""Application entry point."""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from my_app.core.config import Settings
from my_app.assets.resources import get_icon, get_stylesheet
from my_app.views.main_window import MainWindow


def main() -> int:
    """Initialize and run the application."""
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("My App")
    app.setOrganizationName("MyCompany")
    app.setWindowIcon(QIcon(str(get_icon("app_icon.png"))))
    
    # Load settings and apply theme
    settings = Settings.load()
    app.setStyleSheet(get_stylesheet(settings.theme))
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

---

## 6. Configuration Files Reference

### 6.1 Complete `pyproject.toml` Template

```toml
# ============================================================
# PROJECT METADATA
# ============================================================
[project]
name = "my-project"
version = "0.1.0"
description = "A brief description of the project."
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.12"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = ["python", "example"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
]

# Main dependencies
dependencies = [
    "pydantic>=2.7.0",
    "httpx>=0.27.0",
]

# Optional dependency groups (pip install my-project[dev])
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=0.23.0",
]
docs = [
    "mkdocs>=1.6.0",
    "mkdocs-material>=9.5.0",
]

# CLI entry points
[project.scripts]
my-cli = "my_project.main:app"

# GUI entry points (no console window on Windows)
[project.gui-scripts]
my-gui = "my_project.main:main"

[project.urls]
Homepage = "https://github.com/username/my-project"
Documentation = "https://username.github.io/my-project"
Repository = "https://github.com/username/my-project"
Issues = "https://github.com/username/my-project/issues"

# ============================================================
# UV DEPENDENCY GROUPS (for uv only)
# ============================================================
[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "mypy>=1.10.0",
    "ruff>=0.4.0",
    "pre-commit>=3.7.0",
]
docs = [
    "mkdocs>=1.6.0",
    "mkdocs-material>=9.5.0",
]

# ============================================================
# BUILD SYSTEM
# ============================================================
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/my_project"]

# ============================================================
# RUFF (Linting & Formatting)
# ============================================================
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "ERA",    # eradicate (commented code)
    "PL",     # Pylint
    "RUF",    # Ruff-specific rules
]
ignore = [
    "E501",   # line too long (handled by formatter)
    "PLR0913", # too many arguments
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "ARG", "PLR2004"]
"**/__init__.py" = ["F401"]

[tool.ruff.lint.isort]
known-first-party = ["my_project"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

# ============================================================
# MYPY (Type Checking)
# ============================================================
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
show_error_codes = true
show_column_numbers = true

# Per-module overrides
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = [
    "some_untyped_library.*",
]
ignore_missing_imports = true

# ============================================================
# PYTEST
# ============================================================
[tool.pytest.ini_options]
minversion = "8.0"
pythonpath = ["src"]
testpaths = ["tests"]
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--strict-config",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
]

# ============================================================
# COVERAGE
# ============================================================
[tool.coverage.run]
source = ["src"]
branch = true
parallel = true
omit = [
    "**/__init__.py",
    "**/conftest.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
    "@abstractmethod",
]
fail_under = 80
show_missing = true
```

### 6.2 Standard `.gitignore`

```gitignore
# ============================================================
# PYTHON
# ============================================================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# ============================================================
# VIRTUAL ENVIRONMENTS
# ============================================================
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# ============================================================
# IDE
# ============================================================
.idea/
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json
*.swp
*.swo
*~

# ============================================================
# TESTING & COVERAGE
# ============================================================
.tox/
.nox/
.coverage
.coverage.*
htmlcov/
.pytest_cache/
.hypothesis/
coverage.xml
*.cover
*.py,cover

# ============================================================
# TYPE CHECKING & LINTING
# ============================================================
.mypy_cache/
.dmypy.json
dmypy.json
.ruff_cache/
.pytype/

# ============================================================
# DOCUMENTATION
# ============================================================
docs/_build/
site/

# ============================================================
# SECRETS & LOCAL CONFIG
# ============================================================
.env
.env.local
.env.*.local
*.pem
*.key
secrets.yaml
secrets.json

# ============================================================
# OS
# ============================================================
.DS_Store
Thumbs.db

# ============================================================
# PROJECT SPECIFIC
# ============================================================
*.log
logs/
tmp/
temp/
*.sqlite
*.db
```

### 6.3 Pre-commit Configuration (`.pre-commit-config.yaml`)

```yaml
# See https://pre-commit.com for more information
repos:
  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key
      - id: no-commit-to-branch
        args: ['--branch', 'main', '--branch', 'master']

  # Ruff - Linting and Formatting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # MyPy - Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.0]
        args: [--config-file=pyproject.toml]

  # Security checks
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  # Commit message format
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.25.0
    hooks:
      - id: commitizen
        stages: [commit-msg]
```

---

## 7. Best Practices & Tips

### 7.1 Code Organization Principles

| Principle                 | Description                                                        |
| ------------------------- | ------------------------------------------------------------------ |
| **Single Responsibility** | Each module/class should have one reason to change                 |
| **Dependency Injection**  | Pass dependencies as parameters; don't hard-code imports           |
| **Interface Segregation** | Use Protocols (typing) to define contracts between layers          |
| **Explicit Exports**      | Use `__all__` in `__init__.py` to define public API                |
| **Fail Fast**             | Validate inputs early; raise exceptions rather than returning None |

### 7.2 Typing Best Practices

```python
# ✅ DO: Use modern typing syntax (Python 3.12+)
def process_items(items: list[str]) -> dict[str, int]:
    ...

# ✅ DO: Use | instead of Union
def get_user(id: int) -> User | None:
    ...

# ✅ DO: Use Protocols for duck typing
from typing import Protocol

class Repository(Protocol):
    def get(self, id: int) -> Model | None: ...
    def save(self, entity: Model) -> None: ...

# ✅ DO: Type class attributes
class Config:
    debug: bool = False
    database_url: str

# ❌ DON'T: Use Any unless absolutely necessary
# ❌ DON'T: Ignore type errors with # type: ignore without explanation
```

### 7.3 Import Organization

Ruff/isort will handle this automatically, but understand the convention:

```python
"""Module docstring."""

# 1. Future imports (rarely needed in 3.12+)
from __future__ import annotations

# 2. Standard library
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

# 3. Third-party
import httpx
from pydantic import BaseModel

# 4. Local application
from my_project.core import config
from my_project.utils import helpers

# 5. Type-checking only imports (avoid circular imports)
if TYPE_CHECKING:
    from my_project.models import User
```

### 7.4 Exception Handling

```python
# ✅ Define custom exceptions in a dedicated module
# src/my_project/exceptions.py

class MyProjectError(Exception):
    """Base exception for the project."""

class ValidationError(MyProjectError):
    """Raised when validation fails."""

class NotFoundError(MyProjectError):
    """Raised when a resource is not found."""

class ExternalServiceError(MyProjectError):
    """Raised when an external service fails."""
    
    def __init__(self, service: str, message: str):
        self.service = service
        super().__init__(f"{service}: {message}")
```

### 7.5 Configuration Management

```python
# ✅ Use Pydantic Settings for configuration
# src/my_project/core/config.py

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # App
    app_name: str = "My Project"
    debug: bool = False
    log_level: str = "INFO"
    
    # Database
    database_url: str = Field(alias="DATABASE_URL")
    
    # External Services
    api_key: str = Field(default="", alias="API_KEY")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

### 7.6 Testing Patterns

```python
# tests/conftest.py
import pytest
from my_project.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Override settings for testing."""
    return Settings(
        debug=True,
        database_url="sqlite:///:memory:",
    )


@pytest.fixture
def sample_data() -> dict:
    """Provide sample test data."""
    return {
        "name": "Test User",
        "email": "test@example.com",
    }


# Use factories for complex objects
# tests/factories.py
from dataclasses import dataclass
from my_project.models import User


@dataclass
class UserFactory:
    """Factory for creating test users."""
    
    @staticmethod
    def create(**kwargs) -> User:
        defaults = {
            "name": "Test User",
            "email": "test@example.com",
            "is_active": True,
        }
        return User(**(defaults | kwargs))
```

### 7.7 Logging Setup

```python
# src/my_project/core/logging.py
import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel = "INFO",
    format_string: str | None = None,
) -> None:
    """Configure application logging."""
    
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
        )
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
```

---

## 8. Common Commands Cheat Sheet

### 8.1 UV Commands

```bash
# === Project Setup ===
uv init                          # Initialize new project
uv python install 3.12           # Install Python version
uv python pin 3.12               # Pin version for project
uv venv                          # Create virtual environment
uv sync                          # Install all dependencies

# === Dependency Management ===
uv add requests                  # Add dependency
uv add --dev pytest              # Add dev dependency
uv add "fastapi>=0.110"          # Add with version constraint
uv remove requests               # Remove dependency
uv lock                          # Update lockfile
uv sync --frozen                 # Install from lockfile exactly

# === Running Code ===
uv run python script.py          # Run with project env
uv run pytest                    # Run tests
uv run ruff check .              # Run linter
uv run mypy src                  # Run type checker

# === Tools ===
uv tool install ruff             # Install global tool
uv tool run ruff check .         # Run tool without installing
uvx ruff check .                 # Shorthand for above
```

### 8.2 Ruff Commands

```bash
# === Linting ===
ruff check .                     # Check for errors
ruff check . --fix               # Auto-fix errors
ruff check . --watch             # Watch mode

# === Formatting ===
ruff format .                    # Format all files
ruff format . --check            # Check without changing
ruff format . --diff             # Show what would change

# === Combined ===
ruff check . --fix && ruff format .    # Lint then format
```

### 8.3 Pytest Commands

```bash
# === Basic ===
pytest                           # Run all tests
pytest tests/test_core.py        # Run specific file
pytest -k "test_user"            # Run tests matching pattern
pytest -x                        # Stop on first failure

# === Verbosity ===
pytest -v                        # Verbose output
pytest -vv                       # More verbose
pytest -q                        # Quiet output

# === Coverage ===
pytest --cov=src                 # With coverage
pytest --cov=src --cov-report=html   # HTML report
pytest --cov=src --cov-fail-under=80 # Fail if under 80%

# === Markers ===
pytest -m "not slow"             # Skip slow tests
pytest -m integration            # Only integration tests
```

### 8.4 MyPy Commands

```bash
mypy src                         # Check src directory
mypy src --strict                # Strict mode
mypy src --show-error-codes      # Show error codes
mypy src --ignore-missing-imports    # Ignore missing stubs
```

### 8.5 Git Workflow Commands

```bash
# === Initial Setup ===
git init
git add .
git commit -m "Initial commit"
pre-commit install               # Install pre-commit hooks
pre-commit install --hook-type commit-msg  # Commitizen hook

# === Daily Workflow ===
pre-commit run --all-files       # Run all hooks manually
git add .
git commit -m "feat: add new feature"  # Conventional commit
```

---

## Quick Start Checklist

Use this checklist when starting any new Python project:

```markdown
## New Project Checklist

### Initial Setup
- [ ] Create project directory
- [ ] Run `uv init`
- [ ] Set Python version: `uv python pin 3.12`
- [ ] Create src layout: `mkdir -p src/my_project tests`
- [ ] Create `src/my_project/__init__.py`

### Configuration
- [ ] Configure `pyproject.toml` (dependencies, tools)
- [ ] Create `.gitignore`
- [ ] Create `.python-version`
- [ ] Set up `.pre-commit-config.yaml`

### VS Code
- [ ] Create `.vscode/settings.json`
- [ ] Create `.vscode/launch.json`
- [ ] Verify Python interpreter is detected

### Documentation
- [ ] Write `README.md`
- [ ] Create `LICENSE`
- [ ] Set up `docs/` directory (if applicable)

### Git
- [ ] Initialize: `git init`
- [ ] Install hooks: `pre-commit install`
- [ ] Initial commit

### CI/CD
- [ ] Create `.github/workflows/ci.yml`
- [ ] Add test workflow
- [ ] Add release workflow (if applicable)

### Final Verification
- [ ] Run `uv sync`
- [ ] Run `uv run ruff check .`
- [ ] Run `uv run mypy src`
- [ ] Run `uv run pytest`
```

---

This guide should serve as a complete reference for Python project development in 2026. The key principles are: use `uv` for environment/dependency management, `Ruff` for linting/formatting, follow the src layout, and centralize all configuration in `pyproject.toml`.