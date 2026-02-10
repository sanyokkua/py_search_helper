# Comprehensive Guide to uv: The Modern Python Package Manager

## Table of Contents
1. [What is uv?](#1-what-is-uv)
2. [Installation](#2-installation)
3. [Basic Commands Reference](#3-basic-commands-reference)
4. [Managing Python Versions](#4-managing-python-versions)
5. [Project Initialization and Layouts](#5-project-initialization-and-layouts)
6. [Dependency Management](#6-dependency-management)
7. [Running Code](#7-running-code)
8. [Building the Project](#8-building-the-project)
9. [Packaging and Publishing](#9-packaging-and-publishing)
10. [Workspaces (Monorepos)](#10-workspaces-monorepos)
11. [The Pip Compatibility Interface](#11-the-pip-compatibility-interface)
12. [Tools (CLI Utilities)](#12-tools-cli-utilities)
13. [Comparison with Maven (For Java Developers)](#13-comparison-with-maven-for-java-developers)
14. [Cheat Sheet](#14-cheat-sheet)

---

## 1. What is uv?

**uv** is an extremely fast, all-in-one Python package and project manager written in Rust. Developed by [Astral](https://astral.sh/) (the creators of the `ruff` linter), it is designed to be a single binary that replaces the entire fragmented Python tooling ecosystem.

### Tools it Replaces
| Legacy Tool           | Purpose                   | uv Equivalent                  |
| --------------------- | ------------------------- | ------------------------------ |
| `pip`                 | Package installer         | `uv add`, `uv pip install`     |
| `pip-tools`           | Dependency locking        | `uv lock`                      |
| `virtualenv` / `venv` | Environment creation      | `uv venv`, `uv sync`           |
| `poetry` / `pdm`      | Project management        | `uv init`, `uv add`, `uv lock` |
| `pyenv`               | Python version management | `uv python install`            |
| `pipx`                | Running CLI tools         | `uvx`                          |
| `twine`               | Package publishing        | `uv publish`                   |

### Key Features
*   **Speed:** 10-100x faster than `pip` due to Rust implementation and aggressive caching.
*   **Universal Lockfile:** The `uv.lock` file is cross-platform (Windows, macOS, Linux) by default.
*   **Automatic Environment Management:** No need to manually activate virtual environments.
*   **Python Installation:** Can download and manage Python interpreters itself.
*   **Single Binary:** Installable without needing Python pre-installed.

---

## 2. Installation

You do **not** need Python installed to install `uv`.

### Standalone Installer (Recommended)

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Alternative Methods
```bash
# Via Homebrew (macOS)
brew install uv

# Via pip (if Python exists)
pip install uv

# Via Cargo (Rust)
cargo install --locked uv
```

### Post-Installation
```bash
# Verify installation
uv --version

# Update uv to the latest version
uv self update

# Enable shell autocompletion (e.g., for Bash)
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
```

### Uninstallation
```bash
# Clean up stored data first
uv cache clean
rm -r "$(uv python dir)"
rm -r "$(uv tool dir)"

# Remove binaries (macOS/Linux)
rm ~/.local/bin/uv ~/.local/bin/uvx
```

---

## 3. Basic Commands Reference

### Core Project Commands
| Command           | Description                                     |
| ----------------- | ----------------------------------------------- |
| `uv init <name>`  | Create a new project                            |
| `uv add <pkg>`    | Add a dependency                                |
| `uv remove <pkg>` | Remove a dependency                             |
| `uv sync`         | Install dependencies from lockfile into `.venv` |
| `uv lock`         | Resolve and update the `uv.lock` file           |
| `uv run <cmd>`    | Run a command in the project environment        |
| `uv build`        | Build source distribution and wheel             |
| `uv publish`      | Upload package to PyPI                          |

### Python Management Commands
| Command                     | Description                                         |
| --------------------------- | --------------------------------------------------- |
| `uv python install <ver>`   | Install a Python version                            |
| `uv python list`            | List available/installed Python versions            |
| `uv python find <ver>`      | Find path to a Python executable                    |
| `uv python pin <ver>`       | Pin project to a Python version (`.python-version`) |
| `uv python uninstall <ver>` | Remove a Python version                             |

### Tool Commands (for CLI utilities like `ruff`, `black`)
| Command                  | Description                           |
| ------------------------ | ------------------------------------- |
| `uvx <tool>`             | Run a tool in a temporary environment |
| `uv tool install <tool>` | Install a tool globally               |
| `uv tool list`           | List installed tools                  |

### Utility Commands
| Command          | Description              |
| ---------------- | ------------------------ |
| `uv self update` | Update uv itself         |
| `uv cache clean` | Clear the download cache |
| `uv tree`        | Show the dependency tree |
| `uv version`     | Show project version     |

### Common Flags
| Flag                | Description                                   |
| ------------------- | --------------------------------------------- |
| `--python <ver>`    | Specify a Python version for the command      |
| `--dev`             | Target development dependencies               |
| `--group <name>`    | Target a specific dependency group            |
| `--optional <name>` | Target optional dependencies (extras)         |
| `--locked`          | Fail if lockfile is out of date (CI mode)     |
| `--frozen`          | Use lockfile without checking if it's current |
| `--no-sync`         | Don't sync environment before running         |

---

## 4. Managing Python Versions

`uv` can install, manage, and automatically download Python interpreters ("managed Python"). It also discovers existing system installations.

### Installing Python
```bash
# Install the latest 3.12.x
uv python install 3.12

# Install a specific patch version
uv python install 3.11.6

# Install multiple versions
uv python install 3.10 3.11 3.12

# Install PyPy
uv python install pypy@3.10

# Install free-threaded Python 3.13+
uv python install 3.13t
```

### Viewing Python Versions
```bash
# List installed and available versions
uv python list

# Show only installed versions
uv python list --only-installed

# Find a specific version matching a constraint
uv python find ">=3.11"
```

### Pinning a Version for a Project
This creates a `.python-version` file in the current directory.
```bash
uv python pin 3.11
```
When `uv run` or `uv sync` is executed, `uv` will use this pinned version.

### Upgrading Python Versions (Preview)
```bash
# Upgrade a specific minor version to its latest patch
uv python upgrade 3.12

# Upgrade all installed versions
uv python upgrade
```

### Automatic Downloads
By default, if you request a Python version that isn't installed (e.g., `uv run --python 3.13`), `uv` will download it automatically. This can be disabled:
```bash
# Disable for a single command
uv run --no-python-downloads ...

# Disable globally via config
# In pyproject.toml or uv.toml:
# [tool.uv]
# python-downloads = "manual"
```

---

## 5. Project Initialization and Layouts

`uv` supports two primary project types: **Applications** and **Libraries**.

### Initializing a Project

**Application (Default):**
For scripts, web servers, CLIs. Not intended for distribution on PyPI.
```bash
uv init my-app
cd my-app
```
Creates:
```
my-app/
├── .python-version
├── README.md
├── main.py
└── pyproject.toml
```
The `pyproject.toml` has no `[build-system]`.

**Library:**
For packages intended for distribution (e.g., PyPI). Uses the `src/` layout.
```bash
uv init --lib my-lib
cd my-lib
```
Creates:
```
my-lib/
├── .python-version
├── README.md
├── pyproject.toml
└── src/
    └── my_lib/
        ├── __init__.py
        └── py.typed
```
The `pyproject.toml` includes a `[build-system]`.

**Packaged Application:**
An application that also needs to be installable (e.g., for `pytest` discovery or to define CLI entry points).
```bash
uv init --package my-cli
```

### Project File Structure (After First Run)
After running `uv sync` or `uv run`, the full structure looks like this:
```
my-project/
├── .venv/                # Managed virtual environment (Git-ignored)
│   └── ...
├── .python-version       # Pinned Python version
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Universal, cross-platform lockfile (Commit this!)
├── README.md
└── src/                  # (For libraries/packaged apps)
    └── my_project/
        └── __init__.py
```

### The `pyproject.toml` File
This is the central configuration file for your project.

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "My awesome project"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.28",
    "pandas",
]

# Optional dependencies (extras), e.g., `pip install my-project[plot]`
[project.optional-dependencies]
plot = ["matplotlib>=3.6"]

# CLI entry points
[project.scripts]
my-cli = "my_project:main"

# Build system (for libraries)
[build-system]
requires = ["uv_build>=0.9.30,<0.10.0"]
build-backend = "uv_build"

# --- uv-specific configuration ---

# Development dependencies
[dependency-groups]
dev = ["pytest>=8.0", "ruff"]
docs = ["sphinx"]

# Dependency sources (Git, local paths, specific indexes)
[tool.uv.sources]
my-helper = { path = "./packages/helper", editable = true }

# uv settings
[tool.uv]
# default-groups = ["dev"]
```

### The `uv.lock` File
*   **Universal:** Contains resolutions for all platforms (Windows, macOS, Linux).
*   **Human-Readable:** It's TOML, but **should not be edited manually**.
*   **Version Control:** **Always commit `uv.lock`** to ensure reproducible builds for all collaborators.

---

## 6. Dependency Management

### Adding Dependencies

```bash
# Add a package from PyPI
uv add requests

# Add with a version constraint
uv add "pandas>=2.0,<3"

# Add a development dependency (e.g., pytest)
uv add --dev pytest

# Add to a custom group (e.g., linting tools)
uv add --group lint ruff mypy

# Add an optional dependency (an "extra")
uv add --optional plot matplotlib

# Add from a Git repository
uv add git+https://github.com/encode/httpx

# Add from a Git repository with a specific tag/branch/commit
uv add git+https://github.com/encode/httpx --tag 0.27.0
uv add git+https://github.com/encode/httpx --branch main
uv add git+https://github.com/encode/httpx --rev a1b2c3d

# Add a local package (editable)
uv add --editable ./packages/my-local-lib

# Add from an alternative index (e.g., PyTorch CPU)
uv add torch --index pytorch=https://download.pytorch.org/whl/cpu

# Add platform-specific dependency
uv add "pywin32; sys_platform == 'win32'"
```

### Removing Dependencies
```bash
uv remove requests
uv remove --dev pytest
uv remove --group lint ruff
```

### Updating Dependencies
The `uv lock` command is used to update the lockfile.

```bash
# Update ALL packages to the latest versions allowed by constraints
uv lock --upgrade

# Update a SINGLE package
uv lock --upgrade-package requests

# Update a package to a specific version
uv lock --upgrade-package requests==2.31.0
```
Note: Running `uv add "requests>=2.32"` will also update the constraint in `pyproject.toml` and re-lock.

### Syncing the Environment
`uv sync` installs the dependencies from `uv.lock` into the `.venv`.

```bash
# Standard sync (removes extraneous packages)
uv sync

# Sync including a specific group
uv sync --group docs

# Sync including all extras
uv sync --all-extras

# Sync excluding dev dependencies (for production)
uv sync --no-dev

# Sync only dev dependencies (no project itself)
uv sync --only-dev
```

### Importing from `requirements.txt`
```bash
uv add -r requirements.txt
uv add --dev -r requirements-dev.txt
```

### Exporting the Lockfile
```bash
# Export to requirements.txt format
uv export --format requirements.txt > requirements.txt

# Export to PEP 751 pylock.toml
uv export --format pylock.toml
```

---

## 7. Running Code

The `uv run` command executes a command within the project's virtual environment. **It automatically syncs the environment first.**

```bash
# Run a Python script
uv run python main.py

# Run a module
uv run python -m pytest

# Run a command provided by a dependency
uv run flask run --port 5000

# Run an installed entry point
uv run my-cli --help
```

### Flags for `uv run`
```bash
# Run without syncing (faster, for when you know env is up-to-date)
uv run --no-sync python main.py

# Run with a temporary, ephemeral dependency
uv run --with httpx python -c "import httpx; print(httpx.get('https://example.com'))"

# Run with a specific Python version (downloads if needed)
uv run --python 3.10 python --version

# Fail if lockfile is out of date (useful for CI)
uv run --locked pytest
```

### Running Standalone Scripts with Inline Metadata
For single-file scripts, you can embed dependencies directly in the file.

**`my_script.py`:**
```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "rich",
# ]
# ///

import requests
from rich import print

print(requests.get("https://api.github.com").json())
```

**Run it:**
```bash
uv run my_script.py
```
`uv` will create an isolated, temporary environment with `requests` and `rich` installed.

**Add dependencies to a script:**
```bash
uv add --script my_script.py pandas
```

---

## 8. Building the Project

`uv build` creates distributable artifacts: a **source distribution** (`.tar.gz`) and a **wheel** (`.whl`).

```bash
# Build the project in the current directory
uv build
```
Output is placed in the `dist/` directory:
```
dist/
├── my_project-0.1.0-py3-none-any.whl
└── my_project-0.1.0.tar.gz
```

### Build Flags
```bash
# Build only the source distribution
uv build --sdist

# Build only the wheel
uv build --wheel

# Build without using tool.uv.sources (simulates a clean build)
uv build --no-sources
```

### Build Systems
The build process is controlled by the `[build-system]` table in `pyproject.toml`. uv's native backend (`uv_build`) is fast and works for pure Python projects.

```toml
# uv's native backend (pure Python only)
[build-system]
requires = ["uv_build>=0.9.30,<0.10.0"]
build-backend = "uv_build"

# Alternative: Hatchling
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Alternative: Setuptools
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

For **extension modules** (Rust, C++), use `maturin` or `scikit-build-core`:
```bash
uv init --build-backend maturin my-rust-project
```

---

## 9. Packaging and Publishing

### Updating Version Before Publishing
```bash
# Set an exact version
uv version 1.0.0

# Bump version semantically
uv version --bump patch  # 0.1.0 -> 0.1.1
uv version --bump minor  # 0.1.1 -> 0.2.0
uv version --bump major  # 0.2.0 -> 1.0.0

# Preview a change without applying it
uv version 2.0.0 --dry-run
```

### Publishing to PyPI
```bash
# Publish using a PyPI API token
uv publish --token <YOUR_PYPI_TOKEN>

# Or using environment variables
export UV_PUBLISH_TOKEN=<YOUR_PYPI_TOKEN>
uv publish
```

### Publishing to a Private Index
Configure the index in `pyproject.toml`:
```toml
[[tool.uv.index]]
name = "private"
url = "https://my-private-index.com/simple/"
publish-url = "https://my-private-index.com/upload/"
```
Then publish:
```bash
uv publish --index private
```

### Preventing Accidental Uploads
Add a classifier to `pyproject.toml` to make PyPI reject the upload:
```toml
[project]
classifiers = ["Private :: Do Not Upload"]
```

---

## 10. Workspaces (Monorepos)

Workspaces allow managing multiple related packages in a single repository, sharing a single `uv.lock` file.

### Structure
```
my-monorepo/
├── pyproject.toml          # Workspace root
├── uv.lock                 # Shared lockfile
├── packages/
│   ├── core/
│   │   ├── pyproject.toml  # Workspace member
│   │   └── src/core/...
│   └── cli/
│       ├── pyproject.toml  # Workspace member
│       └── src/cli/...
```

### Root `pyproject.toml`
```toml
[project]
name = "my-monorepo"
version = "0.1.0"
dependencies = ["core", "cli"] # Depend on members

[tool.uv.sources]
core = { workspace = true }
cli = { workspace = true }

[tool.uv.workspace]
members = ["packages/*"]
```

### Member `pyproject.toml` (e.g., `packages/cli/pyproject.toml`)
```toml
[project]
name = "cli"
version = "0.1.0"
dependencies = ["core", "click"]

[tool.uv.sources]
core = { workspace = true } # Depend on sibling
```

### Running Commands in a Workspace
```bash
# Run in the root project (default)
uv run pytest

# Run in a specific member
uv run --package cli my-cli-command

# Sync a specific member's dependencies
uv sync --package core
```

---

## 11. The Pip Compatibility Interface

For users migrating from `pip` or needing manual control, `uv` provides drop-in replacements.

```bash
# Create a virtual environment
uv venv
uv venv --python 3.11 .venv-py311

# Activate manually (standard venv activation)
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install packages
uv pip install requests flask

# Install from requirements.txt
uv pip install -r requirements.txt

# Freeze installed packages
uv pip freeze > requirements.txt

# Compile/lock requirements (like pip-tools)
uv pip compile requirements.in -o requirements.txt

# Sync environment to match a requirements file exactly
uv pip sync requirements.txt

# Show dependency tree
uv pip tree
```

---

## 12. Tools (CLI Utilities)

`uvx` is used to run command-line tools without installing them into your project.

### Running Tools Temporarily
```bash
# Run ruff linter
uvx ruff check .

# Run black formatter
uvx black .

# Run a specific version
uvx ruff@0.4.0 check .
```

### Installing Tools Globally
```bash
# Install a tool to be available system-wide
uv tool install ruff

# Now 'ruff' can be called directly
ruff --version

# List installed tools
uv tool list

# Upgrade a tool
uv tool upgrade ruff

# Uninstall
uv tool uninstall ruff
```

---

## 13. Comparison with Maven (For Java Developers)

If you are coming from Java and Maven, here's how `uv` concepts map to familiar ones.

| Maven Concept        | Maven Command                      | uv Equivalent           | uv Command              |
| -------------------- | ---------------------------------- | ----------------------- | ----------------------- |
| Project descriptor   | `pom.xml`                          | `pyproject.toml`        | -                       |
| Dependency lock      | -                                  | `uv.lock`               | -                       |
| Local repo / cache   | `~/.m2/repository`                 | `~/.cache/uv`           | -                       |
| Create project       | `mvn archetype:generate`           | `uv init`               | `uv init my-project`    |
| Add dependency       | Edit `pom.xml`                     | `uv add`                | `uv add requests`       |
| Remove dependency    | Edit `pom.xml`                     | `uv remove`             | `uv remove requests`    |
| Resolve dependencies | `mvn dependency:resolve`           | `uv lock`               | `uv lock`               |
| Install dependencies | `mvn install` (for deps)           | `uv sync`               | `uv sync`               |
| Update dependencies  | `mvn versions:use-latest-versions` | `uv lock --upgrade`     | `uv lock --upgrade`     |
| Clean                | `mvn clean`                        | (Manual)                | `rm -rf .venv dist`     |
| Compile              | `mvn compile`                      | (Not needed for Python) | -                       |
| Run tests            | `mvn test`                         | `uv run pytest`         | `uv run pytest`         |
| Package              | `mvn package`                      | `uv build`              | `uv build`              |
| Deploy / Publish     | `mvn deploy`                       | `uv publish`            | `uv publish`            |
| Run application      | `mvn exec:java`                    | `uv run`                | `uv run python main.py` |
| Dependency tree      | `mvn dependency:tree`              | `uv tree`               | `uv tree`               |
| Multi-module project | Parent `pom.xml` with modules      | Workspaces              | `tool.uv.workspace`     |

### Example Workflow Comparison

**Maven (Java):**
```bash
mvn archetype:generate -DgroupId=com.example -DartifactId=my-app
cd my-app
# Edit pom.xml to add dependencies
mvn clean install
mvn exec:java -Dexec.mainClass="com.example.App"
mvn package
mvn deploy
```

**uv (Python):**
```bash
uv init my-app
cd my-app
uv add requests click  # Add dependencies
uv sync                # Install dependencies
uv run python main.py  # Run application
uv build               # Package into wheel/sdist
uv publish             # Deploy to PyPI
```

---

## 14. Cheat Sheet

### Daily Workflow
```bash
# Start a new project
uv init my-project && cd my-project

# Add dependencies
uv add fastapi uvicorn
uv add --dev pytest httpx

# Run your app (auto-syncs environment)
uv run uvicorn main:app --reload

# Run tests
uv run pytest

# Update a dependency
uv lock --upgrade-package fastapi
```

### CI/CD Pipeline
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync with strict lockfile checking
uv sync --locked --no-dev

# Run tests
uv run --no-sync pytest

# Build
uv build --no-sources

# Publish (with token from CI secrets)
uv publish --token $PYPI_TOKEN
```

### Quick Reference Table
| I want to...               | Command                              |
| -------------------------- | ------------------------------------ |
| Create a new project       | `uv init my-project`                 |
| Add a library              | `uv add requests`                    |
| Add a dev tool             | `uv add --dev pytest`                |
| Remove a library           | `uv remove requests`                 |
| Update all libraries       | `uv lock --upgrade`                  |
| Update one library         | `uv lock --upgrade-package requests` |
| Install Python 3.12        | `uv python install 3.12`             |
| Run my code                | `uv run python main.py`              |
| Run tests                  | `uv run pytest`                      |
| Use a one-off tool         | `uvx ruff check .`                   |
| Build for distribution     | `uv build`                           |
| Publish to PyPI            | `uv publish`                         |
| See dependency tree        | `uv tree`                            |
| Export to requirements.txt | `uv export -f requirements.txt`      |