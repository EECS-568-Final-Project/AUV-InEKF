# AUV Localization For RoboSub 2025

## Environment Setup (Using `uv`)

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environment management. The environment is defined by:

- `.python-version` – specifies the Python version (used with `pyenv`)
- `pyproject.toml` – declares project dependencies
- `uv.lock` – pins exact package versions for reproducibility

### Prerequisites

Make sure you have the following tools installed:

- [pyenv](https://github.com/pyenv/pyenv) – to manage Python versions
- [uv](https://github.com/astral-sh/uv) – modern Python package manager  
  Install via:

  ```bash
  curl -Ls https://astral.sh/uv/install.sh | sh

## Setup Instructions
1. Install correct Python version (via `pyenv`):
   ```bash
    pyenv install --skip-existing $(cat .python-version)
    pyenv local $(cat .python-version)
   ```

2. Create virutal environment::
   ```bash
    uv venv 
    ```

3. Sync dependencies:
   ```bash
    uv sync
   ```

4. Activate the virtual environment:
   ```bash
    source .venv/bin/activate
   ```

## Run the project:
   ```bash
    python runFilter.py 
   ```