# Using uv with ADAPT-VQE-BAI Project

This project now uses `uv` for dependency management. Here's how to use it:

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run Python scripts:**
   ```bash
   Suv run python your_script.py
   ```

3. **Run tests:**
   ```bash
   uv run pytest
   ```

## Common Commands

### Dependency Management
- `uv sync` - Install all dependencies from pyproject.toml
- `uv add package-name` - Add a new dependency
- `uv add --dev package-name` - Add a development dependency
- `uv remove package-name` - Remove a dependency
- `uv lock` - Update the lock file

### Running Code
- `uv run python script.py` - Run a Python script in the virtual environment
- `uv run pytest` - Run tests
- `uv run black .` - Format code with black
- `uv run flake8 .` - Lint code with flake8

### Environment Management
- `uv venv` - Create a new virtual environment
- `uv shell` - Activate the virtual environment in your shell

## Project Structure

The project dependencies are now managed in `pyproject.toml` instead of multiple `requirements*.txt` files. The consolidated dependencies include:

- **Core quantum chemistry:** openfermion, openfermionpyscf, pyscf
- **Quantum computing:** qiskit, qiskit-aer
- **Scientific computing:** numpy, scipy
- **Data analysis:** matplotlib, pandas
- **System monitoring:** psutil
- **Development tools:** pytest, black, flake8

## Benefits of Using uv

1. **Faster installs** - uv is significantly faster than pip
2. **Better dependency resolution** - More reliable than pip's resolver
3. **Reproducible builds** - Lock file ensures consistent environments
4. **Modern Python packaging** - Uses pyproject.toml standard
5. **Integrated virtual environment management** - No need for separate venv tools

## Migration from requirements.txt

The old requirements files are still present but are no longer used:
- `requirements_h5_hamiltonian.txt`
- `tests/requirements.txt`
- `shot-efficient-adapt-vqe/requirements_memory_monitoring.txt`

All dependencies have been consolidated into `pyproject.toml`.



