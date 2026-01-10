# Deploying ADAPT-VQE-BAI with uv on Remote Cluster

This guide explains how to deploy and run the ADAPT-VQE-BAI project using `uv` on your remote cluster.

## Prerequisites

1. **Python module available** on the cluster
2. **Internet access** for downloading `uv` and dependencies
3. **Sufficient disk space** for the virtual environment

## Deployment Steps

### 1. Upload Project Files

Upload these files to your cluster:
- `pyproject.toml` (dependency configuration)
- `uv.lock` (locked dependencies)
- All Python source files (`adaptvqe/`, `entities/`, `operator_pools/`, etc.)
- `submit_all_commands_1.sh` (updated SLURM script)
- `notes.txt` (job parameters)

### 2. First-Time Setup on Cluster

Run this once to set up `uv` and install dependencies:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
uv --version

# Install project dependencies
uv sync
```

### 3. Running Jobs

The updated `submit_all_commands_1.sh` script will:

1. **Auto-install uv** if not present
2. **Set up the environment** with `uv sync`
3. **Run Python scripts** using `uv run python script.py`

Submit jobs as usual:
```bash
sbatch submit_all_commands_1.sh
```

## Key Changes Made

### Environment Setup
- **Removed:** `module load conda3` and `source ~/adapt-vqe-env/bin/activate`
- **Added:** Automatic `uv` installation and `uv sync`

### Python Execution
- **Changed:** `python script.py` → `uv run python script.py`
- **Added:** `uv` version checking in environment tests

### Benefits
- **Faster dependency resolution** and installation
- **Reproducible environments** via lock file
- **No conda environment management** needed
- **Automatic dependency handling** per job

## Troubleshooting

### If uv installation fails:
```bash
# Manual installation
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### If dependencies fail to install:
```bash
# Check Python version compatibility
uv run python --version

# Force reinstall
rm -rf .venv
uv sync
```

### If jobs fail with import errors:
```bash
# Test environment locally
uv run python -c "import numpy, scipy, qiskit, openfermion; print('All imports successful')"
```

## Performance Considerations

- **First job** may take longer due to `uv` installation
- **Subsequent jobs** will be faster due to cached dependencies
- **Virtual environment** is created in `.venv/` directory
- **Lock file** ensures consistent dependency versions across all jobs

## Monitoring

Check job status and logs:
```bash
# Check job queue
squeue -u $USER

# Monitor specific molecule jobs
squeue -u $USER | grep h4
squeue -u $USER | grep lih

# Check job logs
tail -f job_output_file.out
```

The updated script maintains all original functionality while using `uv` for modern, fast dependency management.













