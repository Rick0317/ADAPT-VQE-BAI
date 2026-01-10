#!/bin/bash

# Test script to verify uv environment setup
# This simulates what the SLURM job will do

echo "Testing uv environment setup..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "uv version:"
uv --version

echo "Setting up environment..."
uv sync

echo "Testing Python environment..."
uv run python --version

echo "Testing imports..."
uv run python -c "
import numpy as np
import scipy
import qiskit
import openfermion
import matplotlib
import pandas
import psutil
print('✓ All core dependencies imported successfully!')
print(f'NumPy: {np.__version__}')
print(f'Qiskit: {qiskit.__version__}')
print(f'SciPy: {scipy.__version__}')
"

echo "✓ Environment test completed successfully!"



