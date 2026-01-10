#!/bin/bash
#SBATCH --account=def-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=test_symengine_module
#SBATCH --output=test_symengine_module_%j.out
#SBATCH --error=test_symengine_module_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca

# Test script to verify symengine module loading

echo "=========================================="
echo "SymEngine Module Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load required modules
echo "Loading modules..."
module load StdEnv/2023 gcc python/3.11 symengine/0.11.2

echo "✓ Modules loaded successfully"
echo ""

# Check module status
echo "Module status:"
module list
echo ""

# Activate virtual environment
SCRATCH_ENV="/scratch/rick0317/adapt_vqe_env"
echo "Activating virtual environment: $SCRATCH_ENV"

if [ ! -d "$SCRATCH_ENV" ]; then
    echo "Error: Virtual environment not found at $SCRATCH_ENV"
    exit 1
fi

source "$SCRATCH_ENV/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Test symengine import
echo "Testing symengine import..."
python -c "
try:
    import symengine
    print('✓ SymEngine imported successfully')
    print('✓ SymEngine version:', symengine.__version__)
except ImportError as e:
    print('✗ SymEngine import failed:', e)
except Exception as e:
    print('✗ Unexpected error:', e)
"

echo ""

# Test qiskit import (which depends on symengine)
echo "Testing qiskit import..."
python -c "
try:
    import qiskit
    print('✓ Qiskit imported successfully')
    print('✓ Qiskit version:', qiskit.__version__)
except ImportError as e:
    print('✗ Qiskit import failed:', e)
except Exception as e:
    print('✗ Unexpected error:', e)
"

echo ""
echo "=========================================="
echo "SymEngine module test completed!"
echo "=========================================="

