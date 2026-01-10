#!/bin/bash
#SBATCH --account=def-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=test_venv_fix
#SBATCH --output=test_venv_fix_%j.out
#SBATCH --error=test_venv_fix_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca

# Test script to verify virtual environment fix

echo "=========================================="
echo "Virtual Environment Fix Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate existing virtual environment
echo "Environment variables debug:"
echo "SCRATCH: $SCRATCH"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "USER: $USER"
echo ""

SCRATCH_ENV="/scratch/rick0317/adapt_vqe_env"
echo "Activating existing virtual environment: $SCRATCH_ENV"

if [ ! -d "$SCRATCH_ENV" ]; then
    echo "Error: Virtual environment not found at $SCRATCH_ENV"
    echo "Please create the environment first with:"
    echo "  python -m venv $SCRATCH_ENV"
    echo "  source $SCRATCH_ENV/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

source "$SCRATCH_ENV/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated successfully"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Test qiskit import
echo "Testing qiskit import..."
python -c "import qiskit; print('✓ Qiskit version:', qiskit.__version__)"

echo ""
echo "=========================================="
echo "Virtual environment fix test completed!"
echo "=========================================="

