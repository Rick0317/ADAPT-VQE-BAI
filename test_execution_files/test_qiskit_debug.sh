#!/bin/bash
#SBATCH --account=def-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --job-name=test_qiskit_debug
#SBATCH --output=test_qiskit_debug_%j.out
#SBATCH --error=test_qiskit_debug_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca

# Debug script for qiskit import issues

echo "=========================================="
echo "Qiskit Debug Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Check environment variables
echo "Environment variables:"
echo "SCRATCH: $SCRATCH"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "PATH: $PATH"
echo ""

# Activate virtual environment
SCRATCH_ENV="$SCRATCH/adapt_vqe_env"
echo "Virtual environment path: $SCRATCH_ENV"

if [ ! -d "$SCRATCH_ENV" ]; then
    echo "Error: Virtual environment not found at $SCRATCH_ENV"
    exit 1
fi

echo "Activating virtual environment..."
source "$SCRATCH_ENV/bin/activate"

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Debug Python environment
echo "Python environment debug:"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo ""

# Check if qiskit is installed
echo "Checking qiskit installation:"
pip list | grep -i qiskit
echo ""

# Test qiskit import step by step
echo "Testing qiskit import:"
python -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python path: {sys.path}')
print(f'Virtual environment: {sys.prefix}')
print('')

try:
    import qiskit
    print(f'✓ Qiskit imported successfully')
    print(f'✓ Qiskit version: {qiskit.__version__}')
except ImportError as e:
    print(f'✗ Qiskit import failed: {e}')
    print('Available packages:')
    import pkg_resources
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    qiskit_packages = [p for p in installed_packages if 'qiskit' in p.lower()]
    print(f'Qiskit-related packages: {qiskit_packages}')
except Exception as e:
    print(f'✗ Unexpected error: {e}')
"

echo ""
echo "Testing qiskit_aer import:"
python -c "
try:
    import qiskit_aer
    print(f'✓ Qiskit Aer imported successfully')
    print(f'✓ Qiskit Aer version: {qiskit_aer.__version__}')
except ImportError as e:
    print(f'✗ Qiskit Aer import failed: {e}')
except Exception as e:
    print(f'✗ Unexpected error: {e}')
"

echo ""
echo "=========================================="
echo "Debug test completed"
echo "=========================================="



