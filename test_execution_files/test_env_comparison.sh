#!/bin/bash
#SBATCH --account=def-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --job-name=test_env_comparison
#SBATCH --output=test_env_comparison_%j.out
#SBATCH --error=test_env_comparison_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca

# Environment comparison script

echo "=========================================="
echo "Environment Comparison Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Check if we're on a compute node
echo "Node information:"
hostname
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

echo "✓ Virtual environment activated"
echo ""

# Detailed Python environment analysis
echo "Detailed Python environment analysis:"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check Python path in detail
python -c "
import sys
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Python path:')
for i, path in enumerate(sys.path):
    print(f'  {i}: {path}')
print('')

# Check if virtual environment is properly activated
import os
venv_path = os.environ.get('VIRTUAL_ENV', 'Not set')
print(f'VIRTUAL_ENV environment variable: {venv_path}')
print(f'sys.prefix: {sys.prefix}')
print(f'sys.base_prefix: {sys.base_prefix}')
print('')

# Check if we're in a virtual environment
in_venv = sys.prefix != sys.base_prefix
print(f'In virtual environment: {in_venv}')
print('')

# List all installed packages
print('Installed packages (first 20):')
import pkg_resources
installed_packages = [d.project_name for d in pkg_resources.working_set]
for i, pkg in enumerate(sorted(installed_packages)[:20]):
    print(f'  {pkg}')
if len(installed_packages) > 20:
    print(f'  ... and {len(installed_packages) - 20} more')
print('')

# Check for qiskit specifically
qiskit_packages = [p for p in installed_packages if 'qiskit' in p.lower()]
print(f'Qiskit-related packages: {qiskit_packages}')
print('')

# Try to find qiskit in the file system
import os
venv_site_packages = f'{sys.prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
print(f'Virtual env site-packages: {venv_site_packages}')
if os.path.exists(venv_site_packages):
    qiskit_dirs = [d for d in os.listdir(venv_site_packages) if 'qiskit' in d.lower()]
    print(f'Qiskit directories in site-packages: {qiskit_dirs}')
else:
    print('Site-packages directory does not exist!')
print('')
"

# Test qiskit import with detailed error
echo "Testing qiskit import with detailed error:"
python -c "
try:
    import qiskit
    print(f'✓ Qiskit imported successfully')
    print(f'✓ Qiskit version: {qiskit.__version__}')
    print(f'✓ Qiskit location: {qiskit.__file__}')
except ImportError as e:
    print(f'✗ Qiskit import failed: {e}')
    print('Import error details:')
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f'✗ Unexpected error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=========================================="
echo "Environment comparison completed"
echo "=========================================="



