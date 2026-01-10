#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=test_adapt_vqe_env
#SBATCH --output=test_adapt_vqe_env_%j.out
#SBATCH --error=test_adapt_vqe_env_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca

# Test script for ADAPT-VQE environment setup on Trillium
# This script tests the environment and runs a simple ADAPT-VQE calculation
# Assumes virtual environment 'adapt_vqe_env' already exists

echo "=========================================="
echo "ADAPT-VQE Environment Test Job Started"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate existing virtual environment
SCRATCH_ENV="$SCRATCH/adapt_vqe_env"
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
echo ""

# Set up matplotlib config directory
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
mkdir -p "$MPLCONFIGDIR"

# Set memory and CPU optimizations for Trillium
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "=========================================="
echo "Testing Environment and Libraries"
echo "=========================================="

# Test Python environment
echo "Testing Python environment..."
which python
python --version
echo ""

# Test core scientific libraries
echo "Testing core scientific libraries..."
python -c "
import numpy as np
import scipy
import matplotlib
import pandas as pd
import psutil
print('✓ NumPy version:', np.__version__)
print('✓ SciPy version:', scipy.__version__)
print('✓ Matplotlib version:', matplotlib.__version__)
print('✓ Pandas version:', pd.__version__)
print('✓ Psutil version:', psutil.__version__)
print('✓ All core libraries imported successfully')
"
echo ""

# Test quantum chemistry libraries
echo "Testing quantum chemistry libraries..."
python -c "
try:
    import openfermion
    print('✓ OpenFermion version:', openfermion.__version__)
except ImportError as e:
    print('✗ OpenFermion import failed:', e)

try:
    import openfermionpyscf
    print('✓ OpenFermionPySCF imported successfully')
except ImportError as e:
    print('✗ OpenFermionPySCF import failed:', e)

try:
    import pyscf
    print('✓ PySCF version:', pyscf.__version__)
except ImportError as e:
    print('✗ PySCF import failed:', e)
"
echo ""

# Test basic quantum chemistry functionality
echo "Testing basic quantum chemistry functionality..."
python -c "
try:
    import numpy as np
    from pyscf import gto, scf
    
    # Create a simple H2 molecule
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    mf = scf.RHF(mol)
    mf.kernel()
    
    print('✓ PySCF H2 calculation completed successfully')
    print('✓ Ground state energy:', mf.e_tot, 'Hartree')
    
except Exception as e:
    print('✗ Quantum chemistry test failed:', e)
"
echo ""

# Test ADAPT-VQE basic functionality
echo "Testing ADAPT-VQE basic functionality..."
python -c "
try:
    import numpy as np
    from pyscf import gto, scf
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    
    # Create H2 molecule for ADAPT-VQE test
    geometry = [['H', [0., 0., 0.]], ['H', [0., 0., 0.74]]]
    basis = 'sto-3g'
    multiplicity = 1
    
    # Create molecular data
    molecule = MolecularData(geometry, basis, multiplicity)
    molecule = run_pyscf(molecule, run_scf=True)
    
    print('✓ Molecular data creation successful')
    print('✓ H2 ground state energy:', molecule.hf_energy, 'Hartree')
    print('✓ Number of qubits:', molecule.n_qubits)
    print('✓ Number of electrons:', molecule.n_electrons)
    
except Exception as e:
    print('✗ ADAPT-VQE basic test failed:', e)
"
echo ""

# Test memory and CPU usage
echo "Testing system resources..."
python -c "
import psutil
import os

print('✓ Available memory:', psutil.virtual_memory().available / (1024**3), 'GB')
print('✓ CPU count:', psutil.cpu_count())
print('✓ Current process memory:', psutil.Process().memory_info().rss / (1024**2), 'MB')
print('✓ OMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS', 'Not set'))
"
echo ""

# Print final environment summary
echo "=========================================="
echo "Environment Test Summary"
echo "=========================================="
echo "Virtual environment: $SCRATCH_ENV"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "End time: $(date)"
echo ""

# List installed packages
echo "Installed packages:"
pip list | grep -E "(numpy|scipy|matplotlib|pandas|psutil|openfermion|pyscf)"
echo ""

echo "=========================================="
echo "Environment test completed successfully!"
echo "Ready for ADAPT-VQE calculations!"
echo "=========================================="