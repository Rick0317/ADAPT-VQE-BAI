#!/bin/bash
#SBATCH --account=def-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=test_package_installation
#SBATCH --output=test_package_installation_%j.out
#SBATCH --error=test_package_installation_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca

# Load required modules for Trillium
module load StdEnv/2023 gcc python/3.11 symengine/0.11.2

# Comprehensive package installation test job
# This script tests installation of all packages used in the ADAPT-VQE project

echo "=========================================="
echo "Comprehensive Package Installation Test"
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
    echo "Creating new virtual environment..."
    python -m venv "$SCRATCH_ENV"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment with explicit sourcing
echo "Activating virtual environment..."
source "$SCRATCH_ENV/bin/activate"

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

# Verify activation
echo "✓ Virtual environment activated successfully"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Test qiskit import directly
echo "Testing qiskit import directly..."
python -c "import qiskit; print('✓ Qiskit version:', qiskit.__version__)" 2>&1 || echo "✗ Qiskit import failed"

# If qiskit is not installed, try to install it
if ! python -c "import qiskit" 2>/dev/null; then
    echo "Qiskit not found. Attempting to install..."
    pip install qiskit qiskit-aer
    echo "Testing qiskit import after installation..."
    python -c "import qiskit; print('✓ Qiskit version:', qiskit.__version__)" 2>&1 || echo "✗ Qiskit installation failed"
fi

echo ""

# Set up matplotlib config directory
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"
echo "✓ Matplotlib config directory: $MPLCONFIGDIR"

# Set memory and CPU optimizations for Trillium
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Change to working directory
cd $SLURM_SUBMIT_DIR

echo "=========================================="
echo "Testing Package Installation"
echo "=========================================="

# Create comprehensive package test script
cat > test_package_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive package installation test for ADAPT-VQE project
"""

import sys
import time
import traceback

def test_package_import(package_name, import_statement, version_attr=None):
    """Test importing a package and optionally get its version"""
    try:
        exec(import_statement)
        if version_attr:
            version = eval(version_attr)
            print(f"✓ {package_name}: {version}")
        else:
            print(f"✓ {package_name}: imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"✗ {package_name}: Error - {e}")
        return False

def test_package_installation():
    """Test installation of all required packages"""
    
    print("Testing package installation and imports...")
    print("=" * 50)
    
    # Debug information
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    print(f"Virtual environment: {sys.prefix}")
    print("")
    
    # Core scientific packages
    print("\n1. Core Scientific Packages:")
    print("-" * 30)
    
    core_packages = [
        ("NumPy", "import numpy as np", "np.__version__"),
        ("SciPy", "import scipy", "scipy.__version__"),
        ("Matplotlib", "import matplotlib", "matplotlib.__version__"),
        ("Pandas", "import pandas as pd", "pd.__version__"),
        ("Psutil", "import psutil", "psutil.__version__"),
    ]
    
    core_success = 0
    for name, import_stmt, version_attr in core_packages:
        if test_package_import(name, import_stmt, version_attr):
            core_success += 1
    
    print(f"\nCore packages: {core_success}/{len(core_packages)} successful")
    
    # Quantum chemistry packages
    print("\n2. Quantum Chemistry Packages:")
    print("-" * 30)
    
    qc_packages = [
        ("OpenFermion", "import openfermion", "openfermion.__version__"),
        ("OpenFermionPySCF", "import openfermionpyscf", None),
        ("PySCF", "import pyscf", "pyscf.__version__"),
    ]
    
    qc_success = 0
    for name, import_stmt, version_attr in qc_packages:
        if test_package_import(name, import_stmt, version_attr):
            qc_success += 1
    
    print(f"\nQuantum chemistry packages: {qc_success}/{len(qc_packages)} successful")
    
    # Quantum computing packages
    print("\n3. Quantum Computing Packages:")
    print("-" * 30)
    
    qc_computing_packages = [
        ("Qiskit", "import qiskit", "qiskit.__version__"),
        ("Qiskit Aer", "import qiskit_aer", "qiskit_aer.__version__"),
    ]
    
    qc_computing_success = 0
    for name, import_stmt, version_attr in qc_computing_packages:
        if test_package_import(name, import_stmt, version_attr):
            qc_computing_success += 1
    
    print(f"\nQuantum computing packages: {qc_computing_success}/{len(qc_computing_packages)} successful")
    
    # Test basic functionality
    print("\n4. Basic Functionality Tests:")
    print("-" * 30)
    
    # Test PySCF basic calculation
    try:
        from pyscf import gto, scf
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
        mf = scf.RHF(mol)
        mf.kernel()
        print(f"✓ PySCF H2 calculation: {mf.e_tot:.6f} Hartree")
    except Exception as e:
        print(f"✗ PySCF calculation failed: {e}")
    
    # Test OpenFermion molecular data
    try:
        from openfermion import MolecularData
        from openfermionpyscf import run_pyscf
        
        geometry = [['H', [0., 0., 0.]], ['H', [0., 0., 0.74]]]
        molecule = MolecularData(geometry, 'sto-3g', 1)
        molecule = run_pyscf(molecule, run_scf=True)
        print(f"✓ OpenFermion molecular data: {molecule.hf_energy:.6f} Hartree")
        print(f"✓ Number of qubits: {molecule.n_qubits}")
    except Exception as e:
        print(f"✗ OpenFermion molecular data failed: {e}")
    
    # Test Qiskit basic functionality
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        state = Statevector(qc)
        print(f"✓ Qiskit quantum circuit: {len(state)} qubits")
    except Exception as e:
        print(f"✗ Qiskit quantum circuit failed: {e}")
    
    # Test Qiskit Aer simulator
    try:
        from qiskit_aer import AerSimulator
        
        simulator = AerSimulator()
        print(f"✓ Qiskit Aer simulator: {simulator.name}")
    except Exception as e:
        print(f"✗ Qiskit Aer simulator failed: {e}")
    
    # Summary
    total_packages = len(core_packages) + len(qc_packages) + len(qc_computing_packages)
    total_success = core_success + qc_success + qc_computing_success
    
    print("\n" + "=" * 50)
    print("INSTALLATION SUMMARY")
    print("=" * 50)
    print(f"Core packages: {core_success}/{len(core_packages)}")
    print(f"Quantum chemistry: {qc_success}/{len(qc_packages)}")
    print(f"Quantum computing: {qc_computing_success}/{len(qc_computing_packages)}")
    print(f"Total: {total_success}/{total_packages}")
    
    if total_success == total_packages:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        return True
    else:
        print("⚠️  Some packages failed to install")
        return False

if __name__ == "__main__":
    try:
        success = test_package_installation()
        if success:
            print("\n✓ Package installation test completed successfully!")
        else:
            print("\n✗ Package installation test completed with some failures!")
    except Exception as e:
        print(f"\n✗ Package installation test failed with error: {e}")
        traceback.print_exc()
EOF

# Run the comprehensive test
echo "Running comprehensive package installation test..."
python test_package_installation.py

# Check if test was successful
if [ $? -eq 0 ]; then
    echo "✓ Package installation test completed!"
else
    echo "✗ Package installation test failed!"
fi

echo ""
echo "=========================================="
echo "Package Installation Test Summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "End time: $(date)"
echo "Working directory: $(pwd)"
echo "Matplotlib config: $MPLCONFIGDIR"
echo ""

# Clean up test file
rm -f test_package_installation.py

echo "=========================================="
echo "Package installation test completed!"
echo "=========================================="
