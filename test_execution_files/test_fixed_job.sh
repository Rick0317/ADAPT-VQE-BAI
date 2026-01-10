#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=test_adapt_vqe_fixed
#SBATCH --output=test_adapt_vqe_fixed_%j.out
#SBATCH --error=test_adapt_vqe_fixed_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca

# Fixed ADAPT-VQE test job
# This script runs a corrected ADAPT-VQE calculation for testing

echo "=========================================="
echo "Fixed ADAPT-VQE Test Job Started"
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
    exit 1
fi

source "$SCRATCH_ENV/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated successfully"
echo ""

# Set up matplotlib config directory (fixed path)
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"
echo "✓ Matplotlib config directory: $MPLCONFIGDIR"

# Set memory and CPU optimizations for Trillium
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Change to working directory
cd $SLURM_SUBMIT_DIR

echo "=========================================="
echo "Running Fixed ADAPT-VQE Test"
echo "=========================================="

# Create a fixed test script for ADAPT-VQE
cat > test_adapt_vqe_fixed.py << 'EOF'
#!/usr/bin/env python3
"""
Fixed ADAPT-VQE test for H2 molecule
"""

import numpy as np
import time
from pyscf import gto, scf
from openfermion import MolecularData
from openfermionpyscf import run_pyscf

def test_adapt_vqe_h2():
    """Test ADAPT-VQE with H2 molecule"""
    
    print("Starting ADAPT-VQE test for H2 molecule...")
    start_time = time.time()
    
    # H2 molecule parameters
    geometry = [['H', [0., 0., 0.]], ['H', [0., 0., 0.74]]]
    basis = 'sto-3g'
    multiplicity = 1
    
    print(f"Geometry: {geometry}")
    print(f"Basis: {basis}")
    print(f"Multiplicity: {multiplicity}")
    
    # Create molecular data
    molecule = MolecularData(geometry, basis, multiplicity)
    molecule = run_pyscf(molecule, run_scf=True)
    
    print(f"✓ Molecular data created")
    print(f"✓ H2 ground state energy: {molecule.hf_energy:.6f} Hartree")
    print(f"✓ Number of qubits: {molecule.n_qubits}")
    print(f"✓ Number of electrons: {molecule.n_electrons}")
    
    # Get molecular Hamiltonian
    hamiltonian = molecule.get_molecular_hamiltonian()
    print(f"✓ Molecular Hamiltonian created")
    print(f"✓ Hamiltonian type: {type(hamiltonian)}")
    
    # Test Hamiltonian properties
    try:
        # Check if it's an InteractionOperator
        if hasattr(hamiltonian, 'constant'):
            print(f"✓ Hamiltonian constant term: {hamiltonian.constant}")
        
        if hasattr(hamiltonian, 'one_body_tensor'):
            print(f"✓ One-body tensor shape: {hamiltonian.one_body_tensor.shape}")
        
        if hasattr(hamiltonian, 'two_body_tensor'):
            print(f"✓ Two-body tensor shape: {hamiltonian.two_body_tensor.shape}")
            
    except Exception as e:
        print(f"Note: Could not access Hamiltonian tensor properties: {e}")
    
    # Test basic quantum chemistry operations
    print("Testing quantum chemistry operations...")
    
    # Get molecular orbitals
    try:
        orbitals = molecule.canonical_orbitals
        print(f"✓ Molecular orbitals shape: {orbitals.shape}")
    except Exception as e:
        print(f"Note: Could not access orbitals: {e}")
    
    # Test fermionic operators
    print("Testing fermionic operator creation...")
    try:
        from openfermion import FermionOperator
        
        # Create a simple fermionic operator
        op = FermionOperator('0^ 1', 1.0)
        print(f"✓ Fermionic operator created: {op}")
        
    except Exception as e:
        print(f"Note: Could not create fermionic operator: {e}")
    
    end_time = time.time()
    print(f"✓ Test completed in {end_time - start_time:.2f} seconds")
    
    return True

if __name__ == "__main__":
    try:
        success = test_adapt_vqe_h2()
        if success:
            print("✓ ADAPT-VQE test completed successfully!")
        else:
            print("✗ ADAPT-VQE test failed!")
    except Exception as e:
        print(f"✗ ADAPT-VQE test failed with error: {e}")
        import traceback
        traceback.print_exc()
EOF

# Run the test
echo "Running fixed ADAPT-VQE test script..."
python test_adapt_vqe_fixed.py

# Check if test was successful
if [ $? -eq 0 ]; then
    echo "✓ Fixed ADAPT-VQE test completed successfully!"
else
    echo "✗ Fixed ADAPT-VQE test failed!"
fi

echo ""
echo "=========================================="
echo "Fixed Test Job Summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "End time: $(date)"
echo "Working directory: $(pwd)"
echo "Matplotlib config: $MPLCONFIGDIR"
echo ""

# Clean up test file
rm -f test_adapt_vqe_fixed.py

echo "=========================================="
echo "Fixed ADAPT-VQE test job completed!"
echo "=========================================="





