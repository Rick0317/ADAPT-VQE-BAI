#!/usr/bin/env python3
"""
Minimal ADAPT-VQE Test for H2
This version tests if the memory issue is specific to LiH or general.
"""

import os
import gc
import sys
import psutil

def get_memory():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def print_memory(stage):
    """Print current memory usage."""
    print(f"[Memory] {stage}: {get_memory():.1f} MB")

# Set aggressive memory optimization
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['MKL_MAIN_FREE'] = '1'

print("Starting minimal ADAPT-VQE test for H2")
print_memory("Initial")

try:
    # Step 1: Basic imports only
    print("\n1. Testing basic imports...")
    import numpy as np
    print_memory("After NumPy")
    
    import scipy
    print_memory("After SciPy")
    
    # Step 2: OpenFermion imports
    print("\n2. Testing OpenFermion imports...")
    from openfermion import MolecularData
    print_memory("After OpenFermion")
    
    # Step 3: Qiskit imports
    print("\n3. Testing Qiskit imports...")
    from qiskit import QuantumCircuit
    print_memory("After Qiskit")
    
    # Step 4: Local imports
    print("\n4. Testing local imports...")
    from src.molecules import create_h2
    print_memory("After molecule import")
    
    from src.pools import SD  # Can use SD for H2 (smaller system)
    print_memory("After pool import")
    
    # Step 5: Create molecule
    print("\n5. Creating H2 molecule...")
    molecule = create_h2(0.74)  # H2 equilibrium distance
    print(f"✓ Molecule: {molecule.description}")
    print_memory("After molecule creation")
    
    # Step 6: Create pool
    print("\n6. Creating pool...")
    pool = SD(molecule)
    print(f"✓ Pool size: {pool.size}")
    print_memory("After pool creation")
    
    # Step 7: Import AdaptVQE
    print("\n7. Importing AdaptVQE...")
    from algorithms.adapt_vqe import AdaptVQE
    print_memory("After AdaptVQE import")
    
    # Step 8: Create AdaptVQE instance with minimal parameters
    print("\n8. Creating AdaptVQE instance...")
    adapt_vqe = AdaptVQE(
        pool=pool,
        molecule=molecule,
        max_adapt_iter=1,       # Minimal
        max_opt_iter=1,         # Minimal
        grad_threshold=1e-2,    # Relaxed
        vrb=True,
        optimizer_method='l-bfgs-b',
        shots_assignment='uniform',
        k=5,                    # Minimal
        shots_budget=32,        # Minimal
        N_experiments=1
    )
    print_memory("After AdaptVQE creation")
    
    # Step 9: Try to access some attributes (don't run yet)
    print("\n9. Testing AdaptVQE attributes...")
    print(f"  - Pool size: {adapt_vqe.pool.size}")
    print(f"  - Number of qubits: {adapt_vqe.n}")
    print(f"  - Shots budget: {adapt_vqe.shots_budget}")
    print_memory("After attribute access")
    
    print("\n✅ All components loaded successfully!")
    print("Memory usage looks reasonable. You can try running the algorithm.")
    
    # Ask user if they want to proceed
    response = input("\nDo you want to try running the algorithm? (y/n): ")
    if response.lower() == 'y':
        print("\n10. Running minimal ADAPT-VQE for H2...")
        print_memory("Before run")
        
        # Try to run with minimal iterations
        result = adapt_vqe.run()
        
        print_memory("After run")
        print("✅ Algorithm completed!")
        
        if hasattr(result, 'energy'):
            print(f"Final energy: {result.energy}")
    
except MemoryError as e:
    print(f"\n❌ Memory error at step: {e}")
    print("The issue is likely in the AdaptVQE initialization or pool creation.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\nFinal memory usage: {get_memory():.1f} MB") 