#!/usr/bin/env python3
"""
Ultra Memory-Optimized ADAPT-VQE for LiH
This version uses extremely conservative parameters to prevent memory issues.
"""

import os
import gc
import sys

# Set aggressive memory optimization environment variables
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['MKL_MAIN_FREE'] = '1'

# Force garbage collection before starting
gc.collect()

def main():
    print("Starting Ultra Memory-Optimized ADAPT-VQE for LiH")
    print("Using extremely conservative parameters to prevent memory issues")

    try:
        # Import modules one by one to monitor memory usage
        print("Importing modules...")

        from src.pools import QE, SD  # Use QE instead of SD (smaller pool)
        print("✓ QE pool imported")

        from src.molecules import create_lih
        print("✓ Molecule creation imported")

        from algorithms.adapt_vqe import AdaptVQE
        print("✓ AdaptVQE imported")

        # Force garbage collection after imports
        gc.collect()

        # Create molecule with smaller basis set if possible
        print("Creating LiH molecule...")
        r = 1.5
        molecule = create_lih(r)
        print(f"✓ Molecule created: {molecule.description}")

        # Use QE pool instead of SD (much smaller)
        print("Creating QE operator pool...")
        pool = SD(molecule)
        print(f"✓ Pool created: size = {pool.size}")

        # Ultra-conservative parameters
        print("Creating AdaptVQE with ultra-conservative parameters...")
        adapt_vqe = AdaptVQE(
            pool=pool,
            molecule=molecule,
            max_adapt_iter=10,      # Very small
            max_opt_iter=10,        # Very small
            grad_threshold=1e-3,    # Relaxed threshold
            vrb=True,
            optimizer_method='l-bfgs-b',
            shots_assignment='uniform',
            k=20,                   # Very small
            shots_budget=128,       # Very small
            N_experiments=1         # Single experiment
        )
        print("✓ AdaptVQE instance created")

        # Force garbage collection before running
        gc.collect()

        print("Starting ADAPT-VQE run...")
        print("="*50)

        # Run the algorithm
        result = adapt_vqe.run()

        print("="*50)
        print("✓ ADAPT-VQE completed successfully!")

        # Print results
        if hasattr(result, 'energy'):
            print(f"Final energy: {result.energy}")
        if hasattr(result, 'iterations'):
            print(f"Total iterations: {result.iterations}")

    except MemoryError as e:
        print(f"❌ Memory error: {e}")
        print("Try even more conservative parameters or use H2 instead of LiH")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True

if __name__ == '__main__':
    success = main()
    if success:
        print("\n🎉 Successfully completed ultra memory-optimized ADAPT-VQE!")
    else:
        print("\n💥 Failed to complete ADAPT-VQE")
        print("Consider:")
        print("  - Using H2 instead of LiH")
        print("  - Reducing parameters further")
        print("  - Running on a machine with more RAM")
