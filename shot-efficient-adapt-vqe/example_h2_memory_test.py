#!/usr/bin/env python3
"""
Memory Test: ADAPT-VQE for H2
This version tests memory optimization with H2 (much smaller than LiH)
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
    print("Starting Memory Test: ADAPT-VQE for H2")
    print("H2 is much smaller than LiH and should work with default parameters")
    
    try:
        # Import modules one by one to monitor memory usage
        print("Importing modules...")
        
        from src.pools import SD  # Can use SD for H2 (smaller system)
        print("✓ SD pool imported")
        
        from src.molecules import create_h2
        print("✓ H2 molecule creation imported")
        
        from algorithms.adapt_vqe import AdaptVQE
        print("✓ AdaptVQE imported")
        
        # Force garbage collection after imports
        gc.collect()
        
        # Create H2 molecule
        print("Creating H2 molecule...")
        r = 0.74  # H2 equilibrium bond distance
        molecule = create_h2(r)
        print(f"✓ Molecule created: {molecule.description}")
        
        # Create pool
        print("Creating SD operator pool...")
        pool = SD(molecule)
        print(f"✓ Pool created: size = {pool.size}")
        
        # Use moderate parameters for H2
        print("Creating AdaptVQE with moderate parameters...")
        adapt_vqe = AdaptVQE(
            pool=pool,
            molecule=molecule,
            max_adapt_iter=20,      # Moderate for H2
            max_opt_iter=20,        # Moderate for H2
            grad_threshold=1e-4,
            vrb=True,
            optimizer_method='l-bfgs-b',
            shots_assignment='uniform',
            k=50,                   # Moderate for H2
            shots_budget=512,       # Moderate for H2
            N_experiments=1         # Single experiment
        )
        print("✓ AdaptVQE instance created")
        
        # Force garbage collection before running
        gc.collect()
        
        print("Starting ADAPT-VQE run for H2...")
        print("="*50)
        
        # Run the algorithm
        result = adapt_vqe.run()
        
        print("="*50)
        print("✓ ADAPT-VQE for H2 completed successfully!")
        
        # Print results
        if hasattr(result, 'energy'):
            print(f"Final energy: {result.energy}")
        if hasattr(result, 'iterations'):
            print(f"Total iterations: {result.iterations}")
            
        print("\n🎉 H2 test successful! Memory optimization is working.")
        print("You can now try LiH with more conservative parameters.")
            
    except MemoryError as e:
        print(f"❌ Memory error: {e}")
        print("Even H2 is causing memory issues. Check your system resources.")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print("\n✅ H2 memory test passed!")
        print("Next steps:")
        print("  1. Try example_lih_ultra_optimized.py")
        print("  2. If that works, gradually increase parameters")
        print("  3. If it fails, use even more conservative settings")
    else:
        print("\n❌ H2 memory test failed!")
        print("Your system may need more RAM or the code has fundamental memory issues.") 