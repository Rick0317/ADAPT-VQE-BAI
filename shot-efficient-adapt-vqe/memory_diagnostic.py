#!/usr/bin/env python3
"""
Memory Diagnostic Tool for ADAPT-VQE
This script helps identify where memory usage spikes occur.
"""

import os
import gc
import psutil
import time
import sys
from contextlib import contextmanager

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

@contextmanager
def memory_checkpoint(stage_name):
    """Context manager to check memory at each stage."""
    before = get_memory_usage()
    print(f"[Memory] Before {stage_name}: RSS={before['rss']:.1f}MB, VMS={before['vms']:.1f}MB")
    
    try:
        yield
    finally:
        after = get_memory_usage()
        growth = after['rss'] - before['rss']
        print(f"[Memory] After {stage_name}: RSS={after['rss']:.1f}MB, VMS={after['vms']:.1f}MB, Growth={growth:.1f}MB")
        
        if growth > 100:  # More than 100MB growth
            print(f"⚠️  Large memory growth in {stage_name}: {growth:.1f}MB")

def diagnostic_test():
    """Run diagnostic tests to identify memory issues."""
    print("="*60)
    print("ADAPT-VQE Memory Diagnostic")
    print("="*60)
    
    # Set memory optimization
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    initial_memory = get_memory_usage()
    print(f"Initial memory: RSS={initial_memory['rss']:.1f}MB, VMS={initial_memory['vms']:.1f}MB")
    
    try:
        # Test 1: Basic imports
        with memory_checkpoint("basic imports"):
            import numpy as np
            import scipy
            print("✓ NumPy and SciPy imported")
        
        # Test 2: OpenFermion imports
        with memory_checkpoint("OpenFermion imports"):
            from openfermion import MolecularData
            from openfermion.transforms import jordan_wigner
            print("✓ OpenFermion imported")
        
        # Test 3: Qiskit imports
        with memory_checkpoint("Qiskit imports"):
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            print("✓ Qiskit imported")
        
        # Test 4: Local module imports
        with memory_checkpoint("local module imports"):
            from src.molecules import create_lih
            from src.pools import SD
            print("✓ Local modules imported")
        
        # Test 5: Create molecule
        with memory_checkpoint("molecule creation"):
            molecule = create_lih(1.5)
            print(f"✓ Molecule created: {molecule.description}")
        
        # Test 6: Create pool
        with memory_checkpoint("pool creation"):
            pool = SD(molecule)
            print(f"✓ Pool created: size={pool.size}")
        
        # Test 7: Import AdaptVQE
        with memory_checkpoint("AdaptVQE import"):
            from algorithms.adapt_vqe import AdaptVQE
            print("✓ AdaptVQE imported")
        
        # Test 8: Create AdaptVQE instance
        with memory_checkpoint("AdaptVQE instance creation"):
            adapt_vqe = AdaptVQE(
                pool=pool,
                molecule=molecule,
                max_adapt_iter=5,    # Very small for testing
                max_opt_iter=5,      # Very small for testing
                grad_threshold=1e-3,
                vrb=True,
                optimizer_method='l-bfgs-b',
                shots_assignment='uniform',
                k=10,                # Very small for testing
                shots_budget=64,     # Very small for testing
                N_experiments=1
            )
            print("✓ AdaptVQE instance created")
        
        # Test 9: Run one iteration
        with memory_checkpoint("single iteration"):
            print("Running single iteration...")
            # This might be where the memory explosion occurs
            try:
                # Try to access some internal attributes to see memory usage
                print(f"Pool size: {adapt_vqe.pool.size}")
                print(f"Number of qubits: {adapt_vqe.n}")
                print(f"Shots budget: {adapt_vqe.shots_budget}")
                
                # Don't actually run - just check if we can access attributes
                print("✓ AdaptVQE attributes accessible")
                
            except Exception as e:
                print(f"❌ Error accessing AdaptVQE attributes: {e}")
        
        final_memory = get_memory_usage()
        total_growth = final_memory['rss'] - initial_memory['rss']
        
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        print(f"Total memory growth: {total_growth:.1f}MB")
        print(f"Final RSS: {final_memory['rss']:.1f}MB")
        print(f"Final VMS: {final_memory['vms']:.1f}MB")
        
        if total_growth > 1000:
            print("⚠️  Very large memory growth detected!")
            print("The issue is likely in the AdaptVQE initialization or pool creation.")
        elif total_growth > 500:
            print("⚠️  Large memory growth detected!")
            print("Consider using smaller parameters or different pool types.")
        else:
            print("✅ Memory usage looks reasonable.")
            print("You can try running the actual algorithm.")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    diagnostic_test() 