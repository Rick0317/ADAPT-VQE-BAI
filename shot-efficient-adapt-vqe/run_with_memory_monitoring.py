#!/usr/bin/env python3
"""
Simple wrapper to run example_lih.py with memory monitoring
"""

import os
import sys
import gc
import psutil
import time
import subprocess
from contextlib import contextmanager

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def monitor_memory():
    """Monitor memory usage during execution."""
    initial_memory = get_memory_usage()
    print(f"[Memory] Initial: {initial_memory:.1f} MB")
    
    start_time = time.time()
    
    try:
        # Import and run the original example
        print("[Memory] Importing modules...")
        import example_lih
        
        print("[Memory] Running original example_lih.py...")
        # The example will run automatically when imported due to __main__ block
        
        final_memory = get_memory_usage()
        peak_memory = max(get_memory_usage() for _ in range(10))  # Sample peak
        
        print(f"[Memory] Final: {final_memory:.1f} MB")
        print(f"[Memory] Peak: {peak_memory:.1f} MB")
        print(f"[Memory] Growth: {final_memory - initial_memory:.1f} MB")
        print(f"[Memory] Runtime: {time.time() - start_time:.1f} seconds")
        
    except MemoryError as e:
        print(f"[Memory] Memory error occurred: {e}")
        print("[Memory] Try reducing parameters in example_lih.py:")
        print("  - Reduce max_adapt_iter from 100 to 50")
        print("  - Reduce max_opt_iter from 100 to 50")
        print("  - Reduce shots_budget from 1024 to 512")
        print("  - Reduce N_experiments from 2 to 1")
        
    except Exception as e:
        print(f"[Memory] Error occurred: {e}")

def run_with_optimization():
    """Run with memory optimization settings."""
    print("Setting memory optimization environment variables...")
    
    # Set environment variables for memory optimization
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Force garbage collection before starting
    gc.collect()
    
    print("Running with memory monitoring...")
    monitor_memory()

if __name__ == '__main__':
    print("="*60)
    print("ADAPT-VQE Memory Monitoring Wrapper")
    print("="*60)
    
    run_with_optimization() 