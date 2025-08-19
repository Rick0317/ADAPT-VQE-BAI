#!/usr/bin/env python3
"""
Memory-Optimized ADAPT-VQE Runner
This script provides memory analysis and optimization for running ADAPT-VQE
without modifying the main algorithms.
"""

import os
import sys
import gc
import psutil
import time
import numpy as np
from contextlib import contextmanager
import warnings
from typing import Optional, Dict, Any

# Suppress warnings to reduce memory overhead
warnings.filterwarnings('ignore')

class MemoryMonitor:
    """Monitor and analyze memory usage during ADAPT-VQE execution."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.memory_history = []
        self.peak_memory = 0
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': self.process.memory_percent()
        }
    
    def log_memory(self, stage: str = ""):
        """Log current memory usage."""
        memory = self.get_memory_usage()
        self.memory_history.append({
            'stage': stage,
            'timestamp': time.time(),
            'memory': memory
        })
        
        if memory['rss'] > self.peak_memory:
            self.peak_memory = memory['rss']
            
        if self.verbose:
            print(f"[Memory] {stage}: RSS={memory['rss']:.1f}MB, "
                  f"VMS={memory['vms']:.1f}MB, {memory['percent']:.1f}%")
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        collected = gc.collect()
        if self.verbose:
            print(f"[Memory] Garbage collection freed {collected} objects")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_history:
            return {}
            
        return {
            'peak_memory_mb': self.peak_memory,
            'final_memory_mb': self.memory_history[-1]['memory']['rss'],
            'total_stages': len(self.memory_history),
            'memory_history': self.memory_history
        }

class MemoryOptimizer:
    """Optimize memory usage for ADAPT-VQE execution."""
    
    def __init__(self, max_memory_mb: Optional[float] = None):
        self.max_memory_mb = max_memory_mb
        self.monitor = MemoryMonitor()
        
    @contextmanager
    def memory_optimization_context(self):
        """Context manager for memory optimization."""
        # Set environment variables for memory optimization
        original_env = {}
        
        # Optimize NumPy memory usage
        original_env['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
        original_env['OPENBLAS_NUM_THREADS'] = os.environ.get('OPENBLAS_NUM_THREADS', '1')
        original_env['MKL_NUM_THREADS'] = os.environ.get('MKL_NUM_THREADS', '1')
        
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        # Optimize NumPy settings
        np.set_printoptions(precision=6, suppress=True)
        
        try:
            self.monitor.log_memory("Starting optimization context")
            yield self.monitor
            
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is not None:
                    os.environ[key] = value
                else:
                    os.environ.pop(key, None)
            
            self.monitor.log_memory("Ending optimization context")
    
    def check_memory_limit(self) -> bool:
        """Check if current memory usage exceeds limit."""
        if self.max_memory_mb is None:
            return True
            
        current_memory = self.monitor.get_memory_usage()['rss']
        return current_memory < self.max_memory_mb

def run_adapt_vqe_with_memory_optimization(
    pool_type: str = 'SD',
    molecule_name: str = 'lih',
    max_memory_mb: Optional[float] = None,
    **adapt_vqe_kwargs
):
    """
    Run ADAPT-VQE with memory optimization and monitoring.
    
    Args:
        pool_type: Type of operator pool ('SD', 'QE', 'SingletGSD', 'CEO', 'DVG_CEO')
        molecule_name: Name of molecule ('lih', 'h2', 'beh2')
        max_memory_mb: Maximum memory limit in MB (None for no limit)
        **adapt_vqe_kwargs: Additional arguments for AdaptVQE
    """
    
    optimizer = MemoryOptimizer(max_memory_mb)
    
    with optimizer.memory_optimization_context() as monitor:
        
        # Import modules only when needed to reduce initial memory footprint
        monitor.log_memory("Importing modules")
        
        # Import pool based on type
        if pool_type == 'SD':
            from src.pools import SD as PoolClass
        elif pool_type == 'QE':
            from src.pools import QE as PoolClass
        elif pool_type == 'SingletGSD':
            from src.pools import SingletGSD as PoolClass
        elif pool_type == 'CEO':
            from src.pools import CEO as PoolClass
        elif pool_type == 'DVG_CEO':
            from src.pools import DVG_CEO as PoolClass
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
        
        # Import molecule creation function
        if molecule_name == 'lih':
            from src.molecules import create_lih as create_molecule
        elif molecule_name == 'h2':
            from src.molecules import create_h2 as create_molecule
        elif molecule_name == 'beh2':
            from src.molecules import create_beh2 as create_molecule
        else:
            raise ValueError(f"Unknown molecule: {molecule_name}")
        
        # Import AdaptVQE
        from algorithms.adapt_vqe import AdaptVQE
        
        monitor.log_memory("Modules imported")
        monitor.force_garbage_collection()
        
        # Create molecule
        monitor.log_memory("Creating molecule")
        r = 1.5  # Bond distance
        molecule = create_molecule(r)
        monitor.log_memory("Molecule created")
        
        # Create pool
        monitor.log_memory("Creating operator pool")
        pool = PoolClass(molecule)
        print(f"Pool size: {pool.size}")
        monitor.log_memory("Pool created")
        
        # Set default parameters for memory optimization
        default_params = {
            'max_adapt_iter': 50,  # Reduced from 100
            'max_opt_iter': 50,    # Reduced from 100
            'grad_threshold': 1e-4,
            'vrb': True,
            'optimizer_method': 'l-bfgs-b',
            'shots_assignment': 'uniform',
            'k': 50,               # Reduced from 100
            'shots_budget': 512,   # Reduced from 1024
            'N_experiments': 1,    # Reduced from 2
        }
        
        # Update with user-provided parameters
        default_params.update(adapt_vqe_kwargs)
        
        # Create and run AdaptVQE
        monitor.log_memory("Creating AdaptVQE instance")
        adapt_vqe = AdaptVQE(
            pool=pool,
            molecule=molecule,
            **default_params
        )
        monitor.log_memory("AdaptVQE instance created")
        
        # Run the algorithm
        monitor.log_memory("Starting ADAPT-VQE run")
        try:
            result = adapt_vqe.run()
            monitor.log_memory("ADAPT-VQE completed successfully")
            return result, monitor.get_memory_summary()
            
        except MemoryError as e:
            monitor.log_memory("Memory error occurred")
            print(f"Memory error: {e}")
            print("Try reducing max_adapt_iter, max_opt_iter, or shots_budget")
            return None, monitor.get_memory_summary()
            
        except Exception as e:
            monitor.log_memory("Error occurred")
            print(f"Error: {e}")
            return None, monitor.get_memory_summary()

def analyze_memory_usage(memory_summary: Dict[str, Any]):
    """Analyze and print memory usage summary."""
    if not memory_summary:
        print("No memory data available")
        return
    
    print("\n" + "="*50)
    print("MEMORY USAGE ANALYSIS")
    print("="*50)
    
    print(f"Peak Memory Usage: {memory_summary['peak_memory_mb']:.1f} MB")
    print(f"Final Memory Usage: {memory_summary['final_memory_mb']:.1f} MB")
    print(f"Total Stages Monitored: {memory_summary['total_stages']}")
    
    # Analyze memory growth
    if len(memory_summary['memory_history']) > 1:
        initial_memory = memory_summary['memory_history'][0]['memory']['rss']
        final_memory = memory_summary['memory_history'][-1]['memory']['rss']
        memory_growth = final_memory - initial_memory
        
        print(f"\nMemory Growth: {memory_growth:.1f} MB")
        
        if memory_growth > 1000:  # More than 1GB
            print("⚠️  High memory growth detected!")
            print("Recommendations:")
            print("  - Reduce max_adapt_iter")
            print("  - Reduce max_opt_iter")
            print("  - Reduce shots_budget")
            print("  - Use smaller operator pools")
    
    # Show memory usage by stage
    print(f"\nMemory Usage by Stage:")
    for entry in memory_summary['memory_history']:
        stage = entry['stage']
        memory = entry['memory']['rss']
        print(f"  {stage}: {memory:.1f} MB")

def main():
    """Main function to run memory-optimized ADAPT-VQE."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ADAPT-VQE with memory optimization')
    parser.add_argument('--pool', type=str, default='SD', 
                       choices=['SD', 'QE', 'SingletGSD', 'CEO', 'DVG_CEO'],
                       help='Operator pool type')
    parser.add_argument('--molecule', type=str, default='lih',
                       choices=['lih', 'h2', 'beh2'],
                       help='Molecule to simulate')
    parser.add_argument('--max-memory', type=float, default=None,
                       help='Maximum memory limit in MB')
    parser.add_argument('--max-adapt-iter', type=int, default=50,
                       help='Maximum ADAPT iterations')
    parser.add_argument('--max-opt-iter', type=int, default=50,
                       help='Maximum optimization iterations')
    parser.add_argument('--shots-budget', type=int, default=512,
                       help='Shots budget')
    
    args = parser.parse_args()
    
    print(f"Running ADAPT-VQE for {args.molecule} with {args.pool} pool")
    print(f"Memory limit: {args.max_memory} MB" if args.max_memory else "No memory limit")
    
    # Run with memory optimization
    result, memory_summary = run_adapt_vqe_with_memory_optimization(
        pool_type=args.pool,
        molecule_name=args.molecule,
        max_memory_mb=args.max_memory,
        max_adapt_iter=args.max_adapt_iter,
        max_opt_iter=args.max_opt_iter,
        shots_budget=args.shots_budget
    )
    
    # Analyze memory usage
    analyze_memory_usage(memory_summary)
    
    if result:
        print(f"\nADAPT-VQE completed successfully!")
        print(f"Final energy: {result.energy if hasattr(result, 'energy') else 'N/A'}")
    else:
        print(f"\nADAPT-VQE failed or was interrupted")

if __name__ == '__main__':
    main() 