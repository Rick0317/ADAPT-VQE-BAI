#!/usr/bin/env python3
"""
Simple memory tracking utilities
"""

import os
import time
import gc
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class MemoryTracker:
    """Track memory usage over time"""
    
    def __init__(self, name="MemoryTracker"):
        self.name = name
        self.start_time = time.time()
        self.memory_log = []
        self.checkpoints = {}
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        else:
            return 0.0
    
    def log_memory(self, label=""):
        """Log current memory usage with timestamp"""
        memory_mb = self.get_memory_usage()
        timestamp = time.time() - self.start_time
        entry = {
            'timestamp': timestamp,
            'memory_mb': memory_mb,
            'label': label,
            'datetime': datetime.now().isoformat()
        }
        self.memory_log.append(entry)
        return memory_mb
    
    def checkpoint(self, name):
        """Create a named checkpoint"""
        memory_mb = self.get_memory_usage()
        self.checkpoints[name] = {
            'memory_mb': memory_mb,
            'timestamp': time.time() - self.start_time,
            'datetime': datetime.now().isoformat()
        }
        return memory_mb
    
    def print_summary(self):
        """Print memory usage summary"""
        if not self.memory_log:
            print("No memory data logged")
            return
            
        print(f"\n=== Memory Usage Summary for {self.name} ===")
        print(f"Duration: {time.time() - self.start_time:.1f} seconds")
        
        # Print checkpoints
        if self.checkpoints:
            print("\nCheckpoints:")
            for name, data in self.checkpoints.items():
                print(f"  {name}: {data['memory_mb']:.1f} MB at {data['timestamp']:.1f}s")
        
        # Print min/max/average
        memories = [entry['memory_mb'] for entry in self.memory_log]
        print(f"\nMemory Statistics:")
        print(f"  Min: {min(memories):.1f} MB")
        print(f"  Max: {max(memories):.1f} MB")
        print(f"  Average: {sum(memories)/len(memories):.1f} MB")
        
        # Print recent entries
        print(f"\nRecent Memory Log (last 5 entries):")
        for entry in self.memory_log[-5:]:
            print(f"  {entry['timestamp']:.1f}s: {entry['memory_mb']:.1f} MB - {entry['label']}")
    
    def force_gc(self):
        """Force garbage collection and log memory"""
        gc.collect()
        return self.log_memory("after_gc")

def quick_memory_check(label=""):
    """Quick memory check function"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory {label}: {memory_mb:.1f} MB")
        return memory_mb
    else:
        print(f"Memory {label}: psutil not available")
        return 0.0

def memory_before_after(func):
    """Decorator to track memory before and after function execution"""
    def wrapper(*args, **kwargs):
        if PSUTIL_AVAILABLE:
            before = quick_memory_check(f"before {func.__name__}")
            result = func(*args, **kwargs)
            after = quick_memory_check(f"after {func.__name__}")
            diff = after - before
            print(f"Memory difference for {func.__name__}: {diff:+.1f} MB")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

if __name__ == "__main__":
    # Example usage
    tracker = MemoryTracker("Example")
    
    print("Starting memory tracking...")
    tracker.log_memory("start")
    
    # Simulate some work
    import numpy as np
    tracker.checkpoint("before_array")
    
    large_array = np.random.random((1000, 1000))
    tracker.log_memory("after_creating_array")
    
    del large_array
    tracker.force_gc()
    
    tracker.print_summary() 