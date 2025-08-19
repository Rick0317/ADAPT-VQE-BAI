#!/usr/bin/env python3
"""
Run ADAPT-VQE with Memory Limits
This script monitors memory usage and can kill the process before the system does.
"""

import os
import sys
import time
import signal
import psutil
import subprocess
import threading
from typing import Optional

class MemoryMonitor:
    def __init__(self, max_memory_mb: float = 2000, check_interval: float = 1.0):
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self.process = None
        self.monitoring = False
        self.memory_history = []
        
    def start_monitoring(self, process):
        """Start monitoring a process for memory usage."""
        self.process = process
        self.monitoring = True
        self.memory_history = []
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    def _monitor_loop(self):
        """Monitor memory usage in a loop."""
        while self.monitoring and self.process:
            try:
                # Get memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self.memory_history.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb
                })
                
                print(f"[Monitor] Memory: {memory_mb:.1f}MB / {self.max_memory_mb:.1f}MB")
                
                # Check if memory limit exceeded
                if memory_mb > self.max_memory_mb:
                    print(f"⚠️  Memory limit exceeded: {memory_mb:.1f}MB > {self.max_memory_mb:.1f}MB")
                    print("Terminating process to prevent SIGKILL...")
                    self.terminate_process()
                    break
                
                time.sleep(self.check_interval)
                
            except psutil.NoSuchProcess:
                print("[Monitor] Process terminated")
                break
            except Exception as e:
                print(f"[Monitor] Error: {e}")
                break
                
    def terminate_process(self):
        """Terminate the monitored process gracefully."""
        if self.process:
            try:
                print("Sending SIGTERM to process...")
                self.process.terminate()
                
                # Wait for graceful termination
                try:
                    self.process.wait(timeout=10)
                    print("Process terminated gracefully")
                except psutil.TimeoutExpired:
                    print("Process didn't terminate gracefully, sending SIGKILL...")
                    self.process.kill()
                    
            except Exception as e:
                print(f"Error terminating process: {e}")
                
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        
    def get_memory_summary(self):
        """Get memory usage summary."""
        if not self.memory_history:
            return {}
            
        memory_values = [entry['memory_mb'] for entry in self.memory_history]
        return {
            'peak_memory_mb': max(memory_values),
            'final_memory_mb': memory_values[-1],
            'total_samples': len(memory_values),
            'memory_history': self.memory_history
        }

def run_with_memory_monitoring(script_path: str, max_memory_mb: float = 2000):
    """Run a Python script with memory monitoring."""
    
    print(f"Running {script_path} with memory limit: {max_memory_mb}MB")
    
    # Set environment variables for memory optimization
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_MAIN_FREE'] = '1'
    env['MKL_MAIN_FREE'] = '1'
    
    # Start the process
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Create memory monitor
        monitor = MemoryMonitor(max_memory_mb=max_memory_mb)
        monitor.start_monitoring(process)
        
        # Read output in real-time
        print("="*60)
        print("PROCESS OUTPUT:")
        print("="*60)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        # Wait for process to complete
        return_code = process.wait()
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Get memory summary
        memory_summary = monitor.get_memory_summary()
        
        print("\n" + "="*60)
        print("PROCESS COMPLETED")
        print("="*60)
        print(f"Return code: {return_code}")
        print(f"Peak memory: {memory_summary.get('peak_memory_mb', 0):.1f}MB")
        print(f"Final memory: {memory_summary.get('final_memory_mb', 0):.1f}MB")
        
        if return_code == 0:
            print("✅ Process completed successfully!")
        else:
            print("❌ Process failed or was terminated")
            
        return return_code == 0, memory_summary
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        if 'process' in locals():
            process.terminate()
        return False, {}
        
    except Exception as e:
        print(f"❌ Error running process: {e}")
        return False, {}

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ADAPT-VQE with memory monitoring')
    parser.add_argument('script', help='Python script to run')
    parser.add_argument('--max-memory', type=float, default=2000,
                       help='Maximum memory limit in MB (default: 2000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.script):
        print(f"❌ Script not found: {args.script}")
        return 1
    
    success, memory_summary = run_with_memory_monitoring(args.script, args.max_memory)
    
    if success:
        print("\n🎉 Success! You can now try increasing the memory limit or parameters.")
    else:
        print("\n💥 Failed! Try:")
        print("  - Reducing the memory limit")
        print("  - Using a smaller molecule (H2 instead of LiH)")
        print("  - Using more conservative parameters")
        print("  - Running on a machine with more RAM")
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main()) 