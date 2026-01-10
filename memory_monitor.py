#!/usr/bin/env python3
"""
Memory monitoring script for ADAPT-VQE processes
Run this in a separate terminal to monitor memory usage
"""

import psutil
import time
import os
import sys
from datetime import datetime

def monitor_process_memory(pid=None, interval=5):
    """
    Monitor memory usage of a specific process or current process
    
    Args:
        pid: Process ID to monitor (None for current process)
        interval: Monitoring interval in seconds
    """
    if pid is None:
        pid = os.getpid()
    
    try:
        process = psutil.Process(pid)
        print(f"Monitoring process {pid} ({process.name()})")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        print(f"{'Time':<20} {'Memory (MB)':<15} {'CPU %':<10} {'Status':<10}")
        print("-" * 60)
        
        while True:
            try:
                # Get memory info
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Get CPU usage
                cpu_percent = process.cpu_percent()
                
                # Get status
                status = process.status()
                
                # Print current stats
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"{timestamp:<20} {memory_mb:<15.1f} {cpu_percent:<10.1f} {status:<10}")
                
                time.sleep(interval)
                
            except psutil.NoSuchProcess:
                print(f"Process {pid} no longer exists")
                break
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
                
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")
    except Exception as e:
        print(f"Error monitoring process: {e}")

def find_python_processes():
    """Find all Python processes"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'adapt_vqe' in cmdline:
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return python_processes

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            # List all Python processes
            processes = find_python_processes()
            if processes:
                print("Found Python processes:")
                for proc in processes:
                    print(f"PID: {proc['pid']}, Name: {proc['name']}")
                    print(f"  Command: {proc['cmdline']}")
                    print()
            else:
                print("No Python processes found")
            return
        else:
            # Monitor specific PID
            try:
                pid = int(sys.argv[1])
                interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
                monitor_process_memory(pid, interval)
            except ValueError:
                print("Usage: python memory_monitor.py [PID] [interval_seconds]")
                print("       python memory_monitor.py --list")
                return
    else:
        # Monitor current process
        monitor_process_memory()

if __name__ == "__main__":
    main() 