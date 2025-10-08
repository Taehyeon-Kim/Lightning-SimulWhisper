#!/usr/bin/env python3
"""
MLX Memory Monitor - Tracks unified memory usage on Apple Silicon via MLX Metal APIs
This provides accurate memory tracking for MLX operations.

Usage as standalone:
    python mlx_memory_monitor.py --output mlx_memory_log.txt --interval 0.1

Usage as imported module:
    from mlx_memory_monitor import MemoryLogger
    logger = MemoryLogger("memory_log.txt")
    logger.start()
    # ... your code ...
    logger.stop()
"""

import mlx.core as mx
import time
import threading
import argparse
from datetime import datetime
import sys
import psutil
import os


def format_bytes(bytes_value):
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


class MemoryLogger:
    """Logger for MLX Metal memory usage."""
    
    def __init__(self, output_file, interval=0.1, console_output=True):
        self.output_file = output_file
        self.interval = interval
        self.console_output = console_output
        self.running = False
        self.thread = None
        self.start_time = None
        self.file_handle = None
        
    def _monitor_loop(self):
        """Internal monitoring loop."""
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        
        with open(self.output_file, 'w') as f:
            self.file_handle = f
            # Write header
            f.write("# MLX Metal Memory + Process RSS Usage Log\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# PID: {os.getpid()}\n")
            f.write("# Columns: timestamp(s), mlx_active(MB), mlx_peak(MB), mlx_cache(MB), rss(MB), vms(MB), mlx_active_human, mlx_peak_human, rss_human\n")
            f.write("#" + "-" * 120 + "\n")
            f.flush()
            
            if self.console_output:
                print(f"MLX Memory + RSS monitoring started")
                print(f"Logging to: {self.output_file}")
                print(f"Sample interval: {self.interval}s\n")
            
            while self.running:
                try:
                    # Get MLX Metal memory stats
                    active_memory = mx.metal.get_active_memory()
                    peak_memory = mx.metal.get_peak_memory()
                    cache_memory = mx.metal.get_cache_memory()
                    
                    # Get process memory stats
                    mem_info = process.memory_info()
                    rss = mem_info.rss
                    vms = mem_info.vms
                    
                    active_mb = active_memory / (1024 * 1024)
                    peak_mb = peak_memory / (1024 * 1024)
                    cache_mb = cache_memory / (1024 * 1024)
                    rss_mb = rss / (1024 * 1024)
                    vms_mb = vms / (1024 * 1024)
                    
                    elapsed = time.time() - self.start_time
                    
                    # Write to file
                    line = (f"{elapsed:.2f}\t{active_mb:.2f}\t{peak_mb:.2f}\t{cache_mb:.2f}\t{rss_mb:.2f}\t{vms_mb:.2f}\t"
                           f"{format_bytes(active_memory)}\t{format_bytes(peak_memory)}\t{format_bytes(rss)}\n")
                    f.write(line)
                    f.flush()
                    
                    # Print to console
                    if self.console_output:
                        print(f"\rTime: {elapsed:.1f}s | MLX Active: {format_bytes(active_memory)} | "
                              f"MLX Peak: {format_bytes(peak_memory)} | RSS: {format_bytes(rss)}", 
                              end='', flush=True)
                    
                    time.sleep(self.interval)
                    
                except Exception as e:
                    if self.console_output:
                        print(f"\nError during monitoring: {e}")
                    break
            
            # Write footer
            if self.console_output:
                print("\n\nMonitoring stopped.")
            f.write(f"# Monitoring stopped at {time.time() - self.start_time:.2f}s\n")
            f.write(f"# Final peak memory: {format_bytes(mx.metal.get_peak_memory())}\n")
    
    def start(self):
        """Start monitoring in a background thread."""
        if self.running:
            print("Monitoring is already running!")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.stop()


def main():
    parser = argparse.ArgumentParser(
        description='Monitor MLX Metal memory usage in real-time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script monitors the unified memory used by MLX operations on Apple Silicon
as well as the process's RSS memory.
It tracks:
  - MLX Active Memory: Currently allocated unified memory in use by MLX
  - MLX Peak Memory: Maximum MLX memory allocated during the session
  - MLX Cache Memory: Memory in MLX's cache
  - RSS: Process resident set size (process's own memory)
  - VMS: Virtual memory size

Example:
  python mlx_memory_monitor.py --output mlx_memory.log --interval 0.1
  
To use in your own script:
  from mlx_memory_monitor import MemoryLogger
  
  logger = MemoryLogger("memory.log")
  logger.start()
  # ... your MLX code ...
  logger.stop()
  
Or with context manager:
  with MemoryLogger("memory.log"):
      # ... your MLX code ...
        """
    )
    
    parser.add_argument('--output', '-o', type=str, default='mlx_memory_log.txt',
                        help='Output file for memory log (default: mlx_memory_log.txt)')
    parser.add_argument('--interval', '-i', type=float, default=0.1,
                        help='Sampling interval in seconds (default: 0.1)')
    parser.add_argument('--no-console', action='store_true',
                        help='Disable console output (only write to file)')
    
    args = parser.parse_args()
    
    logger = MemoryLogger(args.output, args.interval, console_output=not args.no_console)
    
    try:
        logger.start()
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.stop()


if __name__ == '__main__':
    main()

