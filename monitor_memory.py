#!/usr/bin/env python3
"""
Memory monitoring script for tracking unified memory usage on Apple Silicon.
Monitors a process by name and logs memory usage to a file in real-time.

Usage:
    python monitor_memory.py --process-name python --output memory_log.txt --interval 0.1
"""

import psutil
import time
import argparse
import sys
from datetime import datetime


def find_process_by_name(process_name):
    """Find all processes matching the given name."""
    matching_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process_name.lower() in proc.info['name'].lower():
                matching_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return matching_processes


def find_process_by_cmdline(search_term):
    """Find process by searching in command line arguments."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any(search_term in arg for arg in cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def format_bytes(bytes_value):
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def monitor_memory(pid, output_file, interval=0.1):
    """Monitor memory usage of a process and log to file."""
    try:
        process = psutil.Process(pid)
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("# Memory monitoring log\n")
            f.write(f"# Process: {process.name()} (PID: {pid})\n")
            f.write(f"# Command: {' '.join(process.cmdline())}\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Columns: timestamp(s), rss(MB), vms(MB), rss_human, vms_human\n")
            f.write("#" + "-" * 80 + "\n")
            f.flush()
            
            start_time = time.time()
            print(f"Monitoring process {pid} ({process.name()})")
            print(f"Logging to: {output_file}")
            print(f"Sample interval: {interval}s")
            print("Press Ctrl+C to stop monitoring\n")
            
            try:
                while True:
                    try:
                        # Get memory info
                        mem_info = process.memory_info()
                        rss_mb = mem_info.rss / (1024 * 1024)  # Resident Set Size in MB
                        vms_mb = mem_info.vms / (1024 * 1024)  # Virtual Memory Size in MB
                        
                        elapsed = time.time() - start_time
                        
                        # Write to file
                        line = f"{elapsed:.2f}\t{rss_mb:.2f}\t{vms_mb:.2f}\t{format_bytes(mem_info.rss)}\t{format_bytes(mem_info.vms)}\n"
                        f.write(line)
                        f.flush()
                        
                        # Print to console (optional, can be removed for less overhead)
                        print(f"\rTime: {elapsed:.1f}s | RSS: {format_bytes(mem_info.rss)} | VMS: {format_bytes(mem_info.vms)}", end='', flush=True)
                        
                        time.sleep(interval)
                        
                    except psutil.NoSuchProcess:
                        print(f"\n\nProcess {pid} has terminated.")
                        f.write(f"# Process terminated at {time.time() - start_time:.2f}s\n")
                        break
                        
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped by user.")
                f.write(f"# Monitoring stopped at {time.time() - start_time:.2f}s\n")
                
    except psutil.NoSuchProcess:
        print(f"Error: Process with PID {pid} not found.")
        sys.exit(1)
    except psutil.AccessDenied:
        print(f"Error: Access denied to process {pid}. Try running with sudo.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor memory usage of a process in real-time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor by PID
  python monitor_memory.py --pid 12345 --output memory.log
  
  # Monitor by searching for process name (will prompt if multiple found)
  python monitor_memory.py --process-name python --output memory.log
  
  # Monitor by searching command line (e.g., script name)
  python monitor_memory.py --cmdline simulstreaming_whisper.py --output memory.log --interval 0.05
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pid', type=int, help='Process ID to monitor')
    group.add_argument('--process-name', type=str, help='Process name to search for')
    group.add_argument('--cmdline', type=str, help='Search for process by command line argument')
    
    parser.add_argument('--output', '-o', type=str, default='memory_log.txt',
                        help='Output file for memory log (default: memory_log.txt)')
    parser.add_argument('--interval', '-i', type=float, default=0.1,
                        help='Sampling interval in seconds (default: 0.1)')
    
    args = parser.parse_args()
    
    if args.pid:
        pid = args.pid
    elif args.cmdline:
        proc = find_process_by_cmdline(args.cmdline)
        if not proc:
            print(f"Error: No process found with '{args.cmdline}' in command line.")
            sys.exit(1)
        pid = proc.pid
        print(f"Found process: {proc.name()} (PID: {pid})")
    else:  # args.process_name
        processes = find_process_by_name(args.process_name)
        if not processes:
            print(f"Error: No process found with name matching '{args.process_name}'")
            sys.exit(1)
        
        if len(processes) == 1:
            pid = processes[0].pid
        else:
            print(f"Found {len(processes)} matching processes:")
            for i, proc in enumerate(processes, 1):
                try:
                    cmdline = ' '.join(proc.cmdline())
                    print(f"  {i}. PID {proc.pid}: {cmdline[:100]}")
                except:
                    print(f"  {i}. PID {proc.pid}: (unable to read cmdline)")
            
            choice = input("\nEnter the number of the process to monitor: ")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(processes):
                    pid = processes[idx].pid
                else:
                    print("Invalid selection.")
                    sys.exit(1)
            except ValueError:
                print("Invalid input.")
                sys.exit(1)
    
    monitor_memory(pid, args.output, args.interval)


if __name__ == '__main__':
    main()

