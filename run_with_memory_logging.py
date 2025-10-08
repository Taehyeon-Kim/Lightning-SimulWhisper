#!/usr/bin/env python3
"""
Wrapper script to run simulstreaming_whisper.py with MLX memory logging.

Usage:
    python run_with_memory_logging.py test.mp3 --language ko --vac --beams 3 -l CRITICAL --model_path mlx_model --cif_ckpt_path cif_model/small.npz --audio_min_len 1.0
"""

import sys
import subprocess
from mlx_memory_monitor import MemoryLogger

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_with_memory_logging.py <args for simulstreaming_whisper.py>")
        sys.exit(1)
    
    # Start memory logging
    logger = MemoryLogger("mlx_memory_log.txt", interval=0.1, console_output=True)
    logger.start()
    
    # Build command
    cmd = [sys.executable, "simulstreaming_whisper.py"] + sys.argv[1:]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        # Run the main script
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    finally:
        logger.stop()
        print(f"\nMemory log saved to: mlx_memory_log.txt")


if __name__ == '__main__':
    main()

